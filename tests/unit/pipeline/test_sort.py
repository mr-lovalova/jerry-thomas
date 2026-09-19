import pickle
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

import jerrythomas.pipelines.sort as sort_module
from jerrythomas.pipelines.sort import (
    SortedRuns,
    SortProgress,
    batch_sort,
    merge_sort_runs,
    write_sort_runs,
)


@dataclass
class SortItem:
    value: int


@dataclass
class StableSortItem:
    value: int
    position: int


@dataclass
class CompressibleSortItem:
    value: int
    padding: str


class UnpickleableSortItem:
    def __init__(self, value: int) -> None:
        self.value = value
        self.callback = lambda: value


def _serialized_size(item) -> int:
    return len(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL))


def test_batch_sort_orders_items_across_buffered_runs() -> None:
    items = [SortItem(3), SortItem(1), SortItem(4), SortItem(2)]
    buffer_bytes = sum(_serialized_size(item) for item in items[:2])

    ordered = list(
        batch_sort(items, buffer_bytes=buffer_bytes, key=lambda item: item.value)
    )

    assert [item.value for item in ordered] == [1, 2, 3, 4]


def test_batch_sort_orders_items_across_merge_passes(monkeypatch) -> None:
    monkeypatch.setattr(sort_module, "_MAX_OPEN_RUNS", 2)
    items = [SortItem(5), SortItem(1), SortItem(4), SortItem(2), SortItem(3)]

    ordered = list(batch_sort(items, buffer_bytes=1, key=lambda item: item.value))

    assert [item.value for item in ordered] == [1, 2, 3, 4, 5]


def test_spill_run_is_compressed_and_readable(tmp_path) -> None:
    items = [
        CompressibleSortItem(value=index, padding="repeated-value-" * 100)
        for index in range(100)
    ]
    serialized = [
        (item.value, pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL))
        for item in items
    ]

    path = sort_module._write_serialized_run(tmp_path, 0, serialized)

    assert path.name == "run-0.pickle.gz"
    assert list(sort_module._read_run(path)) == items
    assert path.stat().st_size < sum(len(payload) for _, payload in serialized) // 4


def test_batch_sort_reports_buffered_and_merged_work(monkeypatch) -> None:
    monkeypatch.setattr(sort_module, "_MAX_OPEN_RUNS", 2)
    monkeypatch.setattr(sort_module, "_MERGE_PROGRESS_INTERVAL", 1)
    progress = Mock(wraps=SortProgress())

    ordered = list(
        batch_sort(
            [SortItem(3), SortItem(1), SortItem(2)],
            buffer_bytes=1,
            key=lambda item: item.value,
            progress=progress,
        )
    )

    assert [item.value for item in ordered] == [1, 2, 3]
    progress.spilling.assert_any_call(3, 3)
    progress.merging.assert_any_call(3, 3, 1)
    progress.emitting.assert_called_once_with(3)


def test_sort_progress_reports_spilling_before_a_run_is_written(
    tmp_path, monkeypatch
) -> None:
    progress = SortProgress()
    write_run = sort_module._write_serialized_run
    snapshots = []

    def observe_write(*args):
        snapshots.append(progress.snapshot(0))
        return write_run(*args)

    monkeypatch.setattr(sort_module, "_write_serialized_run", observe_write)
    runs = write_sort_runs([2, 1], 1, lambda item: item, tmp_path, progress)

    assert [snapshot.phase for snapshot in snapshots] == ["spilling", "spilling"]
    assert [snapshot.completed for snapshot in snapshots] == [0, 1]
    assert progress.snapshot(0).completed == runs.rows == 2


def test_sort_progress_uses_node_output_count_while_emitting() -> None:
    progress = SortProgress()

    progress.spilling(12, 3)
    snapshot = progress.snapshot(0)
    assert snapshot.completed == 12
    assert snapshot.phase == "spilling"
    assert snapshot.detail == "3 spill runs"

    progress.emitting(12)
    snapshot = progress.snapshot(5)
    assert snapshot.completed == 5
    assert snapshot.total == 12
    assert snapshot.phase == "emitting"


def test_batch_sort_resets_reused_progress() -> None:
    progress = SortProgress()

    assert list(batch_sort([2, 1], 1, lambda item: item, progress=progress)) == [1, 2]
    assert progress.snapshot(2).phase == "emitting"

    assert list(batch_sort([], 1, lambda item: item, progress=progress)) == []
    assert progress.snapshot(0).phase == "reading"


def test_batch_sort_preserves_input_order_across_merge_passes(monkeypatch) -> None:
    monkeypatch.setattr(sort_module, "_MAX_OPEN_RUNS", 2)
    items = [StableSortItem(value=1, position=index) for index in reversed(range(6))]
    buffer_bytes = sum(_serialized_size(item) for item in items[:2])

    ordered = list(
        batch_sort(items, buffer_bytes=buffer_bytes, key=lambda item: item.value)
    )

    assert [item.position for item in ordered] == list(reversed(range(6)))


@pytest.mark.parametrize(
    ("budget_adjustment", "spills"),
    [(0, False), (-1, True)],
)
def test_batch_sort_honors_buffer_boundary(
    tmp_path,
    budget_adjustment,
    spills,
) -> None:
    items = [SortItem(2), SortItem(1)]
    buffer_bytes = sum(_serialized_size(item) for item in items) + budget_adjustment
    spill_dir = tmp_path / "sort"

    ordered = list(
        batch_sort(
            items,
            buffer_bytes=buffer_bytes,
            key=lambda item: item.value,
            spill_dir=spill_dir,
        )
    )

    assert [item.value for item in ordered] == [1, 2]
    assert spill_dir.exists() is spills
    if spills:
        assert list(spill_dir.iterdir()) == []


def test_batch_sort_accepts_one_item_larger_than_buffer(tmp_path) -> None:
    item = SortItem(1)
    spill_dir = tmp_path / "sort"

    assert list(
        batch_sort(
            [item],
            buffer_bytes=_serialized_size(item) - 1,
            key=lambda item: item.value,
            spill_dir=spill_dir,
        )
    ) == [item]
    assert not spill_dir.exists()


def test_batch_sort_serializes_each_item_once_before_merge(monkeypatch) -> None:
    items = [SortItem(3), SortItem(1), SortItem(2)]
    dumps = pickle.dumps
    calls = 0

    def count_dumps(*args, **kwargs):
        nonlocal calls
        calls += 1
        return dumps(*args, **kwargs)

    monkeypatch.setattr(sort_module.pickle, "dumps", count_dumps)

    ordered = list(batch_sort(items, buffer_bytes=1, key=lambda item: item.value))

    assert [item.value for item in ordered] == [1, 2, 3]
    assert calls == len(items)


def test_batch_sort_requires_pickleable_items() -> None:
    with pytest.raises(TypeError, match="pickle-serializable"):
        list(
            batch_sort(
                [UnpickleableSortItem(1)],
                buffer_bytes=1_000_000,
                key=lambda item: item.value,
            )
        )


def test_batch_sort_accepts_empty_input(monkeypatch, tmp_path) -> None:
    spill_dir = tmp_path / "sort"
    monkeypatch.setattr(
        sort_module.pickle,
        "dumps",
        lambda *_args, **_kwargs: pytest.fail("empty input was serialized"),
    )

    assert (
        list(
            batch_sort(
                [],
                buffer_bytes=1,
                key=lambda item: item,
                spill_dir=spill_dir,
            )
        )
        == []
    )
    assert not spill_dir.exists()


@pytest.mark.parametrize("buffer_bytes", [0, -1])
def test_batch_sort_rejects_invalid_buffer_size(buffer_bytes) -> None:
    with pytest.raises(ValueError, match="buffer_bytes must be at least 1"):
        list(batch_sort([], buffer_bytes=buffer_bytes, key=lambda item: item))


@pytest.mark.parametrize("buffer_bytes", [1, 10_000])
def test_separately_written_runs_merge_in_original_stable_order(
    tmp_path, monkeypatch, buffer_bytes
) -> None:
    monkeypatch.setattr(sort_module, "_MAX_OPEN_RUNS", 2)
    groups = [
        [
            StableSortItem(value=value, position=stream * 3 + index)
            for index, value in enumerate((1, 0, 1))
        ]
        for stream in range(3)
    ]
    written = []
    # Completion order differs from the order that defines equal-key precedence.
    for stream in (2, 0, 1):
        directory = tmp_path / str(stream)
        directory.mkdir()
        runs = write_sort_runs(
            groups[stream], buffer_bytes, lambda item: item.value, directory
        )
        written.append((stream, runs))
    written.sort(key=lambda entry: entry[0])
    combined = SortedRuns(
        tuple(path for _, runs in written for path in runs.paths),
        sum(runs.rows for _, runs in written),
    )

    actual = list(merge_sort_runs(combined, lambda item: item.value, tmp_path))

    expected = sorted(
        [item for group in groups for item in group], key=lambda item: item.value
    )
    assert actual == expected


def test_empty_sorted_runs_need_no_files(tmp_path) -> None:
    runs = write_sort_runs([], 1, lambda item: item, tmp_path)

    assert runs == SortedRuns((), 0)
    assert list(merge_sort_runs(runs, lambda item: item, tmp_path)) == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("buffer_bytes", [0, -1])
def test_writing_sorted_runs_rejects_invalid_buffer_size(
    tmp_path, buffer_bytes
) -> None:
    with pytest.raises(ValueError, match="buffer_bytes must be at least 1"):
        write_sort_runs([], buffer_bytes, lambda item: item, tmp_path)


def test_merge_rejects_incomplete_output_across_merge_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(sort_module, "_MAX_OPEN_RUNS", 2)
    runs = write_sort_runs([5, 4, 3, 2, 1], 1, lambda item: item, tmp_path)
    runs.paths[1].unlink()
    incomplete = SortedRuns(
        tuple(path for path in runs.paths if path.exists()), runs.rows
    )

    with pytest.raises(ValueError, match="expected 5 rows, got 4"):
        list(merge_sort_runs(incomplete, lambda item: item, tmp_path))


def test_closing_merge_closes_run_readers_but_leaves_files_to_owner(
    tmp_path, monkeypatch
) -> None:
    runs = write_sort_runs([3, 2, 1], 1, lambda item: item, tmp_path)
    readers = []
    read_run = sort_module._read_run

    def track_reader(path):
        reader = read_run(path)
        readers.append(reader)
        return reader

    monkeypatch.setattr(sort_module, "_read_run", track_reader)
    merged = merge_sort_runs(runs, lambda item: item, tmp_path)
    assert next(merged) == 1
    merged.close()

    assert len(readers) == 3
    assert all(reader.gi_frame is None for reader in readers)
    assert all(path.exists() for path in runs.paths)

import gzip
import heapq
import pickle
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from typing import Any, TypeVar

from jerrythomas.execution.events import ProgressSnapshot
from jerrythomas.services.temp_cleanup import sort_spill_directory

T = TypeVar("T")
_BufferedItem = tuple[Any, bytes]
_MAX_OPEN_RUNS = 64
_MERGE_PROGRESS_INTERVAL = 100_000
_SPILL_COMPRESSION_LEVEL = 1


@dataclass(frozen=True)
class SortedRuns:
    paths: tuple[Path, ...]
    rows: int


@dataclass(frozen=True)
class _SortProgressState:
    progress: ProgressSnapshot
    uses_output_items: bool = False


class SortProgress:
    def __init__(self) -> None:
        self._state = _SortProgressState(ProgressSnapshot(completed=0, phase="reading"))

    def reading(self, completed: int) -> None:
        self._state = _SortProgressState(
            ProgressSnapshot(completed=completed, phase="reading")
        )

    def spilling(self, completed: int, runs: int) -> None:
        self._state = _SortProgressState(
            ProgressSnapshot(
                completed=completed,
                phase="spilling",
                detail=f"{runs} spill runs",
            )
        )

    def merging(self, completed: int, total: int, pass_id: int) -> None:
        self._state = _SortProgressState(
            ProgressSnapshot(
                completed=completed,
                total=total,
                phase="merging",
                detail=f"pass {pass_id}",
            )
        )

    def emitting(self, total: int) -> None:
        self._state = _SortProgressState(
            ProgressSnapshot(completed=0, total=total, phase="emitting"),
            uses_output_items=True,
        )

    def snapshot(self, output_items: int) -> ProgressSnapshot:
        state = self._state
        return (
            replace(state.progress, completed=output_items)
            if state.uses_output_items
            else state.progress
        )


def _sorted_runs(
    iterable: Iterable[T],
    buffer_bytes: int,
    key: Callable[[T], Any],
) -> Iterator[tuple[list[_BufferedItem], bool]]:
    batch: list[_BufferedItem] = []
    batch_bytes = 0
    for item in iterable:
        sort_key = key(item)
        try:
            payload = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, TypeError, AttributeError) as exc:
            raise TypeError("batch_sort requires pickle-serializable items") from exc
        payload_bytes = len(payload)
        if batch and batch_bytes + payload_bytes > buffer_bytes:
            batch.sort(key=lambda entry: entry[0])
            yield batch, False
            batch = []
            batch_bytes = 0
        batch.append((sort_key, payload))
        batch_bytes += payload_bytes
    if batch:
        batch.sort(key=lambda entry: entry[0])
        yield batch, True


def batch_sort(
    iterable: Iterable[T],
    buffer_bytes: int,
    key: Callable[[T], Any],
    spill_dir: Path | None = None,
    progress: SortProgress | None = None,
) -> Generator[T, None, None]:
    """Stably sort serialized values, spilling runs when the buffer is exceeded."""
    if buffer_bytes < 1:
        raise ValueError("buffer_bytes must be at least 1")
    if progress is None:
        progress = SortProgress()
    progress.reading(0)
    runs = _sorted_runs(iterable, buffer_bytes, key)

    try:
        first_run, final = next(runs)
    except StopIteration:
        return
    progress.reading(len(first_run))

    if final:
        progress.emitting(len(first_run))
        for _, payload in first_run:
            yield pickle.loads(payload)
        return

    with sort_spill_directory(spill_dir) as temp_dir:
        written = _write_runs(chain(((first_run, False),), runs), temp_dir, progress)
        yield from merge_sort_runs(written, key, temp_dir, progress)


def write_sort_runs(
    iterable: Iterable[T],
    buffer_bytes: int,
    key: Callable[[T], Any],
    directory: Path,
    progress: SortProgress | None = None,
) -> SortedRuns:
    """Snapshot and sort runs in a caller-owned directory, even if input fits RAM."""
    if buffer_bytes < 1:
        raise ValueError("buffer_bytes must be at least 1")
    if progress is None:
        progress = SortProgress()
    progress.reading(0)
    return _write_runs(_sorted_runs(iterable, buffer_bytes, key), directory, progress)


def _write_runs(
    runs: Iterable[tuple[list[_BufferedItem], bool]],
    directory: Path,
    progress: SortProgress,
) -> SortedRuns:
    paths: list[Path] = []
    rows = 0
    for run, _ in runs:
        paths.append(_write_serialized_run(directory, len(paths), run))
        rows += len(run)
        run.clear()
        progress.spilling(rows, len(paths))
    return SortedRuns(tuple(paths), rows)


def merge_sort_runs(
    runs: SortedRuns,
    key: Callable[[T], Any],
    directory: Path,
    progress: SortProgress | None = None,
) -> Generator[T, None, None]:
    """Merge runs in stable path order; the caller owns their directory lifetime."""
    if progress is None:
        progress = SortProgress()
    paths = list(runs.paths)
    pass_id = 0
    while len(paths) > _MAX_OPEN_RUNS:
        pass_id += 1
        paths = _merge_pass(directory, pass_id, paths, key, runs.rows, progress)

    progress.emitting(runs.rows)
    merged = _merge_runs(paths, key)
    count = 0
    try:
        for item in merged:
            count += 1
            yield item
    finally:
        merged.close()
    if count != runs.rows:
        raise ValueError(
            f"Sorted runs returned incomplete output: expected {runs.rows} rows, "
            f"got {count}."
        )


def _write_serialized_run(
    temp_dir: Path,
    run_id: int,
    items: Iterable[_BufferedItem],
) -> Path:
    path = temp_dir / f"run-{run_id}.pickle.gz"
    with gzip.open(
        path,
        "wb",
        compresslevel=_SPILL_COMPRESSION_LEVEL,
    ) as output:
        for _, payload in items:
            output.write(payload)
    return path


def _merge_pass(
    temp_dir: Path,
    pass_id: int,
    run_paths: Sequence[Path],
    key: Callable[[T], Any],
    input_items: int,
    progress: SortProgress,
) -> list[Path]:
    merged_paths: list[Path] = []
    merged_items = 0
    pending_progress = 0
    progress.merging(0, input_items, pass_id)
    for start in range(0, len(run_paths), _MAX_OPEN_RUNS):
        group = run_paths[start : start + _MAX_OPEN_RUNS]
        merged_path = temp_dir / f"merged-{pass_id}-{len(merged_paths)}.pickle.gz"
        with gzip.open(
            merged_path,
            "wb",
            compresslevel=_SPILL_COMPRESSION_LEVEL,
        ) as output:
            for item in _merge_runs(group, key):
                pickle.dump(item, output, protocol=pickle.HIGHEST_PROTOCOL)
                merged_items += 1
                pending_progress += 1
                if pending_progress == _MERGE_PROGRESS_INTERVAL:
                    progress.merging(merged_items, input_items, pass_id)
                    pending_progress = 0
        merged_paths.append(merged_path)
        for path in group:
            path.unlink()
    if pending_progress:
        progress.merging(merged_items, input_items, pass_id)
    return merged_paths


def _merge_runs(
    run_paths: Sequence[Path], key: Callable[[T], Any]
) -> Generator[T, None, None]:
    heap: list[tuple[Any, int, T, Iterator[T]]] = []
    readers: list[Generator[T, None, None]] = []

    try:
        # Each run is a contiguous input slice, so its index is the stable tie-breaker.
        for run_index, path in enumerate(run_paths):
            reader: Generator[T, None, None] = _read_run(path)
            readers.append(reader)
            try:
                first = next(reader)
            except StopIteration:
                continue
            heapq.heappush(heap, (key(first), run_index, first, reader))

        while heap:
            _, run_index, item, heap_reader = heapq.heappop(heap)
            yield item
            try:
                next_item = next(heap_reader)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (key(next_item), run_index, next_item, heap_reader),
            )
    finally:
        for reader in readers:
            reader.close()


def _read_run(path: Path) -> Generator[T, None, None]:
    with gzip.open(path, "rb") as fh:
        while True:
            try:
                yield pickle.load(fh)
            except EOFError:
                return

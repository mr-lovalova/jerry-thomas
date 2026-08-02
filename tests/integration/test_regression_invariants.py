import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import jerrythomas.pipelines.sort as sort_module
from tests.helpers.regression import serve_dataset


_RAW_INPUTS = (
    "humidity_partitioned.jsonl",
    "linear_hourly.jsonl",
    "power_target.jsonl",
    "sine_subhour.jsonl",
)


def _serve_dataset(
    project_root: Path,
    artifact_mode: Literal["AUTO", "FORCE"],
) -> bytes:
    request = serve_dataset(project_root, artifact_mode)
    dataset_path = request.serve_run_plans[0].paths.dataset_dir / "dataset.jsonl"
    return dataset_path.read_bytes()


def _write_jsonl_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _shuffled(lines: list[str], seed: int) -> list[str]:
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    if shuffled == lines:
        shuffled = [*lines[1:], lines[0]]
    return shuffled


def _artifact_snapshot(project_root: Path) -> dict[str, tuple[int, int, int, str]]:
    artifact_dir = project_root / "build" / "build"
    snapshot: dict[str, tuple[int, int, int, str]] = {}
    for path in sorted(
        candidate for candidate in artifact_dir.rglob("*") if candidate.is_file()
    ):
        stat = path.stat()
        snapshot[str(path.relative_to(artifact_dir))] = (
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return snapshot


def test_persisted_dataset_is_independent_of_raw_input_order_and_file_layout(
    copy_fixture,
) -> None:
    project_root = copy_fixture("regression_project")
    data_dir = project_root / "data"
    expected = _serve_dataset(project_root, "FORCE")

    for input_index, filename in enumerate(_RAW_INPUTS):
        path = data_dir / filename
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines()
        variants = {
            "reversed": list(reversed(lines)),
            "seeded shuffle": _shuffled(lines, seed=20260717 + input_index),
        }

        try:
            for order, reordered_lines in variants.items():
                assert reordered_lines != lines
                _write_jsonl_lines(path, reordered_lines)
                actual = _serve_dataset(project_root, "FORCE")
                assert actual == expected, f"{filename} changed output when {order}"
        finally:
            path.write_text(original, encoding="utf-8")

    linear_lines = (
        (data_dir / "linear_hourly.jsonl").read_text(encoding="utf-8").splitlines()
    )
    parts_dir = data_dir / "linear_parts"
    parts_dir.mkdir()
    _write_jsonl_lines(parts_dir / "00-late.jsonl", linear_lines[:2])
    _write_jsonl_lines(parts_dir / "05-middle.jsonl", linear_lines[2:4])
    _write_jsonl_lines(parts_dir / "10-early.jsonl", linear_lines[4:])

    source_path = project_root / "sources" / "metrics.linear.yaml"
    source_config = source_path.read_text(encoding="utf-8")
    old_path = "path: data/linear_hourly.jsonl"
    assert old_path in source_config
    source_path.write_text(
        source_config.replace(old_path, 'path: "data/linear_parts/*.jsonl"'),
        encoding="utf-8",
    )

    assert _serve_dataset(project_root, "FORCE") == expected


def test_auto_reuses_current_artifacts_and_force_rebuilds_them(copy_fixture) -> None:
    project_root = copy_fixture("regression_project")

    first_force_output = _serve_dataset(project_root, "FORCE")
    first_force_artifacts = _artifact_snapshot(project_root)
    state_path = project_root / "build" / "_system" / "build" / "state.json"
    first_force_state = state_path.read_bytes()
    assert first_force_artifacts

    auto_output = _serve_dataset(project_root, "AUTO")
    assert auto_output == first_force_output
    assert _artifact_snapshot(project_root) == first_force_artifacts
    assert state_path.read_bytes() == first_force_state

    second_force_output = _serve_dataset(project_root, "FORCE")
    assert second_force_output == first_force_output
    assert _artifact_snapshot(project_root) != first_force_artifacts
    assert state_path.read_bytes() != first_force_state


def test_auto_rebuilds_after_source_and_config_changes(copy_fixture) -> None:
    project_root = copy_fixture("regression_project")
    original_output = _serve_dataset(project_root, "FORCE")
    original_artifacts = _artifact_snapshot(project_root)

    source_path = project_root / "data" / "linear_hourly.jsonl"
    source = source_path.read_text(encoding="utf-8")
    assert '"value":20.0' in source
    source_path.write_text(
        source.replace('"value":20.0', '"value":200.0'),
        encoding="utf-8",
    )

    changed_source_output = _serve_dataset(project_root, "AUTO")
    changed_source_artifacts = _artifact_snapshot(project_root)
    assert changed_source_output != original_output
    assert changed_source_artifacts != original_artifacts

    dataset_path = project_root / "dataset.yaml"
    dataset = dataset_path.read_text(encoding="utf-8")
    assert "  - id: linear_scaled" in dataset
    dataset_path.write_text(
        dataset.replace("  - id: linear_scaled", "  - id: linear_renamed"),
        encoding="utf-8",
    )

    changed_config_output = _serve_dataset(project_root, "AUTO")
    assert changed_config_output != changed_source_output
    assert _artifact_snapshot(project_root) != changed_source_artifacts


def test_external_sort_spilling_preserves_the_persisted_dataset(
    copy_fixture,
    monkeypatch,
) -> None:
    project_root = copy_fixture("regression_project")
    expected = _serve_dataset(project_root, "FORCE")

    linear_path = project_root / "data" / "linear_hourly.jsonl"
    records = [
        json.loads(line)
        for line in linear_path.read_text(encoding="utf-8").splitlines()
    ]
    padding = "x" * 300_000
    for record in records:
        record["sort_test_padding"] = padding
    _write_jsonl_lines(
        linear_path,
        (json.dumps(record, separators=(",", ":")) for record in records),
    )

    defaults_path = project_root / "profiles" / "serve.defaults.yaml"
    defaults_path.write_text(
        defaults_path.read_text(encoding="utf-8")
        + "\nexecution:\n  sort_buffer_mb: 1\n",
        encoding="utf-8",
    )

    spill_runs = 0
    write_serialized_run = sort_module._write_serialized_run

    def count_spill_runs(
        temp_dir: Path,
        run_id: int,
        items: Iterable[tuple[Any, bytes]],
    ) -> Path:
        nonlocal spill_runs
        spill_runs += 1
        return write_serialized_run(temp_dir, run_id, items)

    monkeypatch.setattr(sort_module, "_write_serialized_run", count_spill_runs)

    assert _serve_dataset(project_root, "FORCE") == expected
    assert spill_runs >= 2

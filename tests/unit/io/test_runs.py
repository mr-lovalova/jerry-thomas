import json

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jerrythomas.io import runs
from tests.run_helpers import empty_recipe


def test_run_ids_include_subsecond_precision(monkeypatch) -> None:
    instants = iter(
        [
            datetime(2026, 1, 1, 12, 0, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 12, 0, 0, 2, tzinfo=timezone.utc),
        ]
    )

    class _Datetime:
        @classmethod
        def now(cls, tz):
            assert tz is timezone.utc
            return next(instants)

    monkeypatch.setattr(runs, "datetime", _Datetime)

    assert runs.make_run_id() == "2026-01-01T12-00-00-000001Z"
    assert runs.make_run_id() == "2026-01-01T12-00-00-000002Z"


def test_finish_run_requires_started_run_metadata(tmp_path: Path) -> None:
    paths = runs.get_run_paths(tmp_path / "serve", "missing")

    with pytest.raises(FileNotFoundError):
        runs.finish_run_success(paths)


def test_finish_run_rejects_non_object_metadata(tmp_path: Path) -> None:
    paths = runs.get_run_paths(tmp_path / "serve", "invalid")
    paths.run_root.mkdir(parents=True)
    paths.metadata_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Input should be an object"):
        runs.finish_run_success(paths)


def test_latest_run_is_a_replaceable_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    probe_target = tmp_path / "probe-target"
    probe_target.mkdir()
    probe_link = tmp_path / "probe-link"
    try:
        probe_link.symlink_to(probe_target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symbolic links are unavailable: {exc}")
    probe_link.unlink()

    serve_root = tmp_path / "serve"
    first = runs.get_run_paths(serve_root, "first")
    second = runs.get_run_paths(serve_root, "second")
    first.run_root.mkdir(parents=True)
    second.run_root.mkdir(parents=True)

    runs.set_latest_run(first)
    latest = serve_root / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == first.run_root.resolve()

    def reject_symlink(
        self: Path,
        target: str | Path,
        target_is_directory: bool = False,
    ) -> None:
        raise PermissionError("symbolic links disabled")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "symlink_to", reject_symlink)
        with pytest.raises(PermissionError, match="symbolic links disabled"):
            runs.set_latest_run(second)
    assert latest.resolve() == first.run_root.resolve()

    runs.set_latest_run(second)
    assert latest.is_symlink()
    assert latest.resolve() == second.run_root.resolve()
    assert not (serve_root / ".latest-second").exists()


def test_latest_run_resolves_relative_serve_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    probe_target = tmp_path / "probe-target"
    probe_target.mkdir()
    probe_link = tmp_path / "probe-link"
    try:
        probe_link.symlink_to(probe_target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symbolic links are unavailable: {exc}")
    probe_link.unlink()

    monkeypatch.chdir(tmp_path)
    paths = runs.get_run_paths(Path("serve"), "run")
    paths.run_root.mkdir(parents=True)

    runs.set_latest_run(paths)

    assert paths.serve_root == (tmp_path / "serve").resolve()
    assert (tmp_path / "serve" / "latest").resolve() == paths.run_root


def test_latest_run_does_not_copy_when_symlinks_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = runs.get_run_paths(tmp_path / "serve", "run")
    paths.dataset_dir.mkdir(parents=True)
    (paths.dataset_dir / "records.jsonl").write_text("record\n", encoding="utf-8")

    def reject_symlink(
        self: Path,
        target: str | Path,
        target_is_directory: bool = False,
    ) -> None:
        raise PermissionError("symbolic links disabled")

    monkeypatch.setattr(Path, "symlink_to", reject_symlink)

    with pytest.raises(PermissionError, match="symbolic links disabled"):
        runs.set_latest_run(paths)

    assert not (paths.serve_root / "latest").exists()
    assert paths.run_root.is_dir()


def test_latest_run_removes_pending_link_when_replace_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    probe_target = tmp_path / "probe-target"
    probe_target.mkdir()
    probe_link = tmp_path / "probe-link"
    try:
        probe_link.symlink_to(probe_target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symbolic links are unavailable: {exc}")
    probe_link.unlink()

    serve_root = tmp_path / "serve"
    first = runs.get_run_paths(serve_root, "first")
    second = runs.get_run_paths(serve_root, "second")
    first.run_root.mkdir(parents=True)
    second.run_root.mkdir(parents=True)
    runs.set_latest_run(first)

    pending = serve_root / ".latest-second"
    original_replace = Path.replace

    def reject_pending_replace(self: Path, target: str | Path) -> Path:
        if self == pending:
            raise PermissionError("replace disabled")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", reject_pending_replace)

    with pytest.raises(PermissionError, match="replace disabled"):
        runs.set_latest_run(second)

    assert (serve_root / "latest").resolve() == first.run_root.resolve()
    assert not pending.exists()
    assert not pending.is_symlink()


def test_latest_run_refuses_to_delete_a_real_directory(tmp_path: Path) -> None:
    paths = runs.get_run_paths(tmp_path / "serve", "run")
    paths.run_root.mkdir(parents=True)
    latest = paths.serve_root / "latest"
    latest.mkdir()
    marker = latest / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="is not a symbolic link"):
        runs.set_latest_run(paths)

    assert marker.read_text(encoding="utf-8") == "user data"
    assert not (paths.serve_root / ".latest-run").exists()


@pytest.mark.parametrize("version", [None, 0, 1, 2, 3, "4", True, 4.0])
def test_saved_run_rejects_missing_or_unsupported_version(tmp_path, version):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, recipe=empty_recipe())
    runs.finish_run_success(paths)
    data = json.loads(paths.metadata_path.read_text())
    if version is None:
        del data["schema_version"]
    else:
        data["schema_version"] = version
    paths.metadata_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema_version"):
        runs.load_run(paths.run_root)


@pytest.mark.parametrize(
    ("identity", "missing"),
    [
        ({"dataset_id": "prices", "dataset_version": "v2"}, "dataset_version"),
        ({"dataset_id": "prices", "dataset_version": None}, None),
        ({"dataset_id": None, "dataset_version": "v2"}, None),
        ({"dataset_id": "prices", "dataset_version": " "}, None),
    ],
)
def test_saved_run_rejects_incomplete_dataset_identity(tmp_path, identity, missing):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, recipe=empty_recipe())
    runs.finish_run_success(paths)
    data = json.loads(paths.metadata_path.read_text())
    data.update(identity)
    if missing is not None:
        del data[missing]
    paths.metadata_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="dataset"):
        runs.load_run(paths.run_root)


@pytest.mark.parametrize("status", ["running", "failed"])
def test_saved_run_rejects_unfinished_or_failed_run(tmp_path, status):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, recipe=empty_recipe())
    if status == "failed":
        runs.finish_run_failed(paths)
    with pytest.raises(ValueError, match="successfully finished"):
        runs.load_run(paths.run_root)


@pytest.fixture
def saved_output():
    return runs.RunOutput(
        profile="dataset",
        operation="dataset",
        output_id=None,
        path="dataset/records.jsonl",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        compression=None,
        row_count=0,
        fold=None,
    )


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/outside",
        "C:/outside",
        "dataset/../../outside",
        "dataset\\outside",
        ".",
        "",
        "dataset//records",
    ],
)
def test_saved_run_rejects_invalid_output_path(tmp_path, saved_output, path):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, recipe=empty_recipe())
    runs.finish_run_success(paths, outputs=(saved_output,))
    data = json.loads(paths.metadata_path.read_text())
    data["outputs"][0]["path"] = path
    paths.metadata_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="relative POSIX file path"):
        runs.load_run(paths.run_root)


@pytest.mark.parametrize("row_count", [-1, True, "3", 1.5])
def test_saved_run_rejects_invalid_row_count(tmp_path, saved_output, row_count):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, recipe=empty_recipe())
    runs.finish_run_success(paths, outputs=(saved_output,))
    data = json.loads(paths.metadata_path.read_text())
    data["outputs"][0]["row_count"] = row_count
    paths.metadata_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="row_count"):
        runs.load_run(paths.run_root)


def test_saved_run_rejects_ambiguous_outputs(tmp_path, saved_output):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, recipe=empty_recipe())
    with pytest.raises(ValueError, match="unique profile/output_id"):
        runs.finish_run_success(paths, outputs=(saved_output, saved_output))
    assert json.loads(paths.metadata_path.read_text())["status"] == "running"


def test_saved_run_checks_only_selected_file(tmp_path, saved_output):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, recipe=empty_recipe())
    runs.finish_run_success(paths, outputs=(saved_output,))
    saved = runs.load_run(paths.run_root)
    assert saved.output("dataset") == saved_output
    with pytest.raises(KeyError, match="Run has no output"):
        saved.output("another")
    with pytest.raises(FileNotFoundError):
        saved.output_path("dataset")
    destination = paths.run_root / saved_output.path
    destination.mkdir()
    with pytest.raises(ValueError, match="not a file"):
        saved.output_path("dataset")
    destination.rmdir()
    outside = tmp_path / "outside.jsonl"
    outside.touch()
    destination.symlink_to(outside)
    with pytest.raises(ValueError, match="outside the run directory"):
        saved.output_path("dataset")


def test_saved_run_pins_latest_at_load_time(tmp_path, saved_output):
    first = runs.get_run_paths(tmp_path, "first")
    second = runs.get_run_paths(tmp_path, "second")
    for paths in (first, second):
        runs.start_run(paths, recipe=empty_recipe())
        (paths.run_root / saved_output.path).touch()
        runs.finish_run_success(paths, outputs=(saved_output,))
    runs.set_latest_run(first)
    saved = runs.load_run(tmp_path / "latest")
    runs.set_latest_run(second)
    assert saved.directory == first.run_root
    assert saved.output_path("dataset") == first.run_root / saved_output.path


@pytest.fixture
def saved_time_split():
    from jerrythomas.config.dataset.split import TimeSplitConfig

    return TimeSplitConfig.model_validate(
        {
            "mode": "time",
            "intervals": [
                {"id": "development", "until": "2024-01-01T00:00:00Z"},
                {"id": "holdout", "until": None},
            ],
            "folds": [
                {"id": "evaluation", "train": ["development"], "test": ["holdout"]}
            ],
        }
    )


@pytest.fixture
def saved_fold_output(saved_output):
    return saved_output.model_copy(
        update={
            "output_id": "evaluation.train",
            "fold": runs.RunFoldOutput(
                id="evaluation", role="train", labels=("development",)
            ),
        }
    )


def test_saved_run_requires_explicit_split_field(tmp_path):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, recipe=empty_recipe())
    runs.finish_run_success(paths)
    data = json.loads(paths.metadata_path.read_text())
    assert data.pop("split") is None
    paths.metadata_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="split"):
        runs.load_run(paths.run_root)


@pytest.mark.parametrize("status", ["success", "failed"])
def test_finishing_run_preserves_original_split(tmp_path, saved_time_split, status):
    from jerrythomas.config.dataset.split import TimeInterval

    paths = runs.get_run_paths(tmp_path, "run")
    started = runs.start_run(paths, split=saved_time_split, recipe=empty_recipe())
    saved_time_split.intervals[0] = TimeInterval(
        id="development", until="2025-01-01T00:00:00Z"
    )
    saved_time_split.folds[0].train.append("changed")
    assert started.split.intervals[0].until == "2024-01-01T00:00:00Z"
    assert started.split.folds[0].train == ["development"]
    finished = runs.finish_run(paths, status)
    assert finished.split == started.split
    manifest = json.loads(paths.metadata_path.read_text())
    assert manifest["split"] == started.split.model_dump(mode="json")


def test_saved_hash_split_preserves_ratios_and_seed(tmp_path, saved_fold_output):
    from jerrythomas.config.dataset.split import HashSplitConfig
    from jerrythomas.pipelines.dataset.split import build_labeler

    split = HashSplitConfig.model_validate(
        {
            "mode": "hash",
            "ratios": {"development": 0.8, "holdout": 0.2},
            "seed": 91,
            "folds": [
                {"id": "evaluation", "train": ["development"], "test": ["holdout"]}
            ],
        }
    )
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, split=split, recipe=empty_recipe())
    runs.finish_run_success(paths, outputs=(saved_fold_output,))
    saved = runs.load_run(paths.run_root)
    assert saved.metadata.split == split
    original = build_labeler(split)
    restored = build_labeler(saved.metadata.split)
    assert [restored.label(key) for key in range(100)] == [
        original.label(key) for key in range(100)
    ]


@pytest.mark.parametrize(
    "changed",
    [
        {"output_id": "missing.train"},
        {"fold": None},
        {"fold": {"id": "different", "role": "train", "labels": ["development"]}},
        {"fold": {"id": "evaluation", "role": "test", "labels": ["development"]}},
        {"fold": {"id": "evaluation", "role": "train", "labels": ["holdout"]}},
    ],
)
def test_saved_run_rejects_fold_metadata_disagreeing_with_split(
    tmp_path, saved_time_split, saved_fold_output, changed
):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, split=saved_time_split, recipe=empty_recipe())
    runs.finish_run_success(paths, outputs=(saved_fold_output,))
    data = json.loads(paths.metadata_path.read_text())
    data["outputs"][0].update(changed)
    paths.metadata_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="saved split"):
        runs.load_run(paths.run_root)


def test_fold_output_requires_saved_split(tmp_path, saved_fold_output):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, recipe=empty_recipe())
    with pytest.raises(ValueError, match="require a saved split"):
        runs.finish_run_success(paths, outputs=(saved_fold_output,))


def test_fold_output_requires_output_id(tmp_path, saved_time_split, saved_fold_output):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(paths, split=saved_time_split, recipe=empty_recipe())
    missing_id = saved_fold_output.model_copy(update={"output_id": None})
    with pytest.raises(ValueError, match="require an output_id"):
        runs.finish_run_success(paths, outputs=(missing_id,))


def test_split_preview_records_rules_without_claiming_fold_output(
    tmp_path, saved_time_split, saved_output, saved_fold_output
):
    paths = runs.get_run_paths(tmp_path, "run")
    runs.start_run(
        paths, preview="records", split=saved_time_split, recipe=empty_recipe()
    )
    with pytest.raises(ValueError, match="preview outputs must not declare a fold"):
        runs.finish_run_success(paths, outputs=(saved_fold_output,))
    stream_output = saved_output.model_copy(update={"output_id": "stream.records"})
    runs.finish_run_success(paths, outputs=(stream_output,))
    saved = runs.load_run(paths.run_root)
    assert saved.metadata.split == saved_time_split
    assert saved.output("dataset", "stream.records").fold is None

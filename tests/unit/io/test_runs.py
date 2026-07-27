from datetime import datetime, timezone
from pathlib import Path

import pytest

from datapipeline.io import runs


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

    with pytest.raises(ValueError, match="Expected JSON object"):
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

from pathlib import Path
import sys

import pytest
import yaml

from jerrythomas.cli.app import main


def _write_yaml(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _source(path: Path, source_id: str) -> None:
    _write_yaml(
        path,
        {
            "id": source_id,
            "loader": {"entrypoint": "core.synthetic.ticks", "args": {}},
            "parser": {"entrypoint": "core.identity", "args": {}},
            "freshness": "opaque",
        },
    )


def _project(root: Path, *, shared: bool = False) -> Path:
    sources = ["owned/sources", "shared/sources"] if shared else ["sources"]
    streams = ["owned/streams", "shared/streams"] if shared else ["streams"]
    project = root / "project.yaml"
    _write_yaml(
        project,
        {
            "schema_version": 7,
            "artifact_revision": 1,
            "paths": {
                "sources": sources,
                "streams": streams,
                "artifacts": "artifacts",
            },
        },
    )
    for directory in sources + streams:
        (root / directory).mkdir(parents=True)
    return project


def _snapshot(directory: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    plugin = tmp_path / "plugin"
    package = plugin / "src" / "test_plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "test-plugin"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    _project(tmp_path / "projects" / "default")
    selected = _project(tmp_path / "projects" / "selected", shared=True)
    _source(selected.parent / "shared/sources/selected.tick.yaml", "selected.tick")
    _write_yaml(
        tmp_path / "jerry.yaml",
        {
            "plugin_root": "plugin",
            "projects": {
                "default": "projects/default",
                "selected": "projects/selected",
            },
            "default_project": "default",
        },
    )
    nested = tmp_path / "nested" / "cwd"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    return tmp_path


def _invoke(
    monkeypatch, arguments: list[str], answers: tuple[str, ...] | list[str] = ()
) -> None:
    responses = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
    monkeypatch.setattr(sys, "argv", ["jerry", *arguments])
    main()


@pytest.mark.parametrize(
    "selector", ["selected", "projects/selected", "projects/selected/project.yaml"]
)
@pytest.mark.parametrize("command", ["source", "stream", "inflow"])
def test_scaffolds_target_selected_project_roots(
    workspace, monkeypatch, selector, command
):
    default_root = workspace / "projects" / "default"
    default_before = _snapshot(default_root)
    selected = workspace / "projects" / "selected"
    shared_before = _snapshot(selected / "shared")
    if command == "source":
        arguments = [
            "demo.fresh",
            "--transport",
            "synthetic",
            "--parser",
            "core.identity",
        ]
        answers = []
    elif command == "stream":
        arguments = ["--identity"]
        answers = ["1", "1", ""]
    else:
        arguments = []
        answers = ["1", "demo", "fresh", "", "3", "3", "", "", "2", ""]

    _invoke(
        monkeypatch,
        [command, "create", "--project", selector, *arguments],
        answers,
    )

    if command in {"source", "inflow"}:
        assert (selected / "owned/sources/demo.fresh.yaml").is_file()
    if command in {"stream", "inflow"}:
        stream_id = "tick.tick" if command == "stream" else "fresh.fresh"
        stream = yaml.safe_load(
            (selected / "owned/streams" / f"{stream_id}.yaml").read_text()
        )
        assert stream["from"]["source"] == (
            "selected.tick" if command == "stream" else "demo.fresh"
        )
    if command == "inflow":
        assert (workspace / "plugin/src/test_plugin/domains/fresh/model.py").is_file()
    assert _snapshot(default_root) == default_before
    assert _snapshot(selected / "shared") == shared_before
    assert not (workspace / "plugin/your-project").exists()


@pytest.mark.parametrize("command", ["source", "stream", "inflow"])
@pytest.mark.parametrize("selector", ["missing", "missing/project.yaml", "projects"])
def test_missing_explicit_project_does_not_fall_back_or_create_files(
    workspace, monkeypatch, command, selector
):
    before = _snapshot(workspace)

    with pytest.raises(SystemExit, match="Unknown project|Project file does not exist"):
        _invoke(monkeypatch, [command, "create", "--project", selector])

    assert _snapshot(workspace) == before


@pytest.mark.parametrize("command", ["source", "stream"])
def test_builtin_scaffolds_work_in_standalone_project(tmp_path, monkeypatch, command):
    project = _project(tmp_path)
    _source(tmp_path / "sources/demo.tick.yaml", "demo.tick")
    monkeypatch.chdir(tmp_path)
    arguments = (
        ["demo.fresh", "--transport", "synthetic", "--parser", "core.identity"]
        if command == "source"
        else ["--identity"]
    )
    answers = [] if command == "source" else ["1", "1", ""]

    _invoke(
        monkeypatch,
        [command, "create", "--project", str(project), *arguments],
        answers,
    )

    expected = (
        "sources/demo.fresh.yaml" if command == "source" else "streams/tick.tick.yaml"
    )
    assert (tmp_path / expected).is_file()


@pytest.mark.parametrize("command", ["source", "stream"])
def test_standalone_interactive_scaffolds_allow_builtin_choices(
    tmp_path, monkeypatch, command
):
    project = _project(tmp_path)
    _source(tmp_path / "sources/demo.tick.yaml", "demo.tick")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = ["demo.fresh", "3", "1"] if command == "source" else ["1", "1", "1", ""]

    _invoke(monkeypatch, [command, "create", "--project", str(project)], answers)

    expected = (
        "sources/demo.fresh.yaml" if command == "source" else "streams/tick.tick.yaml"
    )
    assert (tmp_path / expected).is_file()


def test_standalone_aligned_stream_accepts_manual_combiner(tmp_path, monkeypatch):
    project = _project(tmp_path)
    _source(tmp_path / "sources/demo.tick.yaml", "demo.tick")
    for stream_id in ("demo.one", "demo.two"):
        _write_yaml(
            tmp_path / "streams" / f"{stream_id}.yaml",
            {
                "id": stream_id,
                "from": {"source": "demo.tick"},
                "map": {"entrypoint": "core.identity"},
            },
        )
    monkeypatch.chdir(tmp_path)

    _invoke(
        monkeypatch,
        ["stream", "create", "--project", str(project)],
        ["2", "1,2", "demo.joined", "custom.combine"],
    )

    stream = yaml.safe_load((tmp_path / "streams/demo.joined.yaml").read_text())
    assert stream["from"]["stream"] == "demo.one"
    assert stream["join"]["streams"] == ["demo.two"]
    assert stream["combine"]["entrypoint"] == "custom.combine"


def test_scaffolding_preserves_workspace_default_project(workspace, monkeypatch):
    selected = workspace / "projects" / "selected"
    before = _snapshot(selected)

    _invoke(
        monkeypatch,
        [
            "source",
            "create",
            "demo.fresh",
            "--transport",
            "synthetic",
            "--parser",
            "core.identity",
        ],
    )

    assert (workspace / "projects/default/sources/demo.fresh.yaml").is_file()
    assert _snapshot(selected) == before

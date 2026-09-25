import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jerrythomas.config.sources import SourceConfig
from jerrythomas.services.config_refs import (
    interpolate_config_vars,
    resolve_config_refs,
)
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.project import load_project
from jerrythomas.services.streams.loader import load_streams
from jerrythomas.services.streams.source import build_source
from jerrythomas.config.interpolation import is_missing_interpolation


def _project_variables(project_yaml: Path):
    return load_project(project_yaml).variables


def _sources(project_yaml: Path):
    return load_streams(load_project(project_yaml)).sources


def _artifact_hashes(project_yaml: Path):
    return load_project_definition(project_yaml).artifact_hashes


def _write_project_yaml(
    project_root: Path,
    *,
    artifacts_value: str = "build",
    variant: str | None = None,
    globals_lines: list[str] | None = None,
) -> Path:
    project_yaml = project_root / "project.yaml"
    lines = [
        "schema_version: 7",
        "artifact_revision: 1",
        "name: sample",
    ]
    if variant is not None:
        lines.append(f"variant: {variant}")
    lines.extend(
        [
            "paths:",
            "  streams: streams",
            "  sources: sources",
            "  datasets: datasets",
            f"  artifacts: {artifacts_value}",
            "  operations: operations",
        ]
    )
    if globals_lines:
        lines.append("globals:")
        lines.extend(f"  {line}" for line in globals_lines)
    else:
        lines.append("globals: {}")

    project_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return project_yaml


def _write_project_files(project_root: Path) -> None:
    (project_root / "datasets").mkdir(parents=True, exist_ok=True)
    (project_root / "streams").mkdir(parents=True, exist_ok=True)
    (project_root / "sources").mkdir(parents=True, exist_ok=True)
    (project_root / "operations").mkdir(parents=True, exist_ok=True)
    (project_root / "datasets" / "default.yaml").write_text(
        "version: v1\nsample:\n  rounding: ceil\n  cadence: 1h\nfeatures: []\ntargets: []\n",
        encoding="utf-8",
    )


def _use_source(project_root: Path, source_id: str) -> None:
    (project_root / "streams" / "tracked.yaml").write_text(
        f"id: tracked\nfrom: {{source: {source_id}}}\nmap: {{entrypoint: map}}\n",
        encoding="utf-8",
    )
    (project_root / "datasets" / "default.yaml").write_text(
        "version: v1\nsample: {rounding: ceil, cadence: 1h}\n"
        "features: [{id: value, stream: tracked, field: value}]\n"
        "targets: []\n",
        encoding="utf-8",
    )


def test_project_requires_schema_version(tmp_path: Path) -> None:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text("version: 2\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Project config requires schema_version: 7",
    ):
        load_project(project_yaml)


def test_project_resolves_default_profiles_directory(tmp_path: Path) -> None:
    project_yaml = _write_project_yaml(tmp_path)

    project = load_project(project_yaml)

    assert project.config.paths.profiles == "./profiles"
    assert project.profiles_dir == (tmp_path / "profiles").resolve()


@pytest.mark.parametrize("value", ["1", "2", "3", "4"])
def test_project_rejects_unsupported_schema_version(
    tmp_path: Path,
    value: str,
) -> None:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(f"schema_version: {value}\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"Unsupported project schema version {value}; expected 7",
    ):
        load_project(project_yaml)


@pytest.mark.parametrize("value", ["'3'", "3.0", "true", "null"])
def test_project_schema_version_must_be_integer(
    tmp_path: Path,
    value: str,
) -> None:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(f"schema_version: {value}\n", encoding="utf-8")

    with pytest.raises(TypeError, match="schema_version must be the integer 7"):
        load_project(project_yaml)


def test_globals_resolve_env_refs_from_project_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAW_ROOT", raising=False)
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    (project_root / ".env").write_text("RAW_ROOT=../shared-data\n", encoding="utf-8")
    project_yaml = _write_project_yaml(
        project_root,
        artifacts_value="${raw_root}/artifacts",
        globals_lines=["raw_root: ${env:RAW_ROOT}"],
    )

    project = load_project(project_yaml)
    assert project.variables["raw_root"] == "../shared-data"
    assert (
        project.artifacts_root
        == (project_root / "../shared-data" / "artifacts").resolve()
    )


def test_process_env_overrides_project_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    (project_root / ".env").write_text("RAW_ROOT=../shared-data\n", encoding="utf-8")
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=["raw_root: ${env:RAW_ROOT}"],
    )
    monkeypatch.setenv("RAW_ROOT", "/runtime/raw")

    assert _project_variables(project_yaml)["raw_root"] == "/runtime/raw"


@pytest.mark.parametrize("working_directory", ["project", "elsewhere"])
def test_project_path_refs_resolve_loader_arguments_and_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, working_directory: str
) -> None:
    project_root = tmp_path / "project"
    _write_project_files(project_root)
    (tmp_path / "elsewhere").mkdir()
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=["raw_root: ${path:../shared data}"],
    )
    (project_root / "sources" / "prices.yaml").write_text(
        "id: prices\n"
        "parser: {entrypoint: core.identity}\n"
        "loader:\n"
        "  entrypoint: custom.prices\n"
        "  args: {root: '${raw_root}'}\n"
        "inputs:\n"
        "  files: ['${raw_root}/prices.parquet']\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path / working_directory)

    source = _sources(project_yaml)["prices"]
    expected = str((tmp_path / "shared data").resolve())
    assert source.loader.args == {"root": expected}
    assert source.inputs.files == (expected + "/prices.parquet",)


def test_config_refs_resolve_literal_paths_in_nested_values(tmp_path: Path) -> None:
    project_yaml = tmp_path / "project" / "project.yaml"
    absolute_path = tmp_path / "absolute"
    resolved = resolve_config_refs(
        {
            "root": "${path:../shared data}",
            "files": ["${path:../shared data}/rows.jsonl"],
            "nested": {"absolute": f"${{path:{absolute_path}}}"},
            "mixed": "${path:.}/${env:FILE}",
            "ordinary": "../unchanged",
        },
        project_yaml=project_yaml,
        env={"FILE": "rows.jsonl"},
    )
    assert resolved == {
        "root": str(tmp_path / "shared data"),
        "files": [str(tmp_path / "shared data" / "rows.jsonl")],
        "nested": {"absolute": str(absolute_path)},
        "mixed": str(project_yaml.parent / "rows.jsonl"),
        "ordinary": "../unchanged",
    }
    assert not (tmp_path / "shared data").exists()


@pytest.mark.parametrize("reference", ["${path:}", "${path:   }"])
def test_path_refs_reject_empty_paths(tmp_path: Path, reference: str) -> None:
    with pytest.raises(ValueError, match="must include a path"):
        resolve_config_refs(reference, project_yaml=tmp_path / "project.yaml", env={})


@pytest.mark.parametrize("reference", ["${path:${root}}", "${path:${env:ROOT}/data}"])
def test_path_refs_reject_nested_references(tmp_path: Path, reference: str) -> None:
    with pytest.raises(ValueError, match="literal path"):
        resolve_config_refs(reference, project_yaml=tmp_path / "project.yaml", env={})


_DOTENV_ESCAPE_CASES = [
    # (raw .env line, expected decoded value)
    ('ESCAPED_BACKSLASH="x\\\\ny"', "x\\ny"),
    ('TAB="a\\tb"', "a\tb"),
    ('CARRIAGE_RETURN="a\\rb"', "a\rb"),
    ('NEWLINE="l1\\nl2"', "l1\nl2"),
    ('QUOTE="say \\"hi\\""', 'say "hi"'),
    ('BACKSLASH_BEFORE_QUOTE="abc\\\\"', "abc\\"),
    ('UNKNOWN_ESCAPE="\\q"', "\\q"),
    ("SINGLE_QUOTED='a\\nb'", "a\\nb"),
    ('DOUBLE_QUOTED_HASH="a # b"', "a # b"),
    ("SINGLE_QUOTED_HASH='a # b'", "a # b"),
]


@pytest.mark.parametrize("comment", ["", ' # local "comment"'])
def test_project_dotenv_decodes_escapes_left_to_right(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    comment: str,
) -> None:
    keys = [line.split("=", 1)[0] for line, _ in _DOTENV_ESCAPE_CASES]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    (project_root / ".env").write_text(
        "".join(f"{line}{comment}\n" for line, _ in _DOTENV_ESCAPE_CASES),
        encoding="utf-8",
    )
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=[f"{key.lower()}: ${{env:{key}}}" for key in keys],
    )

    variables = _project_variables(project_yaml)

    for key, (_, expected) in zip(keys, _DOTENV_ESCAPE_CASES, strict=True):
        assert variables[key.lower()] == expected, key


def test_config_refs_resolve_full_and_embedded_env_values(tmp_path: Path) -> None:
    resolved = resolve_config_refs(
        {
            "root": "${env:RAW_ROOT}",
            "path": "${env:RAW_ROOT}/prices.jsonl",
            "files": ["${env:RAW_ROOT}/prices.jsonl"],
        },
        project_yaml=tmp_path / "project.yaml",
        env={"RAW_ROOT": "data/raw"},
    )

    assert resolved == {
        "root": "data/raw",
        "path": "data/raw/prices.jsonl",
        "files": ["data/raw/prices.jsonl"],
    }


def test_project_manifest_resolves_env_before_project_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIG_VALUE", "${window}")
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    project = load_project(
        _write_project_yaml(
            project_root,
            globals_lines=["window: 21"],
        )
    )
    monkeypatch.setenv("CONFIG_VALUE", "999")

    assert project.resolve_config(
        {
            "embedded": "raw/${env:CONFIG_VALUE}",
            "typed": "${env:CONFIG_VALUE}",
        }
    ) == {
        "embedded": "raw/21",
        "typed": 21,
    }


def test_config_refs_reject_unsupported_schemes(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported config reference scheme 'secret'",
    ):
        resolve_config_refs(
            "${secret:RAW_ROOT}",
            project_yaml=tmp_path / "project.yaml",
            env={},
        )


def test_project_variant_can_be_used_in_project_paths(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(
        project_root,
        variant="long",
        artifacts_value="build/${project_variant}",
    )

    assert (
        load_project(project_yaml).artifacts_root
        == (project_root / "build" / "long").resolve()
    )


def test_globals_can_reference_other_globals(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=[
            "canonical_root: data/canonical",
            "canonical_equity_interim_dir: ${canonical_root}/equity/interim",
        ],
    )

    globals_ = _project_variables(project_yaml)

    assert globals_["canonical_equity_interim_dir"] == ("data/canonical/equity/interim")


@pytest.mark.parametrize("field", ["start_time", "end_time"])
def test_project_time_bounds_resolve_global_aliases(
    tmp_path: Path,
    field: str,
) -> None:
    project_yaml = _write_project_yaml(
        tmp_path,
        globals_lines=[
            "data_boundary: 2024-01-01T00:00:00Z",
            f"{field}: ${{data_boundary}}",
        ],
    )

    project = load_project(project_yaml)

    assert getattr(project.config.globals, field) == datetime(
        2024, 1, 1, tzinfo=timezone.utc
    )
    assert project.variables[field] == "2024-01-01T00:00:00Z"


@pytest.mark.parametrize("field", ["start_time", "end_time"])
def test_project_time_bounds_accept_aliases_to_null(
    tmp_path: Path,
    field: str,
) -> None:
    project_yaml = _write_project_yaml(
        tmp_path,
        globals_lines=["optional_boundary:", f"{field}: ${{optional_boundary}}"],
    )

    project = load_project(project_yaml)

    assert getattr(project.config.globals, field) is None
    assert is_missing_interpolation(project.variables[field])


def test_project_time_range_validates_resolved_aliases(tmp_path: Path) -> None:
    project_yaml = _write_project_yaml(
        tmp_path,
        globals_lines=[
            "data_start: 2024-01-02T00:00:00Z",
            "data_end: 2024-01-01T00:00:00Z",
            "start_time: ${data_start}",
            "end_time: ${data_end}",
        ],
    )

    with pytest.raises(ValueError, match="start_time must not be after"):
        load_project(project_yaml)


@pytest.mark.parametrize(
    "value",
    ["2024-01-01T00:00:00.123456Z", '"2024-01-01T00:00:00.123456Z"'],
)
def test_project_preserves_literal_time_bound_precision(
    tmp_path: Path,
    value: str,
) -> None:
    project_yaml = _write_project_yaml(
        tmp_path,
        globals_lines=[f"start_time: {value}"],
    )

    project = load_project(project_yaml)

    assert project.config.globals.start_time == datetime(
        2024, 1, 1, microsecond=123456, tzinfo=timezone.utc
    )


def test_load_sources_resolve_nested_globals_before_fs_path_normalization(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    data_dir = project_root / "data" / "canonical" / "equity" / "interim"
    data_dir.mkdir(parents=True)
    (data_dir / "prices.jsonl").write_text("{}", encoding="utf-8")
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=[
            "canonical_root: data/canonical",
            "canonical_equity_interim_dir: ${canonical_root}/equity/interim",
        ],
    )
    (project_root / "sources" / "prices.yaml").write_text(
        "\n".join(
            [
                "id: equity.prices",
                "parser:",
                "  entrypoint: core.identity",
                "  args: {}",
                "loader:",
                "  transport: fs",
                "  path: ${canonical_equity_interim_dir}/prices.jsonl",
                "  reader:",
                "    format: jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _sources(project_yaml)
    source = build_source(
        SourceConfig.model_validate(loaded["equity.prices"]),
        project_yaml=project_yaml,
    )

    assert source.loader.transport.path == str((data_dir / "prices.jsonl").resolve())


def test_cyclic_global_reference_raises_clear_error(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=[
            "left: ${right}",
            "right: ${left}",
        ],
    )

    with pytest.raises(ValueError, match="Cyclic project global reference"):
        load_project(project_yaml)


def test_unknown_project_global_reference_raises_clear_error(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=["canonical_root: ${canonical_rooot}/canonical"],
    )

    with pytest.raises(
        ValueError,
        match="Unknown interpolation variable 'canonical_rooot' in project globals",
    ):
        load_project(project_yaml)


def test_project_paths_validate_interpolation_without_declared_variables(
    tmp_path: Path,
) -> None:
    _write_project_files(tmp_path)
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        """schema_version: 7
artifact_revision: 1
paths:
  streams: streams
  sources: sources
  datasets: datasets
  artifacts: ${unknown_root}/artifacts
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unknown interpolation variable 'unknown_root'",
    ):
        load_project(project_yaml)


@pytest.mark.parametrize(
    "configured_value",
    ["${missing_root}", "${missing_root}/prices.jsonl"],
)
def test_unknown_config_interpolation_raises_clear_error(
    configured_value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unknown interpolation variable 'missing_root'",
    ):
        interpolate_config_vars(
            {"path": configured_value},
            {"configured_root": "data"},
        )


def test_global_reference_to_null_stays_missing(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=[
            "optional_root:",
            "derived_root: ${optional_root}",
        ],
    )

    globals_ = _project_variables(project_yaml)

    assert globals_["optional_root"] is None
    assert is_missing_interpolation(globals_["derived_root"])


def test_missing_global_cannot_be_embedded_in_another_global(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=[
            "optional_root:",
            "missing_root: ${optional_root}",
            "derived_root: data/${missing_root}",
        ],
    )

    with pytest.raises(
        ValueError,
        match="Interpolation variable 'missing_root' has no value and cannot be embedded",
    ):
        load_project(project_yaml)


def test_null_config_value_cannot_be_embedded_in_text() -> None:
    with pytest.raises(
        ValueError,
        match="Interpolation variable 'optional_root' has no value and cannot be embedded",
    ):
        interpolate_config_vars(
            {"path": "data/${optional_root}/prices.jsonl"},
            {"optional_root": None},
        )


def test_full_placeholder_interpolation_preserves_global_value_type(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=[
            "adv_window_days: 21",
            "annualization_factor: 15.874507866387544",
        ],
    )

    interpolated = interpolate_config_vars(
        {
            "stream": [
                {
                    "operation": "rolling",
                    "field": "dollar_volume",
                    "window": "${adv_window_days}",
                    "to": "adv_${adv_window_days}",
                },
                {
                    "operation": "derive",
                    "left": "return_pstdev_63",
                    "operator": "mul",
                    "right_value": "${annualization_factor}",
                    "to": "volatility_63",
                },
            ]
        },
        _project_variables(project_yaml),
    )

    rolling = interpolated["stream"][0]
    derive = interpolated["stream"][1]
    assert rolling["window"] == 21
    assert isinstance(rolling["window"], int)
    assert rolling["to"] == "adv_21"
    assert derive["right_value"] == 15.874507866387544
    assert isinstance(derive["right_value"], float)


def test_load_sources_resolve_env_backed_globals_before_fs_path_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAW_GLOB", raising=False)
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    (project_root / ".env").write_text("RAW_GLOB=data/*.jsonl\n", encoding="utf-8")
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "rows.jsonl").write_text("{}", encoding="utf-8")
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=["raw_glob: ${env:RAW_GLOB}"],
    )
    (project_root / "sources" / "sample.yaml").write_text(
        "\n".join(
            [
                "id: sample.fs",
                "parser:",
                "  entrypoint: core.identity",
                "  args: {}",
                "loader:",
                "  transport: fs",
                "  path: ${raw_glob}",
                "  reader:",
                "    format: jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _sources(project_yaml)
    source = build_source(
        SourceConfig.model_validate(loaded["sample.fs"]),
        project_yaml=project_yaml,
    )

    assert source.loader.transport.pattern == str(
        (project_root / "data" / "*.jsonl").resolve()
    )


def test_missing_env_ref_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_ROOT", raising=False)
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=["raw_root: ${env:MISSING_ROOT}"],
    )

    with pytest.raises(ValueError, match="MISSING_ROOT"):
        load_project(project_yaml)


def test_sources_dir_resolves_env_backed_project_path_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAW_ROOT", raising=False)
    project_root = tmp_path / "project"
    raw_root = tmp_path / "shared"
    (raw_root / "sources").mkdir(parents=True)
    _write_project_files(project_root)
    (project_root / ".env").write_text(f"RAW_ROOT={raw_root}\n", encoding="utf-8")
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=["raw_root: ${env:RAW_ROOT}"],
    )
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "  sources: sources",
            "  sources: ${raw_root}/sources",
        ),
        encoding="utf-8",
    )

    assert load_project(project_yaml).source_dirs[0] == (raw_root / "sources").resolve()


def test_artifact_hash_changes_when_env_value_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAW_ROOT", raising=False)
    project_root = tmp_path / "project"
    _write_project_files(project_root)
    (project_root / "sources" / "sample.yaml").write_text(
        "\n".join(
            [
                "id: sample.fs",
                "parser:",
                "  entrypoint: core.identity",
                "  args: {}",
                "loader:",
                "  transport: fs",
                "  path: ${raw_root}/rows.jsonl",
                "  reader:",
                "    format: jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    project_yaml = _write_project_yaml(
        project_root,
        globals_lines=["raw_root: ${env:RAW_ROOT}"],
    )
    _use_source(project_root, "sample.fs")

    monkeypatch.setenv("RAW_ROOT", "/tmp/raw-one")
    hash_one = _artifact_hashes(project_yaml)

    monkeypatch.setenv("RAW_ROOT", "/tmp/raw-two")
    hash_two = _artifact_hashes(project_yaml)

    assert hash_one != hash_two


def test_artifact_hash_includes_multiple_source_roots(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    common_root = tmp_path / "common"
    _write_project_files(project_root)
    (common_root / "sources").mkdir(parents=True)
    (common_root / "sources" / "common.yaml").write_text(
        "\n".join(
            [
                "id: common.fs",
                "parser:",
                "  entrypoint: core.identity",
                "loader:",
                "  transport: fs",
                "  path: rows.jsonl",
                "  reader:",
                "    format: jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    project_yaml = _write_project_yaml(project_root)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "  sources: sources",
            "  sources:\n    - sources\n    - ../common/sources",
        ),
        encoding="utf-8",
    )
    _use_source(project_root, "common.fs")

    hash_one = _artifact_hashes(project_yaml)

    (common_root / "sources" / "common.yaml").write_text(
        "\n".join(
            [
                "id: common.fs",
                "parser:",
                "  entrypoint: core.identity",
                "loader:",
                "  transport: fs",
                "  path: changed.jsonl",
                "  reader:",
                "    format: jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    hash_two = _artifact_hashes(project_yaml)

    assert hash_one != hash_two


def test_artifact_hash_ignores_source_document_order(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(project_root)
    data = project_root / "data"
    data.mkdir()
    (data / "alpha.jsonl").write_text("alpha\n", encoding="utf-8")
    (data / "zeta.jsonl").write_text("zeta\n", encoding="utf-8")
    alpha_path = project_root / "sources" / "alpha.yaml"
    zeta_path = project_root / "sources" / "zeta.yaml"
    alpha_path.write_text(
        "id: zeta\n"
        "inputs: {files: [data/zeta.jsonl]}\n"
        "parser: {entrypoint: core.identity}\n"
        "loader: {entrypoint: custom.loader}\n",
        encoding="utf-8",
    )
    zeta_path.write_text(
        "id: alpha\n"
        "inputs: {files: [data/alpha.jsonl]}\n"
        "parser: {entrypoint: core.identity}\n"
        "loader: {entrypoint: custom.loader}\n",
        encoding="utf-8",
    )
    first = load_project_definition(project_yaml)

    temporary_path = project_root / "sources" / "temporary.yaml"
    alpha_path.rename(temporary_path)
    zeta_path.rename(alpha_path)
    temporary_path.rename(zeta_path)
    second = load_project_definition(project_yaml)

    assert second.streams == first.streams
    assert second.artifact_hashes == first.artifact_hashes


@pytest.mark.parametrize(
    ("source_yaml", "source_id", "tracks_inventory"),
    [
        pytest.param(
            """\
id: sample.fs
parser: {entrypoint: core.identity, args: {}}
loader:
  transport: fs
  path: data/*.jsonl
  reader:
    format: jsonl
""",
            "sample.fs",
            True,
            id="built-in-glob",
        ),
        pytest.param(
            """\
id: sample.file
parser: {entrypoint: core.identity, args: {}}
loader: {transport: fs, path: data/a.jsonl, reader: {format: jsonl}}
""",
            "sample.file",
            False,
            id="built-in-file",
        ),
        pytest.param(
            """\
id: sample.custom
inputs:
  files: [data/*.jsonl]
parser: {entrypoint: core.identity, args: {}}
loader: {entrypoint: custom.loader, args: {}}
""",
            "sample.custom",
            True,
            id="custom-loader",
        ),
    ],
)
def test_artifact_hash_tracks_local_source_snapshot(
    tmp_path: Path,
    source_yaml: str,
    source_id: str,
    tracks_inventory: bool,
) -> None:
    project_root = tmp_path / "project"
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(project_root)
    (project_root / "sources" / "sample.yaml").write_text(
        source_yaml,
        encoding="utf-8",
    )
    _use_source(project_root, source_id)
    data_dir = project_root / "data"
    data_dir.mkdir()
    first = data_dir / "a.jsonl"
    first.write_text("one\n", encoding="utf-8")

    baseline_definition = load_project_definition(project_yaml)
    baseline = baseline_definition.artifact_hashes
    assert _artifact_hashes(project_yaml) == baseline

    (data_dir / "ignored.csv").write_text("ignored\n", encoding="utf-8")
    assert _artifact_hashes(project_yaml) == baseline

    previous = first.stat()
    first.write_text("two\n", encoding="utf-8")
    os.utime(
        first,
        ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000),
    )
    edited_definition = load_project_definition(project_yaml)
    edited = edited_definition.artifact_hashes
    assert edited != baseline

    second = data_dir / "b.jsonl"
    second.write_text("three\n", encoding="utf-8")
    added_definition = load_project_definition(project_yaml)
    added = added_definition.artifact_hashes
    assert (added != edited) is tracks_inventory

    second.unlink()
    restored_definition = load_project_definition(project_yaml)
    assert restored_definition.artifact_hashes == edited

    first.unlink()
    assert _artifact_hashes(project_yaml) != edited


def test_artifact_hash_rejects_source_input_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(project_root)
    (project_root / "sources" / "sample.yaml").write_text(
        """\
id: sample.custom
inputs: {files: [data]}
parser: {entrypoint: core.identity, args: {}}
loader: {entrypoint: custom.loader, args: {}}
""",
        encoding="utf-8",
    )
    (project_root / "data").mkdir()
    _use_source(project_root, "sample.custom")

    with pytest.raises(ValueError, match="not a regular file"):
        _artifact_hashes(project_yaml)


@pytest.mark.parametrize(
    ("source_yaml", "invalid_field"),
    [
        (
            """\
id: sample.fs
loader: {transport: fs, path: data/a.jsonl, reader: {format: jsonl}}
""",
            "parser",
        ),
        (
            """\
id: sample.fs
parser: {entrypoint: core.identity, args: {}}
""",
            "loader",
        ),
        (
            """\
id: sample.fs
parser: core.identity
loader: {transport: fs, path: data/a.jsonl, reader: {format: jsonl}}
""",
            "parser",
        ),
    ],
)
def test_artifact_hash_rejects_malformed_source_config(
    tmp_path: Path,
    source_yaml: str,
    invalid_field: str,
) -> None:
    project_root = tmp_path / "project"
    _write_project_files(project_root)
    project_yaml = _write_project_yaml(project_root)
    (project_root / "sources" / "malformed.yaml").write_text(
        source_yaml,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=invalid_field):
        _artifact_hashes(project_yaml)

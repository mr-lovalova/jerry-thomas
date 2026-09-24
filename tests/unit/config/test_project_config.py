import pytest
from pydantic import ValidationError

from jerrythomas.config.project import ProjectConfig
from jerrythomas.services.config_refs import project_vars_from_data


def _project_data(**overrides):
    data = {
        "schema_version": 7,
        "artifact_revision": 1,
        "name": "momentum",
        "paths": {
            "streams": "streams",
            "sources": "sources",
            "datasets": "datasets",
            "artifacts": "artifacts",
        },
    }
    data.update(overrides)
    return data


def test_project_config_accepts_variant() -> None:
    cfg = ProjectConfig.model_validate(_project_data(variant="long"))

    assert cfg.variant == "long"


def test_project_config_requires_artifact_revision() -> None:
    data = _project_data()
    del data["artifact_revision"]

    with pytest.raises(ValidationError, match="artifact_revision"):
        ProjectConfig.model_validate(data)


def test_project_config_requires_schema_version() -> None:
    data = _project_data()
    del data["schema_version"]

    with pytest.raises(ValidationError, match="schema_version"):
        ProjectConfig.model_validate(data)


def test_project_config_accepts_positive_artifact_revision() -> None:
    cfg = ProjectConfig.model_validate(_project_data(artifact_revision=3))

    assert cfg.artifact_revision == 3


@pytest.mark.parametrize("value", [0, -1, "2", 2.0, True])
def test_project_config_rejects_invalid_artifact_revision(value: object) -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(_project_data(artifact_revision=value))


def test_project_config_accepts_multiple_discovery_roots() -> None:
    cfg = ProjectConfig.model_validate(
        _project_data(
            paths={
                "streams": ["streams", "../common/streams"],
                "sources": ["sources", "../common/sources"],
                "datasets": "datasets",
                "artifacts": "artifacts",
            }
        )
    )

    assert cfg.paths.streams == ["streams", "../common/streams"]
    assert cfg.paths.sources == ["sources", "../common/sources"]


def test_project_config_defaults_profiles_path() -> None:
    cfg = ProjectConfig.model_validate(_project_data())

    assert cfg.paths.profiles == "./profiles"


def test_project_config_rejects_null_profiles_path() -> None:
    data = _project_data()
    data["paths"]["profiles"] = None

    with pytest.raises(ValidationError, match="paths.profiles"):
        ProjectConfig.model_validate(data)


def test_project_config_rejects_empty_discovery_roots() -> None:
    with pytest.raises(ValidationError, match="project path lists must not be empty"):
        ProjectConfig.model_validate(
            _project_data(
                paths={
                    "streams": [],
                    "sources": "sources",
                    "datasets": "datasets",
                    "artifacts": "artifacts",
                }
            )
        )


@pytest.mark.parametrize(
    "field",
    ["streams", "sources", "datasets", "artifacts", "operations", "profiles"],
)
def test_project_config_rejects_blank_paths(field: str) -> None:
    data = _project_data()
    data["paths"][field] = "   "

    with pytest.raises(ValidationError, match="at least 1 character"):
        ProjectConfig.model_validate(data)


@pytest.mark.parametrize("field", ["streams", "sources"])
def test_project_config_rejects_blank_discovery_root(field: str) -> None:
    data = _project_data()
    data["paths"][field] = ["valid", "   "]

    with pytest.raises(ValidationError, match="at least 1 character"):
        ProjectConfig.model_validate(data)


def test_project_fields_have_one_interpolation_name() -> None:
    vars_ = project_vars_from_data(_project_data(variant="long"))

    assert vars_["project_name"] == "momentum"
    assert vars_["project_variant"] == "long"
    assert "project" not in vars_
    assert "variant" not in vars_
    assert "schema_version" not in vars_


@pytest.mark.parametrize("field", ["project_name", "project_variant"])
def test_project_globals_cannot_override_project_variables(field: str) -> None:
    with pytest.raises(ValueError, match=f"reserved variable '{field}'"):
        project_vars_from_data(_project_data(globals={field: "override"}))


@pytest.mark.parametrize("field", ["start_time", "end_time"])
def test_project_globals_require_timezone_aware_bounds(field: str) -> None:
    with pytest.raises(
        ValidationError, match=f"globals.{field} must be timezone-aware"
    ):
        ProjectConfig.model_validate(
            _project_data(globals={field: "2024-01-01T00:00:00"})
        )


def test_project_globals_reject_reversed_time_range() -> None:
    with pytest.raises(
        ValidationError,
        match="globals.start_time must not be after globals.end_time",
    ):
        ProjectConfig.model_validate(
            _project_data(
                globals={
                    "start_time": "2024-01-02T00:00:00Z",
                    "end_time": "2024-01-01T00:00:00Z",
                }
            )
        )


def test_project_config_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProjectConfig.model_validate(_project_data(varaint="long"))


def test_project_config_rejects_unknown_path_fields() -> None:
    data = _project_data()
    data["paths"]["profiels"] = "profiles"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProjectConfig.model_validate(data)


def test_project_rejects_older_schema_version() -> None:
    with pytest.raises(ValidationError, match="Input should be 7"):
        ProjectConfig.model_validate(_project_data(schema_version=1))

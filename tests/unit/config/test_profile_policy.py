import pytest
from pydantic import ValidationError

from jerrythomas.config.observability import ObservabilityConfig
from jerrythomas.config.profiles.build import BuildProfile
from jerrythomas.config.profiles.defaults import (
    BuildProfileDefaults,
    InspectProfileDefaults,
    MaterializeProfileDefaults,
    ServeProfileDefaults,
)
from jerrythomas.profiles.loader import apply_profile_defaults


@pytest.mark.parametrize(
    "config",
    [BuildProfile, BuildProfileDefaults],
)
def test_build_mode_is_rejected_with_migration_guidance(config) -> None:
    values = {"cmd": "build", "mode": "auto", "artifact_mode": "rebuild"}
    if config is BuildProfile:
        values.update(name="metadata", operation="metadata")
    with pytest.raises(ValidationError, match="use artifact_mode"):
        config.model_validate(values)


@pytest.mark.parametrize(
    ("config", "cmd"),
    [
        (BuildProfileDefaults, "build"),
        (ServeProfileDefaults, "serve"),
        (InspectProfileDefaults, "inspect"),
        (MaterializeProfileDefaults, "materialize"),
    ],
)
@pytest.mark.parametrize(
    "value", ["AUTO", "FORCE", "OFF", "force", "off", " auto ", False]
)
def test_artifact_modes_reject_legacy_spellings(config, cmd, value) -> None:
    with pytest.raises(ValidationError, match="Input should be.*require_current"):
        config.model_validate({"cmd": cmd, "artifact_mode": value})


@pytest.mark.parametrize("mode", ["auto", "rebuild", "require_current"])
def test_build_profile_policy_overrides_defaults(mode) -> None:
    defaults = BuildProfileDefaults(
        cmd="build",
        artifact_mode="rebuild",
        observability=ObservabilityConfig(visuals=True),
    )
    profile = BuildProfile(
        cmd="build",
        name="metadata",
        operation="metadata",
        artifact_mode=mode,
        observability=ObservabilityConfig(visuals=False),
    )
    merged = apply_profile_defaults(profile, defaults)
    assert isinstance(merged, BuildProfile)
    assert merged.artifact_mode == mode
    assert merged.observability.visuals is False


@pytest.mark.parametrize("value", ["ON", "OFF", "on", "off", "true", "false", 0, 1])
def test_visuals_rejects_strings_and_numbers(value) -> None:
    with pytest.raises(ValidationError, match="valid boolean"):
        ObservabilityConfig(visuals=value)

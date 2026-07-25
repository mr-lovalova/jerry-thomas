from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic.type_adapter import TypeAdapter

from datapipeline.config.profiles import (
    BuildProfile,
    BuildProfileDefaults,
    InspectProfile,
    InspectProfileDefaults,
    MaterializeProfile,
    MaterializeProfileDefaults,
    Profile,
    ProfileCommand,
    ProfileDefaults,
    ServeProfile,
    ServeProfileDefaults,
)
from datapipeline.services.definitions import ProjectManifest
from datapipeline.utils.load import read_yaml_document

ProfileModel = Annotated[
    ServeProfile | BuildProfile | InspectProfile | MaterializeProfile,
    Field(discriminator="cmd"),
]

PROFILE_KINDS: tuple[ProfileCommand, ...] = (
    "serve",
    "build",
    "inspect",
    "materialize",
)
PROFILE_ADAPTER: TypeAdapter[ProfileModel] = TypeAdapter(ProfileModel)
ProfileDefaultsModel = Annotated[
    ServeProfileDefaults
    | BuildProfileDefaults
    | InspectProfileDefaults
    | MaterializeProfileDefaults,
    Field(discriminator="cmd"),
]
PROFILE_DEFAULTS_ADAPTER: TypeAdapter[ProfileDefaultsModel] = TypeAdapter(
    ProfileDefaultsModel
)


def _load_profile_doc(path: Path, project: ProjectManifest):
    document = read_yaml_document(path, require_mapping=False)
    return project.resolve_config(document.data)


def _profile_identity_from_filename(
    path: Path,
) -> tuple[ProfileCommand, str] | None:
    stem = path.stem.strip()
    if not stem or "." not in stem:
        return None
    prefix, name = stem.split(".", 1)
    command = prefix.lower()
    name = name.strip()
    if not name:
        return None
    for profile_command in PROFILE_KINDS:
        if command == profile_command:
            return profile_command, name
    return None


def _validate_profile_layout(root: Path) -> None:
    if not root.exists() or root.is_file():
        return
    nested_files = sorted(
        path for path in root.rglob("*.y*ml") if path.is_file() and path.parent != root
    )
    if nested_files:
        listed = ", ".join(str(path.relative_to(root)) for path in nested_files)
        raise ValueError(
            "Profile files must be flat under profiles/ using "
            "{serve,build,inspect,materialize}.<name|defaults>.yaml naming; "
            f"found nested profile files: {listed}"
        )

    invalid = sorted(
        path
        for path in root.glob("*.y*ml")
        if path.is_file() and _profile_identity_from_filename(path) is None
    )
    if not invalid:
        return
    listed = ", ".join(str(path.relative_to(root)) for path in invalid)
    raise ValueError(
        "Profile files must use {serve,build,inspect,materialize}.<name|defaults>.yaml "
        "naming under profiles/; "
        f"found invalid profile locations: {listed}"
    )


def _load_profile_specs(
    project: ProjectManifest,
    command: ProfileCommand | None = None,
) -> tuple[list[Profile], dict[str, ProfileDefaults]]:
    root = project.profiles_dir
    _validate_profile_layout(root)
    specs: list[Profile] = []
    defaults_by_kind: dict[str, ProfileDefaults] = {}
    default_paths: dict[str, Path] = {}
    profile_paths: dict[tuple[str, str], Path] = {}
    for path in sorted(root.glob("*.y*ml")):
        identity = _profile_identity_from_filename(path)
        if identity is None:
            continue
        expected_kind, profile_name = identity
        if command is not None and expected_kind != command:
            continue
        if profile_name.lower() == "defaults":
            doc = _load_profile_doc(path, project)
            if not isinstance(doc, dict):
                raise TypeError(f"{path} must define a mapping profile defaults.")
            if "cmd" in doc:
                raise ValueError(
                    "Profile command comes from the defaults filename; "
                    "remove the 'cmd' key."
                )
            defaults = PROFILE_DEFAULTS_ADAPTER.validate_python(
                {"cmd": expected_kind, **doc}
            )
            existing = defaults_by_kind.get(expected_kind)
            if existing is not None:
                raise ValueError(
                    f"Duplicate {expected_kind} defaults are not allowed: "
                    f"{default_paths[expected_kind]}, {path}"
                )
            defaults_by_kind[expected_kind] = defaults
            default_paths[expected_kind] = path
            continue
        doc = _load_profile_doc(path, project)
        if not isinstance(doc, dict):
            raise TypeError(f"{path} must define one mapping profile.")
        if "cmd" in doc or "name" in doc:
            raise ValueError(
                "Profile command and name come from the filename; "
                "remove the 'cmd' and 'name' keys."
            )
        spec = PROFILE_ADAPTER.validate_python(
            {"cmd": expected_kind, "name": profile_name, **doc}
        )
        identity_key = (expected_kind, profile_name)
        existing_path = profile_paths.get(identity_key)
        if existing_path is not None:
            raise ValueError(
                f"Duplicate {expected_kind} profile names are not allowed: "
                f"{profile_name} ({existing_path}, {path})"
            )
        profile_paths[identity_key] = path
        specs.append(spec)
    return specs, defaults_by_kind


def profile_specs(
    project: ProjectManifest,
    cmd: ProfileCommand | None = None,
) -> list[Profile]:
    if cmd is not None:
        profiles, _ = profile_specs_with_defaults(project, cmd=cmd)
        return profiles

    specs, _ = _load_profile_specs(project)
    grouped = _group_profiles(specs)
    return [profile for kind in PROFILE_KINDS for profile in grouped[kind]]


def profile_specs_with_defaults(
    project: ProjectManifest,
    cmd: ProfileCommand,
) -> tuple[list[Profile], ProfileDefaults]:
    specs, defaults_by_kind = _load_profile_specs(project, command=cmd)
    grouped = _group_profiles(specs)
    defaults = defaults_by_kind.get(cmd)
    if defaults is None:
        defaults = PROFILE_DEFAULTS_ADAPTER.validate_python({"cmd": cmd})
    return list(grouped[cmd]), defaults


def apply_profile_defaults(
    profile: Profile,
    defaults: ProfileDefaults,
) -> Profile:
    if profile.cmd != defaults.cmd:
        raise ValueError(
            f"Cannot apply {defaults.cmd} defaults to {profile.cmd} profile '{profile.name}'."
        )

    defaults_payload = defaults.model_dump(
        exclude_unset=True,
        exclude_none=True,
        exclude={"execution", "artifact_mode"},
    )
    profile_payload = profile.model_dump(exclude_unset=True)
    merged_payload = {**defaults_payload, **profile_payload}

    default_observability = defaults_payload.get("observability")
    profile_observability = profile_payload.get("observability")
    if isinstance(default_observability, dict) and isinstance(
        profile_observability, dict
    ):
        observability = {**default_observability, **profile_observability}
        default_logging = default_observability.get("logging")
        profile_logging = profile_observability.get("logging")
        if isinstance(default_logging, dict) and isinstance(profile_logging, dict):
            observability["logging"] = {**default_logging, **profile_logging}
        merged_payload["observability"] = observability

    return PROFILE_ADAPTER.validate_python(merged_payload)


def _ordered_profiles(specs: list[Profile]) -> list[Profile]:
    ordered = [spec for spec in specs if spec.order is not None]
    unordered = [spec for spec in specs if spec.order is None]
    ordered.sort(key=lambda spec: (spec.order, spec.name))
    return ordered + unordered


def _group_profiles(specs: list[Profile]) -> dict[str, list[Profile]]:
    grouped: dict[str, list[Profile]] = {kind: [] for kind in PROFILE_KINDS}
    for spec in specs:
        grouped[spec.cmd].append(spec)
    for kind_specs in grouped.values():
        kind_specs[:] = _ordered_profiles(kind_specs)
    return grouped

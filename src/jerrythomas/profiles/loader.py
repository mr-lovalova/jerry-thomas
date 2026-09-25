from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic.type_adapter import TypeAdapter

from jerrythomas.config.profiles.base import OperationProfile, Profile, ProfileCommand
from jerrythomas.config.profiles.build import BuildProfile
from jerrythomas.config.profiles.defaults import (
    BuildProfileDefaults,
    InspectProfileDefaults,
    ExportProfileDefaults,
    ProfileDefaults,
    ServeProfileDefaults,
)
from jerrythomas.config.profiles.inspect import InspectProfile
from jerrythomas.config.profiles.export import ExportProfile
from jerrythomas.config.profiles.serve import ServeProfile
from jerrythomas.services.definitions import ProjectDefinition, ProjectManifest
from jerrythomas.io.yaml import read_yaml_document

ProfileModel = Annotated[
    ServeProfile | BuildProfile | InspectProfile | ExportProfile,
    Field(discriminator="cmd"),
]

PROFILE_KINDS: tuple[ProfileCommand, ...] = (
    "serve",
    "build",
    "inspect",
    "export",
)
PROFILE_ADAPTER: TypeAdapter[ProfileModel] = TypeAdapter(ProfileModel)
ProfileDefaultsModel = Annotated[
    ServeProfileDefaults
    | BuildProfileDefaults
    | InspectProfileDefaults
    | ExportProfileDefaults,
    Field(discriminator="cmd"),
]
PROFILE_DEFAULTS_ADAPTER: TypeAdapter[ProfileDefaultsModel] = TypeAdapter(
    ProfileDefaultsModel
)


def _load_profile_doc(path: Path, project: ProjectManifest):
    document = read_yaml_document(path, require_mapping=False)
    return project.resolve_config(
        document.data,
        variables={
            "dataset_id": "${dataset_id}",
            "dataset_version": "${dataset_version}",
        },
    )


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
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError(f"Profiles path must be a directory: {root}")
    nested_files = sorted(
        path for path in root.rglob("*.y*ml") if path.is_file() and path.parent != root
    )
    if nested_files:
        listed = ", ".join(str(path.relative_to(root)) for path in nested_files)
        raise ValueError(
            "Profile files must be flat under profiles/ using "
            "{serve,build,inspect,export}.<name|defaults>.yaml naming; "
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
        "Profile files must use {serve,build,inspect,export}.<name|defaults>.yaml "
        "naming under profiles/; "
        f"found invalid profile locations: {listed}"
    )


def _load_profile_specs(
    project: ProjectManifest,
    command: ProfileCommand,
) -> tuple[list[Profile], ProfileDefaults | None]:
    root = project.profiles_dir
    _validate_profile_layout(root)
    specs: list[Profile] = []
    defaults: ProfileDefaults | None = None
    defaults_path: Path | None = None
    profile_paths: dict[str, Path] = {}
    for path in sorted(root.glob("*.y*ml")):
        identity = _profile_identity_from_filename(path)
        if identity is None:
            continue
        expected_kind, profile_name = identity
        if expected_kind != command:
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
            loaded_defaults = PROFILE_DEFAULTS_ADAPTER.validate_python(
                {"cmd": expected_kind, **doc}
            )
            if defaults is not None:
                raise ValueError(
                    f"Duplicate {expected_kind} defaults are not allowed: "
                    f"{defaults_path}, {path}"
                )
            defaults = loaded_defaults
            defaults_path = path
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
        existing_path = profile_paths.get(profile_name)
        if existing_path is not None:
            raise ValueError(
                f"Duplicate {expected_kind} profile names are not allowed: "
                f"{profile_name} ({existing_path}, {path})"
            )
        profile_paths[profile_name] = path
        specs.append(spec)
    return specs, defaults


def profile_specs_with_defaults(
    project: ProjectManifest,
    cmd: ProfileCommand,
) -> tuple[list[Profile], ProfileDefaults]:
    specs, defaults = _load_profile_specs(project, command=cmd)
    if defaults is None:
        defaults = PROFILE_DEFAULTS_ADAPTER.validate_python({"cmd": cmd})
    return _ordered_profiles(specs), defaults


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
        exclude={"execution"}
        if profile.cmd == "build"
        else {"execution", "artifact_mode"},
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


def bind_profile(
    profile: Profile,
    definition: ProjectDefinition,
    *,
    dataset_version: str | None = None,
) -> Profile:
    """Resolve dataset variables after operation selection.

    A version override is used only to locate previously saved runs.
    """
    if not isinstance(profile, OperationProfile):
        raise TypeError("Profiles must select an operation")
    operations = {
        **definition.artifact_graph.tasks_by_id,
        **{task.id: task for task in definition.runtime_operations},
    }
    operation = operations.get(profile.operation)
    if operation is None:
        return profile
    variables = {}
    if operation.dataset is not None:
        dataset = definition.require_dataset(operation.dataset)
        variables = {
            "dataset_id": operation.dataset,
            "dataset_version": dataset_version
            if dataset_version is not None
            else dataset.version,
        }
    payload = definition.project.resolve_config(
        profile.model_dump(mode="json"), variables=variables
    )
    return type(profile).model_validate(payload)

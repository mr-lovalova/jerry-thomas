from dataclasses import dataclass, replace
from pathlib import Path

from jerrythomas.config.streams import SourceStreamConfig
from jerrythomas.io.sinks.files import AtomicBinaryFileSink
from jerrythomas.services.project import load_project
from jerrythomas.services.scaffold.domain import (
    create_domain,
    domain_scaffold_paths,
    validate_domain_creation,
)
from jerrythomas.services.scaffold.dto import (
    create_dto,
    dto_scaffold_paths,
    validate_dto_creation,
)
from jerrythomas.services.scaffold.layout import ep_key_from_name
from jerrythomas.services.scaffold.locking import acquire_scaffold_lock
from jerrythomas.services.scaffold.mapper import (
    create_mapper,
    mapper_scaffold_paths,
    validate_mapper_creation,
)
from jerrythomas.services.scaffold.parser import (
    create_parser,
    parser_scaffold_paths,
    validate_parser_creation,
)
from jerrythomas.services.scaffold.paths import pkg_root
from jerrythomas.services.scaffold.source_yaml import (
    create_source_yaml,
    validate_source_id,
)
from jerrythomas.services.scaffold.stream_yaml import write_source_stream
from jerrythomas.services.scaffold.utils import (
    is_python_identifier,
    rollback_new_scaffold_paths,
)
from jerrythomas.services.streams.loader import (
    declared_source_ids,
    declared_stream_ids,
)


@dataclass(frozen=True)
class PythonType:
    class_name: str
    module: str


@dataclass(frozen=True)
class ParserReference:
    entrypoint: str


@dataclass(frozen=True)
class ParserCreation:
    name: str
    dto: PythonType


ParserPlan = ParserReference | ParserCreation


@dataclass(frozen=True)
class MapperReference:
    entrypoint: str


@dataclass(frozen=True)
class MapperCreation:
    name: str
    input_type: PythonType


MapperPlan = MapperReference | MapperCreation


@dataclass(frozen=True)
class SourceReference:
    source_id: str


@dataclass(frozen=True)
class SourceCreation:
    source_id: str
    loader: dict[str, object]
    parser: ParserPlan


SourcePlan = SourceReference | SourceCreation


@dataclass(frozen=True)
class DomainReference:
    name: str


@dataclass(frozen=True)
class DomainCreation:
    name: str


DomainPlan = DomainReference | DomainCreation


@dataclass(frozen=True)
class StreamPlan:
    project_yaml: Path
    stream_id: str
    root: Path | None
    source: SourcePlan
    mapper: MapperPlan
    domain: DomainPlan
    dto_to_create: str | None


@dataclass(frozen=True)
class StreamPlanResult:
    path: Path
    entry_points_changed: bool


def _validated_stream_config(plan: StreamPlan) -> SourceStreamConfig:
    if isinstance(plan.source, SourceCreation):
        validate_source_id(plan.source.source_id)
        if isinstance(plan.source.parser, ParserCreation):
            if not is_python_identifier(plan.source.parser.dto.class_name):
                raise ValueError("DTO name must be a valid Python identifier")

    if not is_python_identifier(plan.domain.name):
        raise ValueError("Domain name must be a valid Python identifier")

    mapper_entrypoint = (
        ep_key_from_name(plan.mapper.name)
        if isinstance(plan.mapper, MapperCreation)
        else plan.mapper.entrypoint
    )
    return SourceStreamConfig.model_validate(
        {
            "id": plan.stream_id,
            "from": {"source": plan.source.source_id},
            "map": {"entrypoint": mapper_entrypoint},
        }
    )


def _validate_component_creation(plan: StreamPlan) -> None:
    if isinstance(plan.source, SourceCreation) and isinstance(
        plan.source.parser, ParserCreation
    ):
        validate_parser_creation(plan.source.parser.name, plan.root)
    if isinstance(plan.domain, DomainCreation):
        validate_domain_creation(plan.domain.name, plan.root)
    if plan.dto_to_create is not None:
        validate_dto_creation(plan.dto_to_create, plan.root)
    if isinstance(plan.mapper, MapperCreation):
        validate_mapper_creation(plan.mapper.name, plan.root)


def _validate_config_creation(
    plan: StreamPlan,
    config: SourceStreamConfig,
) -> None:
    if not plan.project_yaml.exists():
        return

    project = load_project(plan.project_yaml)
    if isinstance(plan.source, SourceCreation):
        source_path = project.source_dirs[0] / f"{plan.source.source_id}.yaml"
        if source_path.exists():
            raise FileExistsError(f"{source_path} already exists")
    destination = project.stream_dirs[0] / f"{config.id}.yaml"
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    existing_project = replace(
        project,
        source_dirs=tuple(path for path in project.source_dirs if path.exists()),
        stream_dirs=tuple(path for path in project.stream_dirs if path.exists()),
    )
    if isinstance(
        plan.source, SourceCreation
    ) and plan.source.source_id in declared_source_ids(existing_project):
        raise FileExistsError(f"Source id '{plan.source.source_id}' already exists")
    if config.id in declared_stream_ids(existing_project):
        raise FileExistsError(f"Stream id '{config.id}' already exists")


def _stream_plan_paths(
    plan: StreamPlan,
    config: SourceStreamConfig,
) -> tuple[Path, ...]:
    paths: list[Path] = []

    if isinstance(plan.domain, DomainCreation):
        paths.extend(domain_scaffold_paths(plan.domain.name, plan.root))
    if plan.dto_to_create is not None:
        paths.extend(dto_scaffold_paths(plan.dto_to_create, plan.root))
    if isinstance(plan.source, SourceCreation) and isinstance(
        plan.source.parser, ParserCreation
    ):
        paths.extend(parser_scaffold_paths(plan.source.parser.name, plan.root))
    if isinstance(plan.mapper, MapperCreation):
        paths.extend(mapper_scaffold_paths(plan.mapper.name, plan.root))

    if plan.project_yaml.exists():
        project = load_project(plan.project_yaml)
        source_dirs = project.source_dirs
        stream_dirs = project.stream_dirs
        config_dirs = [*source_dirs, *stream_dirs, project.profiles_dir]
        if project.operations_dir is not None:
            config_dirs.append(project.operations_dir)
    else:
        project_dir = plan.project_yaml.parent
        source_dirs = (project_dir / "sources",)
        stream_dirs = (project_dir / "streams",)
        config_dirs = [*source_dirs, *stream_dirs, project_dir / "profiles"]

    paths.extend(config_dirs)
    paths.extend((plan.project_yaml, plan.project_yaml.parent / ".env.example"))
    if isinstance(plan.source, SourceCreation):
        paths.append(source_dirs[0] / f"{plan.source.source_id}.yaml")
    paths.append(stream_dirs[0] / f"{config.id}.yaml")
    return tuple(paths)


def _restore_pyproject(path: Path, content: bytes) -> None:
    sink = AtomicBinaryFileSink(path.resolve())
    try:
        sink.write_bytes(content)
        sink.close()
    except BaseException:
        sink.abort()
        raise


def execute_stream_plan(plan: StreamPlan) -> StreamPlanResult:
    config = _validated_stream_config(plan)
    _, _, pyproject_path = pkg_root(plan.root)

    with acquire_scaffold_lock(pyproject_path.parent) as plugin_lock:
        with acquire_scaffold_lock(
            plan.project_yaml.parent,
            plugin_lock,
        ) as project_lock:
            _validate_component_creation(plan)
            _validate_config_creation(plan, config)
            before_pyproject = pyproject_path.read_bytes()
            entry_points_changed = isinstance(plan.mapper, MapperCreation) or (
                isinstance(plan.source, SourceCreation)
                and isinstance(plan.source.parser, ParserCreation)
            )
            with rollback_new_scaffold_paths(_stream_plan_paths(plan, config)):
                try:
                    if isinstance(plan.domain, DomainCreation):
                        create_domain(
                            plan.domain.name,
                            plan.root,
                            scaffold_lock=plugin_lock,
                        )

                    if plan.dto_to_create is not None:
                        create_dto(
                            plan.dto_to_create,
                            plan.root,
                            scaffold_lock=plugin_lock,
                        )

                    if isinstance(plan.source, SourceCreation):
                        if isinstance(plan.source.parser, ParserCreation):
                            parser_entrypoint = create_parser(
                                name=plan.source.parser.name,
                                dto_class=plan.source.parser.dto.class_name,
                                dto_module=plan.source.parser.dto.module,
                                root=plan.root,
                                scaffold_lock=plugin_lock,
                            )
                        else:
                            parser_entrypoint = plan.source.parser.entrypoint

                    if isinstance(plan.mapper, MapperCreation):
                        mapper_entrypoint = create_mapper(
                            name=plan.mapper.name,
                            input_class=plan.mapper.input_type.class_name,
                            input_module=plan.mapper.input_type.module,
                            domain=plan.domain.name,
                            root=plan.root,
                            scaffold_lock=plugin_lock,
                        )
                    else:
                        mapper_entrypoint = plan.mapper.entrypoint

                    if isinstance(plan.source, SourceCreation):
                        create_source_yaml(
                            source_id=plan.source.source_id,
                            loader=plan.source.loader,
                            parser_ep=parser_entrypoint,
                            root=plan.root,
                            project_yaml=plan.project_yaml,
                            scaffold_lock=project_lock,
                        )

                    path = write_source_stream(
                        project_yaml=plan.project_yaml,
                        stream_id=plan.stream_id,
                        source=plan.source.source_id,
                        mapper_entrypoint=mapper_entrypoint,
                        scaffold_lock=project_lock,
                    )
                except BaseException:
                    if entry_points_changed:
                        _restore_pyproject(pyproject_path, before_pyproject)
                    raise

            return StreamPlanResult(
                path=path,
                entry_points_changed=entry_points_changed,
            )

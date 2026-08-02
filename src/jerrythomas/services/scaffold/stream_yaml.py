from pathlib import Path

from jerrythomas.config.streams import (
    AlignedStreamConfig,
    BroadcastStreamConfig,
    SourceStreamConfig,
)
from jerrythomas.services.project import load_project
from jerrythomas.services.scaffold.locking import ScaffoldLock, acquire_scaffold_lock
from jerrythomas.services.scaffold.paths import ensure_project_scaffold
from jerrythomas.services.scaffold.templates import render
from jerrythomas.services.scaffold.utils import write_new_file
from jerrythomas.services.streams.loader import declared_stream_ids


def _new_stream_path(
    project_yaml: Path,
    stream_id: str,
    scaffold_lock: ScaffoldLock,
) -> Path:
    if project_yaml.exists():
        project = load_project(project_yaml)
        path = project.stream_dirs[0] / f"{stream_id}.yaml"
        if path.exists():
            raise FileExistsError(f"{path} already exists")
    project = ensure_project_scaffold(project_yaml, scaffold_lock)
    if stream_id in declared_stream_ids(project):
        raise FileExistsError(f"Stream id '{stream_id}' already exists")
    return project.stream_dirs[0] / f"{stream_id}.yaml"


def write_source_stream(
    project_yaml: Path,
    stream_id: str,
    source: str,
    mapper_entrypoint: str,
    scaffold_lock: ScaffoldLock | None = None,
) -> Path:
    with acquire_scaffold_lock(project_yaml.parent, scaffold_lock) as project_lock:
        config = SourceStreamConfig.model_validate(
            {
                "id": stream_id,
                "from": {"source": source},
                "map": {"entrypoint": mapper_entrypoint},
            }
        )
        path = _new_stream_path(project_yaml, config.id, project_lock)
        content = render(
            "streams/source.yaml.j2",
            source=config.from_.source,
            stream_id=config.id,
            mapper_entrypoint=config.map.entrypoint,
        )
        write_new_file(path, content)
        return path


def write_aligned_stream(
    project_yaml: Path,
    stream_id: str,
    input_streams: list[str],
    combine_entrypoint: str,
    scaffold_lock: ScaffoldLock | None = None,
) -> Path:
    with acquire_scaffold_lock(project_yaml.parent, scaffold_lock) as project_lock:
        config = AlignedStreamConfig.model_validate(
            {
                "id": stream_id,
                "from": {"align": input_streams},
                "combine": {"entrypoint": combine_entrypoint},
            }
        )
        path = _new_stream_path(project_yaml, config.id, project_lock)
        content = (
            render(
                "streams/aligned.yaml.j2",
                stream_id=config.id,
                input_streams=list(config.from_.align),
                combine_entrypoint=config.combine.entrypoint,
            ).strip()
            + "\n"
        )
        write_new_file(path, content)
        return path


def write_broadcast_stream(
    project_yaml: Path,
    stream_id: str,
    primary_stream: str,
    broadcast_stream: str,
    combine_entrypoint: str,
    scaffold_lock: ScaffoldLock | None = None,
) -> Path:
    with acquire_scaffold_lock(project_yaml.parent, scaffold_lock) as project_lock:
        config = BroadcastStreamConfig.model_validate(
            {
                "id": stream_id,
                "from": {
                    "stream": primary_stream,
                    "broadcast": broadcast_stream,
                },
                "combine": {"entrypoint": combine_entrypoint},
            }
        )
        path = _new_stream_path(project_yaml, config.id, project_lock)
        content = (
            render(
                "streams/broadcast.yaml.j2",
                stream_id=config.id,
                primary_stream=config.from_.stream,
                broadcast_stream=config.from_.broadcast,
                combine_entrypoint=config.combine.entrypoint,
            ).strip()
            + "\n"
        )
        write_new_file(path, content)
        return path

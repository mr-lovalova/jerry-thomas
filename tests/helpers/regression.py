import json
from pathlib import Path
from typing import Literal

from jerrythomas.config.profiles.output import ServeOutputConfig
from jerrythomas.config.preview import PreviewStage
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.models import RuntimeRunRequest
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import build_runtime_run_request


def serve_dataset(
    project_root: Path,
    artifact_mode: Literal["AUTO", "FORCE"] = "FORCE",
    cli_output: ServeOutputConfig | None = None,
    preview: PreviewStage | None = None,
    limit: int | None = None,
) -> RuntimeRunRequest:
    request = build_runtime_run_request(
        "serve",
        str(project_root / "project.yaml"),
        profile_name="dataset",
        artifact_mode=artifact_mode,
        limit=limit,
        preview=preview,
        cli_output=cli_output,
        command_observability=CommandObservability(
            visuals="off",
            log_level="CRITICAL",
        ),
    )
    assert request is not None
    assert len(request.serve_run_plans) == 1
    run_profiles(request)
    return request


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

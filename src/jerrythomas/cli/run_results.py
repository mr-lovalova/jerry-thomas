"""Completed profile results and stdout reservation for machine-readable output."""

import json
from dataclasses import asdict
from typing import Sequence

from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.models import (
    MaterializeRunRequest,
    MaterializeRunResult,
    ProfileRunResult,
    RuntimeRunRequest,
    ServeRunResult,
)


def validate_result_json_outputs(
    request: RuntimeRunRequest | MaterializeRunRequest,
) -> None:
    if any(job.output.transport == "stdout" for job in request.jobs):
        raise ProfileCommandError(
            "--result-json requires filesystem data outputs; stdout is reserved for run results."
        )
    log_outputs = (
        request.artifact_settings.observability.log_output,
        *(job.observability.log_output for job in request.jobs),
    )
    if any(
        target.transport == "stdout"
        for output in log_outputs
        for target in output.outputs
    ):
        raise ProfileCommandError(
            "--result-json cannot use stdout logging; select stderr or filesystem logs."
        )


def write_run_results(results: Sequence[ProfileRunResult]) -> None:
    runs: list[dict[str, object]] = []
    for result in results:
        if isinstance(result, ServeRunResult):
            runs.append(
                {
                    "run_id": result.paths.run_id,
                    "directory": str(result.paths.run_root),
                    "preview": result.preview,
                    "outputs": [str(path) for path in result.outputs],
                }
            )
        elif isinstance(result, MaterializeRunResult):
            runs.append(
                {
                    "command": "materialize",
                    "run_id": result.run_id,
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                    "status": "success",
                    "outputs": [
                        {**asdict(output), "path": str(output.path)}
                        for output in result.outputs
                    ],
                }
            )
        else:
            raise TypeError(f"Unsupported run result: {type(result).__name__}")
    print(json.dumps({"schema_version": 1, "runs": runs}))

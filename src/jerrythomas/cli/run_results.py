"""Completed profile results and stdout reservation for machine-readable output."""

import json
from typing import Sequence

from jerrythomas.io.runs import SavedRun
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.models import (
    ExportRunRequest,
    RuntimeRunRequest,
)


def validate_result_json_outputs(
    request: RuntimeRunRequest | ExportRunRequest,
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


def write_run_results(results: Sequence[SavedRun]) -> None:
    print(
        json.dumps(
            {
                "schema_version": 3,
                "runs": [
                    {
                        "receipt": str(result.metadata_path),
                        **result.metadata.model_dump(mode="json"),
                    }
                    for result in results
                ],
            }
        )
    )

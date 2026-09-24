import pytest
from pydantic import ValidationError

from jerrythomas.config.tasks.scaler import ScalerTask


def test_scaler_task_binds_dataset_and_output() -> None:
    task = ScalerTask()

    assert task.model_dump() == {
        "kind": "artifact",
        "id": "scaler",
        "entrypoint": "core.artifact.scaler",
        "output": "build/scaler.json",
        "dataset": None,
    }


def test_scaler_task_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScalerTask.model_validate({"unexpected": True})

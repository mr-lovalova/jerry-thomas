import pytest
from pydantic import ValidationError

from jerrythomas.config.tasks.scaler import ScalerTask


def test_scaler_task_defaults_its_path_from_the_dataset() -> None:
    assert ScalerTask().model_dump() == {
        "kind": "artifact",
        "id": "scaler",
        "entrypoint": "core.scaler",
        "path": "scaler.json",
        "dataset": None,
    }
    assert ScalerTask(dataset="alpha").path == "datasets/alpha/scaler.json"
    assert ScalerTask(dataset="alpha", path="custom.json").path == "custom.json"


def test_scaler_task_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScalerTask.model_validate({"unexpected": True})

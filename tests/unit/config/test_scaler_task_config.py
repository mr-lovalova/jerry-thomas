import pytest
from pydantic import ValidationError

from jerrythomas.config.tasks.scaler import ScalerTask


def test_scaler_task_has_only_fitting_options() -> None:
    task = ScalerTask()

    assert task.model_dump() == {
        "kind": "artifact",
        "id": "scaler",
        "entrypoint": "core.artifact.scaler",
        "output": "build/scaler.json",
        "with_mean": True,
        "with_std": True,
        "epsilon": 1e-12,
    }


def test_scaler_task_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScalerTask.model_validate({"unexpected": True})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("with_mean", "true"),
        ("with_std", 1),
        ("epsilon", 0),
        ("epsilon", float("inf")),
    ],
)
def test_scaler_task_options_are_strict(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ScalerTask.model_validate({field: value})

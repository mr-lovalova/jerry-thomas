from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from jerrythomas.artifacts.models import (
    FoldedMetadataLayout,
    ListVectorMetadataEntry,
    SampleMetadata,
    ScalarVectorMetadataEntry,
    UnsplitMetadataLayout,
    VectorMetadata,
    VectorSchema,
    Window,
)


def _time(hour: int) -> datetime:
    return datetime(2024, 1, 1, hour, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "change",
    [
        {"start": _time(2), "end": _time(1)},
        {"mode": "unknown"},
        {"mode": "relaxed"},
        {"size": 0},
    ],
)
def test_metadata_window_rejects_invalid_contract(change: dict[str, object]) -> None:
    payload = {
        "start": _time(1),
        "end": _time(2),
        "mode": "union",
        "size": 2,
        **change,
    }

    with pytest.raises(ValidationError):
        Window.model_validate(payload)


@pytest.mark.parametrize(
    ("first", "last"),
    [
        (_time(1), None),
        (None, _time(1)),
        (_time(2), _time(1)),
    ],
)
def test_vector_metadata_rejects_invalid_observation_bounds(first, last) -> None:
    with pytest.raises(ValidationError, match="observation bounds|first_observed"):
        ScalarVectorMetadataEntry(
            id="price",
            base_id="price",
            kind="scalar",
            present_count=1,
            null_count=0,
            first_observed=first,
            last_observed=last,
        )


@pytest.mark.parametrize("length", [0, -1, True, 2.0, "2"])
def test_list_metadata_requires_a_positive_integer_length(length: object) -> None:
    with pytest.raises(ValidationError, match="length"):
        ListVectorMetadataEntry.model_validate(
            {
                "id": "history",
                "base_id": "history",
                "kind": "list",
                "present_count": 2,
                "null_count": 0,
                "first_observed": _time(1),
                "last_observed": _time(2),
                "length": length,
                "observed_elements": 3,
            }
        )


def test_list_metadata_rejects_observed_elements_above_capacity() -> None:
    with pytest.raises(ValidationError, match="sequence capacity"):
        ListVectorMetadataEntry.model_validate(
            {
                "id": "history",
                "base_id": "history",
                "kind": "list",
                "present_count": 3,
                "null_count": 1,
                "length": 2,
                "observed_elements": 5,
            }
        )


@pytest.mark.parametrize(
    "domain",
    [
        [{"key": [None], "start": _time(1), "end": _time(1)}],
        [{"key": [float("nan")], "start": _time(1), "end": _time(1)}],
        [
            {"key": [True], "start": _time(1), "end": _time(1)},
            {"key": [1], "start": _time(2), "end": _time(2)},
        ],
    ],
)
def test_sample_metadata_rejects_unstable_identity_values(domain) -> None:
    with pytest.raises(ValidationError, match="Sample key field"):
        SampleMetadata.model_validate(
            {"cadence": "1h", "keys": ["security_id"], "domain": domain}
        )


def _metadata_payload() -> dict[str, object]:
    return {
        "schema_version": 4,
        "catalog": {
            "features": [],
            "targets": [],
            "counts": {"feature_vectors": 1, "target_vectors": 1},
        },
        "layout": {"kind": "unsplit"},
    }


@pytest.mark.parametrize("version", [1, 2, 3, 5, "4"])
def test_vector_metadata_rejects_unsupported_schema_versions(version: object) -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        VectorMetadata.model_validate(
            {**_metadata_payload(), "schema_version": version}
        )


def test_vector_metadata_requires_schema_version() -> None:
    payload = _metadata_payload()
    del payload["schema_version"]

    with pytest.raises(ValidationError, match="schema_version"):
        VectorMetadata.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generated_at", _time(1)),
        ("meta", {"producer": "legacy"}),
    ],
)
def test_vector_metadata_rejects_removed_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        VectorMetadata.model_validate({**_metadata_payload(), field: value})


def test_metadata_ids_share_one_feature_and_target_namespace() -> None:
    with pytest.raises(ValidationError, match="across features and targets"):
        VectorSchema.model_validate(
            {
                "features": [
                    {
                        "id": "price",
                        "base_id": "price",
                        "kind": "scalar",
                        "present_count": 1,
                        "null_count": 0,
                    }
                ],
                "targets": [
                    {
                        "id": "price",
                        "base_id": "return",
                        "kind": "scalar",
                        "present_count": 1,
                        "null_count": 0,
                    }
                ],
                "counts": {"feature_vectors": 1, "target_vectors": 1},
            }
        )


def _folded_metadata_payload() -> dict[str, object]:
    return {
        **_metadata_payload(),
        "layout": {
            "kind": "folded",
            "folds": [
                {
                    "id": "walk_0",
                    "training_schema": {
                        "features": [],
                        "targets": [],
                        "counts": {
                            "feature_vectors": 1,
                            "target_vectors": 1,
                        },
                    },
                    "outputs": [
                        {"role": "train", "labels": ["train_0"]},
                        {"role": "validation", "labels": ["validation_0"]},
                    ],
                }
            ],
        },
    }


def test_vector_metadata_selects_unsplit_layout_from_discriminator() -> None:
    metadata = VectorMetadata.model_validate(_metadata_payload())

    assert isinstance(metadata.layout, UnsplitMetadataLayout)
    assert metadata.catalog.counts.feature_vectors == 1


def test_vector_metadata_selects_folded_layout_from_discriminator() -> None:
    metadata = VectorMetadata.model_validate(_folded_metadata_payload())

    assert isinstance(metadata.layout, FoldedMetadataLayout)
    assert metadata.layout.folds[0].id == "walk_0"
    assert metadata.layout.folds[0].outputs[1].role == "validation"


@pytest.mark.parametrize("kind", [None, "split", "unknown"])
def test_vector_metadata_requires_a_supported_layout_discriminator(
    kind: object,
) -> None:
    payload = _metadata_payload()
    payload["layout"] = {} if kind is None else {"kind": kind}

    with pytest.raises(ValidationError, match="kind|discriminator"):
        VectorMetadata.model_validate(payload)


def test_folded_metadata_requires_unique_fold_ids() -> None:
    payload = _folded_metadata_payload()
    layout = payload["layout"]
    assert isinstance(layout, dict)
    folds = layout["folds"]
    assert isinstance(folds, list)
    folds.append(folds[0])

    with pytest.raises(ValidationError, match="fold ids must be unique"):
        VectorMetadata.model_validate(payload)


def test_fold_metadata_requires_unique_output_roles() -> None:
    payload = _folded_metadata_payload()
    layout = payload["layout"]
    assert isinstance(layout, dict)
    folds = layout["folds"]
    assert isinstance(folds, list)
    fold = folds[0]
    assert isinstance(fold, dict)
    fold["outputs"] = [
        {"role": "train", "labels": ["train_0"]},
        {"role": "train", "labels": ["train_1"]},
    ]

    with pytest.raises(ValidationError, match="output roles must be unique"):
        VectorMetadata.model_validate(payload)


def test_fold_metadata_requires_a_train_output() -> None:
    payload = _folded_metadata_payload()
    layout = payload["layout"]
    assert isinstance(layout, dict)
    folds = layout["folds"]
    assert isinstance(folds, list)
    fold = folds[0]
    assert isinstance(fold, dict)
    fold["outputs"] = [{"role": "validation", "labels": ["validation_0"]}]

    with pytest.raises(ValidationError, match="requires one train output"):
        VectorMetadata.model_validate(payload)


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["train_0", "train_0"],
        [""],
        [" train_0"],
    ],
)
def test_fold_output_requires_nonempty_unique_labels(labels: list[str]) -> None:
    payload = _folded_metadata_payload()
    layout = payload["layout"]
    assert isinstance(layout, dict)
    folds = layout["folds"]
    assert isinstance(folds, list)
    fold = folds[0]
    assert isinstance(fold, dict)
    fold["outputs"] = [{"role": "train", "labels": labels}]

    with pytest.raises(ValidationError, match="labels"):
        VectorMetadata.model_validate(payload)


def test_fold_labels_belong_to_only_one_output_role() -> None:
    payload = _folded_metadata_payload()
    layout = payload["layout"]
    assert isinstance(layout, dict)
    folds = layout["folds"]
    assert isinstance(folds, list)
    fold = folds[0]
    assert isinstance(fold, dict)
    fold["outputs"] = [
        {"role": "train", "labels": ["shared"]},
        {"role": "test", "labels": ["shared"]},
    ]

    with pytest.raises(ValidationError, match="only one output role"):
        VectorMetadata.model_validate(payload)

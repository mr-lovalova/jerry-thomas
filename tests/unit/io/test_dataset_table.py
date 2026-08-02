from datetime import datetime, timezone, tzinfo

import pytest

from jerrythomas.artifacts.models import (
    ListVectorMetadataEntry,
    ScalarVectorMetadataEntry,
)
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.io.dataset_table import DatasetTable


TIME = datetime(2024, 1, 2, tzinfo=timezone.utc)


class _InvalidTimezone(tzinfo):
    def utcoffset(self, dt):
        return None


def _scalar(
    id_: str,
    value_types: tuple[str, ...] = ("float",),
) -> ScalarVectorMetadataEntry:
    return ScalarVectorMetadataEntry(
        id=id_,
        base_id=id_,
        kind="scalar",
        present_count=2,
        null_count=0,
        value_types=value_types,
    )


def _sequence(
    id_: str,
    length: int = 2,
    element_types: tuple[str, ...] = ("float",),
) -> ListVectorMetadataEntry:
    return ListVectorMetadataEntry(
        id=id_,
        base_id=id_,
        kind="list",
        present_count=2,
        null_count=0,
        element_types=element_types,
        length=length,
        observed_elements=2 * length,
    )


def _table() -> DatasetTable:
    return DatasetTable(
        sample_keys=("ticker", "exchange"),
        sample_key_types=("string", "string"),
        feature_entries=(_scalar("price"), _sequence("history")),
        target_entries=(_scalar("forward_return"),),
    )


def test_dataset_table_has_stable_named_columns() -> None:
    table = _table()

    assert [column.name for column in table.columns] == [
        "sample.time",
        "sample.ticker",
        "sample.exchange",
        "features.price",
        "features.history.0",
        "features.history.1",
        "targets.forward_return",
    ]
    assert [column.value_type for column in table.columns] == [
        "datetime",
        "string",
        "string",
        "float",
        "float",
        "float",
        "float",
    ]


def test_dataset_table_projects_every_declared_column_in_order() -> None:
    row = _table().project(
        Sample(
            key=(TIME, "AAPL", "NASDAQ"),
            features=Vector(values={"price": 187.5, "history": [185.0, 186.0]}),
            targets=Vector(values={"forward_return": 0.03}),
        )
    )

    assert list(row) == [
        "sample.time",
        "sample.ticker",
        "sample.exchange",
        "features.price",
        "features.history.0",
        "features.history.1",
        "targets.forward_return",
    ]
    assert row == {
        "sample.time": TIME,
        "sample.ticker": "AAPL",
        "sample.exchange": "NASDAQ",
        "features.price": 187.5,
        "features.history.0": 185.0,
        "features.history.1": 186.0,
        "targets.forward_return": 0.03,
    }


def test_dataset_table_fills_missing_values_and_normalizes_nan() -> None:
    row = _table().project(
        Sample(
            key=(TIME, "AAPL", "NASDAQ"),
            features=Vector(values={"price": float("nan")}),
        )
    )

    assert row["features.price"] is None
    assert row["features.history.0"] is None
    assert row["features.history.1"] is None
    assert row["targets.forward_return"] is None


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"price": [1.0]}, "'price' must contain a scalar"),
        ({"history": 1.0}, "'history' must contain a list"),
        ({"history": [1.0]}, "'history' requires 2 values; got 1"),
    ],
)
def test_dataset_table_enforces_vector_shape(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _table().project(
            Sample(
                key=(TIME, "AAPL", "NASDAQ"),
                features=Vector(values=values),
            )
        )


@pytest.mark.parametrize(
    ("features", "targets", "message"),
    [
        ({"unknown": 1.0}, None, "unexpected feature ids"),
        ({}, {"unknown": 1.0}, "unexpected target ids"),
    ],
)
def test_dataset_table_rejects_unexpected_vector_ids(
    features: dict[str, object],
    targets: dict[str, object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _table().project(
            Sample(
                key=(TIME, "AAPL", "NASDAQ"),
                features=Vector(values=features),
                targets=None if targets is None else Vector(values=targets),
            )
        )


@pytest.mark.parametrize(
    ("key", "error", "message"),
    [
        ("not-a-tuple", TypeError, "key must be a tuple"),
        ((TIME, "AAPL"), ValueError, "has 2 values; expected 3"),
        (
            (datetime(2024, 1, 2), "AAPL", "NASDAQ"),
            ValueError,
            "time must be timezone-aware",
        ),
        (
            (datetime(2024, 1, 2, tzinfo=_InvalidTimezone()), "AAPL", "NASDAQ"),
            ValueError,
            "time must be timezone-aware",
        ),
        ((TIME, 7, "NASDAQ"), TypeError, "sample.ticker.*requires string"),
    ],
)
def test_dataset_table_enforces_sample_key_contract(
    key: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _table().project(Sample(key=key, features=Vector(values={})))


def test_dataset_table_rejects_ambiguous_or_unsupported_metadata_types() -> None:
    with pytest.raises(ValueError, match="one concrete value type"):
        DatasetTable((), (), (_scalar("mixed", ("float", "str")),), ())

    with pytest.raises(ValueError, match="does not support value type 'Decimal'"):
        DatasetTable((), (), (_scalar("decimal", ("Decimal",)),), ())


def test_dataset_table_rejects_colliding_flattened_columns() -> None:
    with pytest.raises(ValueError, match="features.history.0.*produced twice"):
        DatasetTable(
            (),
            (),
            (_sequence("history"), _scalar("history.0")),
            (),
        )


def test_dataset_table_null_columns_accept_only_null() -> None:
    table = DatasetTable((), (), (_scalar("empty", ()),), ())

    assert (
        table.project(Sample(key=(TIME,), features=Vector(values={})))["features.empty"]
        is None
    )
    with pytest.raises(TypeError, match="accepts only null"):
        table.project(Sample(key=(TIME,), features=Vector(values={"empty": 1.0})))


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_dataset_table_rejects_infinity(value: float) -> None:
    with pytest.raises(ValueError, match="must not contain infinity"):
        _table().project(
            Sample(
                key=(TIME, "AAPL", "NASDAQ"),
                features=Vector(values={"price": value}),
            )
        )


def test_dataset_table_uses_float_columns_for_scaled_series() -> None:
    table = DatasetTable(
        (),
        (),
        (_scalar("count", ("int",)),),
        (),
        scaled_feature_ids=("count",),
    )

    assert table.columns[1].value_type == "float"
    assert (
        table.project(Sample(key=(TIME,), features=Vector(values={"count": 1.5})))[
            "features.count"
        ]
        == 1.5
    )


def test_dataset_table_promotes_mixed_integer_and_float_metadata() -> None:
    table = DatasetTable(
        (),
        (),
        (_scalar("value", ("int", "float")),),
        (),
    )

    assert table.columns[1].value_type == "float"
    assert (
        table.project(Sample(key=(TIME,), features=Vector(values={"value": 1})))[
            "features.value"
        ]
        == 1.0
    )


def test_dataset_table_rejects_lossy_integer_to_float_promotion() -> None:
    table = DatasetTable(
        (),
        (),
        (_scalar("value", ("int", "float")),),
        (),
    )

    with pytest.raises(ValueError, match="cannot represent.*exactly as float64"):
        table.project(
            Sample(
                key=(TIME,),
                features=Vector(values={"value": 2**60 + 1}),
            )
        )


def test_dataset_table_rejects_integer_outside_signed_int64_range() -> None:
    table = DatasetTable((), (), (_scalar("count", ("int",)),), ())

    with pytest.raises(ValueError, match="requires a signed 64-bit integer"):
        table.project(Sample(key=(TIME,), features=Vector(values={"count": 2**63})))

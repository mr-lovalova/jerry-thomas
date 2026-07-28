from datapipeline.config.interpolation import (
    MissingInterpolation,
    is_missing_interpolation,
    normalize_interpolated_args,
)


def test_missing_interpolation_is_distinct_from_null_data() -> None:
    missing = MissingInterpolation("optional")

    assert is_missing_interpolation(missing)
    assert not is_missing_interpolation(None)
    assert normalize_interpolated_args(
        {"missing": missing, "null": None, "value": 1}
    ) == {
        "missing": None,
        "null": None,
        "value": 1,
    }

import pytest

from jerrythomas.domain.sample_key import SampleKeyContract, SampleKeyValueType


def test_empty_contract_accepts_empty_types() -> None:
    contract = SampleKeyContract(())

    contract.merge_types(())

    assert contract.inferred_types == ()
    assert contract.types == ()


def test_unknown_types_remain_unresolved_after_empty_stream() -> None:
    contract = SampleKeyContract(("exchange", "security_id"))

    contract.merge_types((None, None))

    assert contract.inferred_types == (None, None)
    with pytest.raises(
        ValueError,
        match="Sample key fields produced no values: exchange, security_id",
    ):
        _ = contract.types


def test_partial_types_merge_without_erasing_previous_inference() -> None:
    contract = SampleKeyContract(("exchange", "security_id"))
    initial_types = contract.inferred_types

    contract.merge_types(("string", None))

    assert initial_types == (None, None)
    assert contract.inferred_types == ("string", None)
    with pytest.raises(
        ValueError,
        match="Sample key fields produced no values: security_id",
    ):
        _ = contract.types

    contract.merge_types((None, "integer"))
    contract.merge_types((None, None))

    assert contract.types == ("string", "integer")


def test_merge_and_validate_share_inferred_types() -> None:
    contract = SampleKeyContract(("exchange", "security_id"))
    contract.validate(("NYSE", 42))

    contract.merge_types(("string", "integer"))
    contract.validate(("NASDAQ", 17))

    assert contract.types == ("string", "integer")


def test_merge_rejects_conflicting_types_like_validate() -> None:
    contract = SampleKeyContract(("security_id",), expected_types=("integer",))

    with pytest.raises(TypeError) as merged_error:
        contract.merge_types(("boolean",))
    with pytest.raises(TypeError) as validated_error:
        contract.validate((True,))

    assert str(merged_error.value) == str(validated_error.value)
    assert str(merged_error.value) == (
        "Sample key field 'security_id' changed type from integer to boolean."
    )


def test_validate_rejects_type_conflicting_with_merged_types() -> None:
    contract = SampleKeyContract(("security_id",))
    contract.merge_types(("integer",))

    with pytest.raises(TypeError, match="changed type from integer to string"):
        contract.validate(("42",))


@pytest.mark.parametrize("inferred_types", [(), ("string", "integer")])
def test_merge_rejects_wrong_type_count(
    inferred_types: tuple[SampleKeyValueType | None, ...],
) -> None:
    contract = SampleKeyContract(("security_id",))

    with pytest.raises(
        ValueError,
        match="Sample key type count must match the configured sample keys",
    ):
        contract.merge_types(inferred_types)

    assert contract.inferred_types == (None,)

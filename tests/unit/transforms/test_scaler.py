import json
from math import isclose
from statistics import mean, pstdev

import pytest
from pydantic import ValidationError

from jerrythomas.artifacts.scaler import (
    FoldedScalerArtifact,
    PositionalScalerStatistics,
    ScalerStatistics,
    StandardScalerArtifact,
    load_scaler_artifact,
    save_scaler_artifact,
)
from jerrythomas.transforms.vector.scaler import ScalerAccumulator


def _standard_artifact(
    mean: float = 2.0,
    std: float = 1.0,
    count: int = 2,
) -> StandardScalerArtifact:
    return StandardScalerArtifact(
        with_mean=True,
        with_std=True,
        epsilon=1e-12,
        observations=count,
        statistics={
            "x": ScalerStatistics(mean=mean, std=std, count=count),
        },
    )


def test_accumulator_fits_population_statistics() -> None:
    accumulator = ScalerAccumulator()

    accumulator.observe("x", 1.0)
    accumulator.observe("x", 2.0)
    accumulator.observe("x", 3.0)

    artifact = accumulator.artifact()
    statistics = artifact.statistics["x"]
    assert statistics.count == 3
    assert statistics.mean == 2.0
    assert isclose(statistics.std, 0.816496580927726)
    assert artifact.with_mean is True
    assert artifact.with_std is True
    assert artifact.epsilon == 1e-12
    assert artifact.observations == 3
    assert artifact.version == 4


def test_accumulator_uses_epsilon_for_constant_values() -> None:
    accumulator = ScalerAccumulator(epsilon=0.25)
    accumulator.observe("x", 4.0)
    accumulator.observe("x", 4.0)

    assert accumulator.artifact().statistics["x"].std == 0.25


@pytest.mark.parametrize(
    "values",
    [
        [1e16, 1e16 + 2.0],
        [1e12 + index * 0.001 for index in range(1000)],
        [1e16, -1e16, 1.0, 1.0],
        [1e16, -1e16, 1.0, 1e16, -1e16],
        [0.25, 0.125, 0.5],
    ],
)
def test_accumulator_preserves_spread_at_large_offsets(values: list[float]) -> None:
    accumulator = ScalerAccumulator()
    for value in values:
        accumulator.observe("x", value)

    statistics = accumulator.artifact().statistics["x"]

    assert statistics.mean == mean(values)
    assert statistics.std == pytest.approx(pstdev(values), rel=1e-12)


def test_accumulator_preserves_spread_in_list_positions_at_large_offsets() -> None:
    accumulator = ScalerAccumulator()
    accumulator.observe("x", [1e16, None])
    accumulator.observe("x", [1e16 + 2.0, 1e16])
    accumulator.observe("x", [None, 1e16 + 2.0])

    statistics = accumulator.artifact().statistics["x"]

    assert isinstance(statistics, PositionalScalerStatistics)
    assert [position.std for position in statistics.positions] == [1.0, 1.0]
    assert [position.count for position in statistics.positions] == [2, 2]


def test_accumulator_fits_large_constant_values_without_sum_overflow() -> None:
    accumulator = ScalerAccumulator()
    for _ in range(10):
        accumulator.observe("x", 1e308)

    assert accumulator.artifact().statistics["x"] == ScalerStatistics(
        mean=1e308, std=1e-12, count=10
    )


def test_accumulator_ignores_none() -> None:
    accumulator = ScalerAccumulator()
    accumulator.observe("x", None)
    accumulator.observe("x", 1.0)

    assert accumulator.observations == 1
    assert accumulator.artifact().statistics["x"].count == 1


def test_accumulator_fits_list_positions_independently() -> None:
    accumulator = ScalerAccumulator()

    accumulator.observe("x", [1.0, None])
    accumulator.observe("x", [3.0, 5.0])

    artifact = accumulator.artifact()
    statistics = artifact.statistics["x"]
    assert isinstance(statistics, PositionalScalerStatistics)
    first, second = statistics.positions
    assert first.count == 2
    assert first.mean == 2.0
    assert first.std == 1.0
    assert second.count == 1
    assert second.mean == 5.0
    assert second.std == 1e-12
    assert statistics.count == 3
    assert artifact.observations == 3


def test_accumulator_requires_an_observation_at_each_list_position() -> None:
    accumulator = ScalerAccumulator()
    accumulator.observe("x", [1.0, None])
    accumulator.observe("x", [2.0, None])

    with pytest.raises(RuntimeError, match=r"'x' positions: 1"):
        accumulator.artifact()


def test_accumulator_rejects_changing_list_width_without_mutation() -> None:
    accumulator = ScalerAccumulator()
    accumulator.observe("x", [1.0, 2.0])

    with pytest.raises(ValueError, match=r"'x'.*different lengths: 2 and 1"):
        accumulator.observe("x", [3.0])

    statistics = accumulator.artifact().statistics["x"]
    assert isinstance(statistics, PositionalScalerStatistics)
    assert statistics.positions == (
        ScalerStatistics(mean=1.0, std=1e-12, count=1),
        ScalerStatistics(mean=2.0, std=1e-12, count=1),
    )


def test_accumulator_rejects_scalar_after_list() -> None:
    accumulator = ScalerAccumulator()
    accumulator.observe("x", [1.0])

    with pytest.raises(ValueError, match=r"'x'.*mixes scalar and list"):
        accumulator.observe("x", 2.0)


def test_accumulator_rejects_list_after_scalar() -> None:
    accumulator = ScalerAccumulator()
    accumulator.observe("x", 1.0)

    with pytest.raises(ValueError, match=r"'x'.*mixes scalar and list"):
        accumulator.observe("x", [2.0])


def test_accumulator_requires_an_observation() -> None:
    with pytest.raises(RuntimeError, match="no numeric observations"):
        ScalerAccumulator().artifact()


@pytest.mark.parametrize(
    ("epsilon", "error"),
    [
        (True, TypeError),
        ("0.1", TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
    ],
)
def test_accumulator_rejects_invalid_epsilon(
    epsilon: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ScalerAccumulator(epsilon=epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, "1.0", [1.0, "2.0"], [[1.0]]])
def test_accumulator_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(TypeError, match="numeric or None"):
        ScalerAccumulator().observe("x", value)


def test_accumulator_does_not_partially_observe_an_invalid_list() -> None:
    accumulator = ScalerAccumulator()
    accumulator.observe("x", [1.0, 2.0])

    with pytest.raises(TypeError, match="numeric or None"):
        accumulator.observe("x", [3.0, "invalid"])

    artifact = accumulator.artifact()
    assert artifact.observations == 2
    assert artifact.statistics["x"] == PositionalScalerStatistics(
        positions=(
            ScalerStatistics(mean=1.0, std=1e-12, count=1),
            ScalerStatistics(mean=2.0, std=1e-12, count=1),
        )
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_accumulator_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ScalerAccumulator().observe("x", value)


def test_standard_scaler_rejects_inconsistent_observation_count() -> None:
    with pytest.raises(ValidationError, match="observations"):
        StandardScalerArtifact(
            with_mean=True,
            with_std=True,
            epsilon=1e-12,
            observations=2,
            statistics={
                "x": ScalerStatistics(mean=0.0, std=1.0, count=1),
            },
        )


def test_standard_scaler_counts_positional_observations() -> None:
    artifact = StandardScalerArtifact(
        with_mean=True,
        with_std=True,
        epsilon=1e-12,
        observations=3,
        statistics={
            "x": PositionalScalerStatistics(
                positions=(
                    ScalerStatistics(mean=1.0, std=1.0, count=2),
                    ScalerStatistics(mean=2.0, std=1.0, count=1),
                )
            )
        },
    )

    assert artifact.observations == 3


def test_folded_scaler_is_keyed_by_fold_id() -> None:
    train = _standard_artifact(mean=1.0, count=1)
    validation = _standard_artifact(mean=2.0, count=1)

    artifact = FoldedScalerArtifact(
        folds={
            "walk_0": train,
            "walk_1": validation,
        }
    )

    assert artifact.for_fold("walk_0") is train
    assert artifact.for_fold("walk_1") is validation


def test_folded_scaler_requires_a_known_fold_id() -> None:
    artifact = FoldedScalerArtifact(folds={"walk_0": _standard_artifact()})

    with pytest.raises(KeyError, match="has no fold 'walk_1'"):
        artifact.for_fold("walk_1")


@pytest.mark.parametrize("fold_id", ["", " ", " walk_0", "walk_0 "])
def test_folded_scaler_rejects_invalid_fold_ids(fold_id: str) -> None:
    with pytest.raises(ValidationError, match="fold ids"):
        FoldedScalerArtifact(folds={fold_id: _standard_artifact()})


@pytest.mark.parametrize(
    "artifact",
    [
        _standard_artifact(),
        StandardScalerArtifact(
            with_mean=True,
            with_std=True,
            epsilon=1e-12,
            observations=2,
            statistics={
                "x": PositionalScalerStatistics(
                    positions=(
                        ScalerStatistics(mean=1.0, std=1.0, count=1),
                        ScalerStatistics(mean=2.0, std=1.0, count=1),
                    )
                )
            },
        ),
        FoldedScalerArtifact(folds={"walk_0": _standard_artifact()}),
    ],
)
def test_scaler_artifact_round_trip(tmp_path, artifact) -> None:
    path = tmp_path / "scaler.json"

    save_scaler_artifact(path, artifact)

    assert load_scaler_artifact(path) == artifact


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "standard_scaler",
            "version": 2,
            "with_mean": True,
            "with_std": True,
            "epsilon": 1e-12,
            "observations": 1,
            "statistics": {"x": {"mean": 0.0, "std": 1.0, "count": 1}},
        },
        {
            "kind": "standard_scaler",
            "version": 3,
            "with_mean": True,
            "with_std": True,
            "epsilon": 1e-12,
            "observations": 1,
            "statistics": {"x": {"mean": 0.0, "std": 1.0, "count": 1}},
        },
        {
            "kind": "standard_scaler",
            "version": 4,
            "with_mean": True,
            "with_std": True,
            "epsilon": 1e-12,
            "observations": 1,
            "statistics": {"x": {"mean": 0.0, "count": 1}},
        },
        {
            "kind": "standard_scaler",
            "version": 4,
            "with_mean": "false",
            "with_std": True,
            "epsilon": 1e-12,
            "observations": 1,
            "statistics": {"x": {"mean": 0.0, "std": 1.0, "count": 1}},
        },
    ],
)
def test_scaler_artifact_rejects_old_incomplete_or_coerced_payloads(
    tmp_path,
    payload,
) -> None:
    path = tmp_path / "scaler.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_scaler_artifact(path)


def test_accumulator_extends_disjoint_vectors_without_sharing_mutable_moments() -> None:
    combined = ScalerAccumulator()
    combined.observe("existing", 7.0)
    source = ScalerAccumulator()
    source.observe("scalar", 1e16)
    source.observe("scalar", 1e16 + 2.0)
    source.observe("positions", [None, 0.125])
    source.observe("positions", [3.0, 0.25])
    expected = source.artifact()

    combined.extend(source)
    combined.extend(ScalerAccumulator())
    source.observe("scalar", -1e16)
    source.observe("positions", [100.0, 100.0])

    result = combined.artifact()
    assert result.observations == 1 + expected.observations
    assert result.statistics["scalar"] == expected.statistics["scalar"]
    assert result.statistics["positions"] == expected.statistics["positions"]
    assert result.statistics["existing"].mean == 7.0


@pytest.mark.parametrize(
    "settings",
    [{"with_mean": False}, {"with_std": False}, {"epsilon": 0.1}],
)
def test_accumulator_rejects_different_settings_before_extending(settings) -> None:
    combined = ScalerAccumulator()
    combined.observe("existing", 1.0)
    before = combined.artifact()
    source = ScalerAccumulator(**settings)
    source.observe("new", 2.0)

    with pytest.raises(ValueError, match="different settings"):
        combined.extend(source)

    assert combined.artifact() == before


def test_accumulator_rejects_overlapping_ids_before_extending() -> None:
    combined = ScalerAccumulator()
    combined.observe("existing", 1.0)
    before = combined.artifact()
    source = ScalerAccumulator()
    source.observe("new", 2.0)
    source.observe("existing", 3.0)

    with pytest.raises(ValueError, match="overlapping vector IDs: existing"):
        combined.extend(source)

    assert combined.artifact() == before

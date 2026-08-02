import math
from collections.abc import Collection, Iterator
from dataclasses import dataclass, replace
from numbers import Real

from jerrythomas.artifacts.scaler import (
    PositionalScalerStatistics,
    ScalerStatistics,
    StandardScalerArtifact,
)
from jerrythomas.domain.series_id import base_id
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector


@dataclass(slots=True)
class _RunningStatistics:
    count: int = 0
    mean: float = 0.0
    squared_deviations: float = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.squared_deviations += delta * (value - self.mean)

    def finish(self, epsilon: float) -> ScalerStatistics:
        variance = self.squared_deviations / self.count
        return ScalerStatistics(
            mean=self.mean,
            std=max(math.sqrt(max(variance, 0.0)), epsilon),
            count=self.count,
        )


class ScalerAccumulator:
    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        epsilon: float = 1e-12,
    ) -> None:
        if not isinstance(with_mean, bool):
            raise TypeError("with_mean must be a boolean")
        if not isinstance(with_std, bool):
            raise TypeError("with_std must be a boolean")
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise TypeError("epsilon must be numeric")
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

        self.with_mean: bool = with_mean
        self.with_std: bool = with_std
        self.epsilon: float = epsilon
        self._statistics: dict[
            str,
            _RunningStatistics | tuple[_RunningStatistics, ...],
        ] = {}
        self.observations: int = 0

    def observe(self, vector_id: str, value: object) -> None:
        if not vector_id.strip():
            raise ValueError("vector_id must not be empty")
        if value is None:
            return
        if isinstance(value, list):
            self._observe_positions(vector_id, value)
        else:
            self._observe_scalar(vector_id, value)

    def _observe_scalar(self, vector_id: str, value: object) -> None:
        current = self._statistics.get(vector_id)
        if isinstance(current, tuple):
            raise ValueError(f"Vector {vector_id!r} mixes scalar and list values.")
        number = _finite_number(value)
        if current is None:
            current = _RunningStatistics()
            self._statistics[vector_id] = current
        current.observe(number)
        self.observations += 1

    def _observe_positions(self, vector_id: str, values: list[object]) -> None:
        current = self._statistics.get(vector_id)
        if isinstance(current, _RunningStatistics):
            raise ValueError(f"Vector {vector_id!r} mixes scalar and list values.")
        if not values:
            raise ValueError(f"Vector {vector_id!r} cannot fit an empty list value.")
        if current is not None and len(current) != len(values):
            raise ValueError(
                f"Vector {vector_id!r} contains list values with different "
                f"lengths: {len(current)} and {len(values)}."
            )
        numbers = tuple(
            None if value is None else _finite_number(value) for value in values
        )
        if current is None:
            current = tuple(_RunningStatistics() for _ in values)
        for statistics, number in zip(current, numbers, strict=True):
            if number is not None:
                statistics.observe(number)
        self._statistics[vector_id] = current
        self.observations += sum(number is not None for number in numbers)

    def artifact(self) -> StandardScalerArtifact:
        if not self._statistics:
            raise RuntimeError("Scaler fitting produced no numeric observations.")
        statistics: dict[
            str,
            ScalerStatistics | PositionalScalerStatistics,
        ] = {}
        for vector_id, running in self._statistics.items():
            if isinstance(running, tuple):
                missing = [
                    str(position)
                    for position, entry in enumerate(running)
                    if entry.count == 0
                ]
                if missing:
                    raise RuntimeError(
                        f"Scaler fitting produced no numeric observations for vector "
                        f"{vector_id!r} positions: {', '.join(missing)}."
                    )
                statistics[vector_id] = PositionalScalerStatistics(
                    positions=tuple(entry.finish(self.epsilon) for entry in running),
                )
            else:
                statistics[vector_id] = running.finish(self.epsilon)
        return StandardScalerArtifact(
            with_mean=self.with_mean,
            with_std=self.with_std,
            epsilon=self.epsilon,
            observations=self.observations,
            statistics=statistics,
        )


class SampleScaler:
    def __init__(
        self,
        artifact: StandardScalerArtifact,
        scaled_feature_ids: Collection[str],
        scaled_target_ids: Collection[str],
    ) -> None:
        self.artifact = artifact
        self.scaled_feature_ids = frozenset(scaled_feature_ids)
        self.scaled_target_ids = frozenset(scaled_target_ids)

    def apply(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        for sample in samples:
            yield self.scale(sample)

    def scale(self, sample: Sample) -> Sample:
        features = _scale_vector(
            sample.features,
            self.scaled_feature_ids,
            self.artifact,
        )
        targets = sample.targets
        if targets is not None:
            targets = _scale_vector(
                targets,
                self.scaled_target_ids,
                self.artifact,
            )
        return replace(sample, features=features, targets=targets)


def _scale_vector(
    vector: Vector,
    scaled_ids: frozenset[str],
    artifact: StandardScalerArtifact,
) -> Vector:
    if not scaled_ids:
        return vector
    values = {
        vector_id: (
            _scale_value(vector_id, value, artifact)
            if base_id(vector_id) in scaled_ids
            else value
        )
        for vector_id, value in vector.values.items()
    }
    return replace(vector, values=values)


def _scale_value(
    vector_id: str,
    value: object,
    artifact: StandardScalerArtifact,
) -> object:
    if value is None:
        return None
    statistics = _statistics_for(vector_id, artifact)
    if isinstance(statistics, PositionalScalerStatistics):
        if not isinstance(value, list):
            raise TypeError(
                f"Vector {vector_id!r} requires a list value for positional scaling."
            )
        if len(value) != len(statistics.positions):
            raise ValueError(
                f"Vector {vector_id!r} requires {len(statistics.positions)} values "
                f"for positional scaling, got {len(value)}."
            )
        return [
            _scale_scalar(item, position, artifact)
            for item, position in zip(value, statistics.positions, strict=True)
        ]
    if isinstance(value, list):
        return [_scale_scalar(item, statistics, artifact) for item in value]
    return _scale_scalar(value, statistics, artifact)


def _statistics_for(
    vector_id: str,
    artifact: StandardScalerArtifact,
) -> ScalerStatistics | PositionalScalerStatistics:
    try:
        return artifact.statistics[vector_id]
    except KeyError as exc:
        raise KeyError(f"Missing scaler statistics for vector {vector_id!r}.") from exc


def _scale_scalar(
    value: object,
    statistics: ScalerStatistics,
    artifact: StandardScalerArtifact,
) -> float | None:
    if value is None:
        return None
    number = _finite_number(value)
    if artifact.with_mean:
        number -= statistics.mean
    if artifact.with_std:
        number /= statistics.std
    return number


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Vector value must be numeric or None, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Vector value must be finite, got {value!r}.")
    return number

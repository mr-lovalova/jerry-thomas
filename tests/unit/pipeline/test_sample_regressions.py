import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jerrythomas.artifacts.registry import SCALER_SPEC
from jerrythomas.artifacts.scaler import save_scaler_artifact
from jerrythomas.artifacts.specs import SCALER_STATISTICS
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig, TargetSeriesConfig
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.transforms import (
    FillConfig,
    RollingConfig,
    TransformConfig,
)
from jerrythomas.domain.series import SeriesRecord
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.pipelines.series.pipeline import run_series_pipeline
from jerrythomas.pipelines.sample.input import open_samples
from jerrythomas.runtime import Runtime, SourceRuntimeStream
from jerrythomas.transforms.vector.scaler import SampleScaler, ScalerAccumulator
from tests.series_helpers import register_series


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 1, hour=hour, minute=minute, tzinfo=timezone.utc)


def _record(ts: datetime, value: float | None) -> TemporalRecord:
    rec = TemporalRecord(time=ts)
    setattr(rec, "value", value)
    return rec


def _air_density(pressure_hpa: float, temp_c: float, rh_percent: float | None) -> float:
    pressure_pa = pressure_hpa * 100.0
    temp_k = temp_c + 273.15
    density = pressure_pa / (287.05 * temp_k)
    if rh_percent is not None:
        rh = rh_percent / 100.0
        saturation = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
        vapor_pressure = rh * saturation * 100.0
        density = (pressure_pa - 0.378 * vapor_pressure) / (287.05 * temp_k)
    return density


def _identity(iterable):
    return iterable


class _StubSource:
    def __init__(self, rows):
        self._rows = rows

    def stream(self):
        return iter(self._rows)


def _runtime_with_streams(
    tmp_path: Path,
    streams: dict[str, list[TemporalRecord]],
    stream_transforms: dict[str, list[TransformConfig]] | None = None,
) -> Runtime:
    project_yaml = tmp_path / "project.yaml"
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(
        project_yaml=project_yaml,
        artifacts_root=artifacts_root,
        dataset=DatasetConfig(sample=SampleConfig(rounding="ceil", cadence="1h")),
        execution=ExecutionConfig(),
    )

    stream_transforms = stream_transforms or {}
    for alias, rows in streams.items():
        runtime.streams[alias] = SourceRuntimeStream(
            source=_StubSource(rows),
            mapper=_identity,
            preprocess=(),
            partition_by=(),
            presorted=False,
            transforms=tuple(stream_transforms.get(alias, ())),
        )

    return runtime


def _register_scaler(runtime: Runtime, configs: list[SeriesConfig]) -> None:
    sanitized = [
        cfg.model_copy(update={"scale": False, "sequence": None, "collect": None})
        for cfg in configs
    ]
    accumulator = ScalerAccumulator()
    for cfg in sanitized:
        stream = run_series_pipeline(runtime, cfg)
        try:
            for feature in stream:
                if not isinstance(feature, SeriesRecord):
                    raise TypeError("Scaler tests require scalar feature records")
                accumulator.observe(feature.id, feature.value)
        finally:
            closer = getattr(stream, "close", None)
            if callable(closer):
                closer()
    if not accumulator.observations:
        raise RuntimeError("Unable to compute scaler statistics for test runtime.")

    destination = runtime.artifacts_root / "scaler.json"
    save_scaler_artifact(destination, accumulator.artifact())
    runtime.artifacts.register(
        SCALER_STATISTICS,
        relative_path=destination.relative_to(runtime.artifacts_root).as_posix(),
    )


def test_sample_targets_respect_partitioned_ids(tmp_path) -> None:
    def _partitioned_record(value: float, code: str) -> TemporalRecord:
        rec = _record(_ts(0), value)
        setattr(rec, "municipality", code)
        return rec

    streams = {
        "wind_speed_stream": [
            _partitioned_record(2.5, "06019"),
            _partitioned_record(3.1, "06030"),
        ],
        "wind_production_stream": [
            _partitioned_record(0.3, "06019"),
            _partitioned_record(0.5, "06030"),
        ],
    }
    runtime = _runtime_with_streams(tmp_path, streams)
    runtime.streams["wind_speed_stream"] = replace(
        runtime.streams["wind_speed_stream"],
        partition_by=("municipality",),
    )
    runtime.streams["wind_production_stream"] = replace(
        runtime.streams["wind_production_stream"],
        partition_by=("municipality",),
    )

    feature_cfgs = [
        SeriesConfig(
            stream="wind_speed_stream",
            id="wind_speed",
            field="value",
        ),
    ]
    target_cfgs = [
        TargetSeriesConfig(
            stream="wind_production_stream",
            id="wind_production",
            field="value",
            horizon="0s",
        ),
    ]
    register_series(runtime, feature_cfgs, "1h", targets=target_cfgs)

    samples = list(
        open_samples(
            runtime,
            {
                "wind_speed__@municipality:06019",
                "wind_speed__@municipality:06030",
            },
            target_ids={
                "wind_production__@municipality:06019",
                "wind_production__@municipality:06030",
            },
        )
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.targets is not None
    assert set(sample.targets.keys()) == {
        "wind_production__@municipality:06019",
        "wind_production__@municipality:06030",
    }
    assert all(not key.startswith("wind_production") for key in sample.features.keys())


def test_samples_can_group_by_record_key_fields(tmp_path) -> None:
    def _equity_record(hour: int, security_id: str, value: float) -> TemporalRecord:
        rec = _record(_ts(hour), value)
        setattr(rec, "security_id", security_id)
        return rec

    streams = {
        "momentum_stream": [
            _equity_record(0, "MSFT", 2.0),
            _equity_record(0, "AAPL", 1.0),
            _equity_record(1, "AAPL", 3.0),
        ],
        "return_stream": [
            _equity_record(0, "AAPL", 0.1),
            _equity_record(0, "MSFT", 0.2),
            _equity_record(1, "AAPL", 0.3),
        ],
    }
    runtime = _runtime_with_streams(tmp_path, streams)
    feature_cfgs = [
        SeriesConfig(
            stream="momentum_stream",
            id="momentum",
            field="value",
        ),
    ]
    target_cfgs = [
        TargetSeriesConfig(
            stream="return_stream",
            id="forward_return",
            field="value",
            horizon="0s",
        ),
    ]
    register_series(
        runtime,
        feature_cfgs,
        "1h",
        targets=target_cfgs,
        sample_keys=["security_id"],
    )

    samples = list(
        open_samples(
            runtime,
            [config.id for config in feature_cfgs],
            target_ids=[config.id for config in target_cfgs],
        )
    )

    assert [sample.key for sample in samples] == [
        (_ts(0), "AAPL"),
        (_ts(0), "MSFT"),
        (_ts(1), "AAPL"),
    ]
    assert [sample.features.values["momentum"] for sample in samples] == [
        1.0,
        2.0,
        3.0,
    ]
    assert [sample.targets.values["forward_return"] for sample in samples] == [
        0.1,
        0.2,
        0.3,
    ]


def test_samples_keep_entity_buckets_contiguous(tmp_path) -> None:
    def _equity_record(hour: int, security_id: str, value: float) -> TemporalRecord:
        rec = _record(_ts(hour), value)
        setattr(rec, "security_id", security_id)
        return rec

    streams = {
        "momentum_stream": [
            _equity_record(1, "AAPL", 1.0),
            _equity_record(1, "MSFT", 10.0),
            _equity_record(2, "AAPL", 2.0),
            _equity_record(2, "MSFT", 20.0),
        ],
        "volume_stream": [
            _equity_record(1, "AAPL", 100.0),
            _equity_record(1, "MSFT", 1000.0),
            _equity_record(2, "AAPL", 200.0),
            _equity_record(2, "MSFT", 2000.0),
        ],
    }
    runtime = _runtime_with_streams(tmp_path, streams)
    for stream_id in streams:
        runtime.streams[stream_id] = replace(
            runtime.streams[stream_id],
            partition_by=("security_id",),
        )
    feature_cfgs = [
        SeriesConfig(
            stream="momentum_stream",
            id="momentum",
            field="value",
            collect=2,
        ),
        SeriesConfig(
            stream="volume_stream",
            id="volume",
            field="value",
            collect=2,
        ),
    ]
    register_series(
        runtime,
        feature_cfgs,
        "1d",
        sample_keys=["security_id"],
    )

    samples = list(
        open_samples(
            runtime,
            [config.id for config in feature_cfgs],
        )
    )

    assert [sample.key for sample in samples] == [
        (_ts(0) + timedelta(days=1), "AAPL"),
        (_ts(0) + timedelta(days=1), "MSFT"),
    ]
    assert [sample.features.values for sample in samples] == [
        {"momentum": [1.0, 2.0], "volume": [100.0, 200.0]},
        {"momentum": [10.0, 20.0], "volume": [1000.0, 2000.0]},
    ]


def test_sequence_series_are_windowed_by_sample_keys(tmp_path) -> None:
    def _equity_record(hour: int, security_id: str, value: float) -> TemporalRecord:
        rec = _record(_ts(hour), value)
        setattr(rec, "security_id", security_id)
        return rec

    streams = {
        "monthly_returns": [
            _equity_record(0, "AAPL", 1.0),
            _equity_record(0, "MSFT", 10.0),
            _equity_record(1, "AAPL", 2.0),
            _equity_record(1, "MSFT", 20.0),
        ],
    }
    runtime = _runtime_with_streams(tmp_path, streams)
    runtime.streams["monthly_returns"] = replace(
        runtime.streams["monthly_returns"],
        partition_by=("security_id",),
    )
    feature_cfgs = [
        SeriesConfig(
            stream="monthly_returns",
            id="monthly_return",
            field="value",
            sequence={"size": 2, "stride": 1},
        ),
    ]
    register_series(
        runtime,
        feature_cfgs,
        "1h",
        sample_keys=["security_id"],
    )

    samples = list(
        open_samples(
            runtime,
            [config.id for config in feature_cfgs],
        )
    )

    assert [sample.key for sample in samples] == [
        (_ts(1), "AAPL"),
        (_ts(1), "MSFT"),
    ]
    assert [sample.features.values for sample in samples] == [
        {"monthly_return": [1.0, 2.0]},
        {"monthly_return": [10.0, 20.0]},
    ]


def test_sample_input_selects_exact_wide_feature_ids(
    tmp_path,
) -> None:
    def _equity_record(hour: int, security_id: str, value: float) -> TemporalRecord:
        rec = _record(_ts(hour), value)
        setattr(rec, "security_id", security_id)
        return rec

    streams = {
        "monthly_returns": [
            _equity_record(0, "AAPL", 1.0),
            _equity_record(0, "MSFT", 10.0),
            _equity_record(1, "AAPL", 2.0),
            _equity_record(1, "MSFT", 20.0),
        ],
    }
    runtime = _runtime_with_streams(tmp_path, streams)
    runtime.streams["monthly_returns"] = replace(
        runtime.streams["monthly_returns"],
        partition_by=("security_id",),
    )
    feature_cfgs = [
        SeriesConfig(
            stream="monthly_returns",
            id="monthly_return",
            field="value",
            sequence={"size": 2, "stride": 1},
        ),
    ]
    register_series(
        runtime,
        feature_cfgs,
        "1h",
    )

    samples = list(
        open_samples(
            runtime,
            ["monthly_return__@security_id:AAPL"],
        )
    )

    assert samples[0].key == (_ts(1),)
    assert samples[0].features.values == {
        "monthly_return__@security_id:AAPL": [1.0, 2.0],
    }


def test_stream_transforms_use_explicit_stream_partition(tmp_path) -> None:
    def _equity_record(hour: int, security_id: str, value: float) -> TemporalRecord:
        rec = _record(_ts(hour), value)
        setattr(rec, "security_id", security_id)
        return rec

    streams = {
        "daily_prices": [
            _equity_record(0, "AAPL", 10.0),
            _equity_record(0, "MSFT", 100.0),
            _equity_record(1, "AAPL", 20.0),
            _equity_record(1, "MSFT", 200.0),
        ],
    }
    runtime = _runtime_with_streams(
        tmp_path,
        streams,
        stream_transforms={
            "daily_prices": [
                RollingConfig(
                    field="value",
                    to="value_mean_2",
                    window=2,
                    min_samples=2,
                )
            ]
        },
    )
    runtime.streams["daily_prices"] = replace(
        runtime.streams["daily_prices"],
        partition_by=("security_id",),
    )
    feature_cfgs = [
        SeriesConfig(
            stream="daily_prices",
            id="value_mean_2",
            field="value_mean_2",
        ),
    ]
    register_series(
        runtime,
        feature_cfgs,
        "1h",
        sample_keys=["security_id"],
    )

    samples = list(
        open_samples(
            runtime,
            [config.id for config in feature_cfgs],
        )
    )

    assert [sample.key for sample in samples] == [
        (_ts(0), "AAPL"),
        (_ts(0), "MSFT"),
        (_ts(1), "AAPL"),
        (_ts(1), "MSFT"),
    ]
    assert [sample.features.values for sample in samples] == [
        {"value_mean_2": None},
        {"value_mean_2": None},
        {"value_mean_2": 15.0},
        {"value_mean_2": 150.0},
    ]


def test_regression_scaled_shapes_airpressure_high_freq_and_windspeed_hourly(
    tmp_path,
) -> None:
    # Fake raw streams
    # high-frequency mock series (multiple samples per hour)
    high_freq_raw = [
        _record(_ts(0, 10), 1010.0),
        _record(_ts(0, 20), 1020.0),
        _record(_ts(0, 55), 1000.0),
        _record(_ts(1, 5), 1005.0),
        _record(_ts(1, 15), 1007.0),
        _record(_ts(1, 45), 1009.0),
    ]
    # hourly mock series (single sample per hour)
    hourly_raw = [
        _record(_ts(0, 0), 5.0),
        _record(_ts(1, 0), 7.0),
    ]

    streams = {
        "air_pressure": high_freq_raw,
        "wind_speed": hourly_raw,
    }
    runtime = _runtime_with_streams(tmp_path, streams)
    group_by = "1h"

    configs = [
        SeriesConfig(
            stream="air_pressure",
            id="air_pressure",
            field="value",
            scale=True,
            collect=3,
        ),
        SeriesConfig(
            stream="wind_speed",
            id="wind_speed",
            field="value",
            scale=True,
        ),
    ]

    _register_scaler(runtime, configs)
    register_series(runtime, configs, group_by)

    raw = open_samples(
        runtime,
        [config.id for config in configs],
    )
    out = list(
        SampleScaler(
            runtime.artifacts.load(SCALER_SPEC),
            scaled_feature_ids=[config.id for config in configs],
            scaled_target_ids=(),
        ).apply(raw)
    )

    # Off-grid pressure values become available at the following hourly tick.
    assert [sample.key for sample in out] == [(_ts(0),), (_ts(1),), (_ts(2),)]

    # Shapes: air_pressure -> lists (multiple per hour); wind_speed -> scalars (single per hour)
    wind_only = out[0].features.values
    v0 = out[1].features.values
    v1 = out[2].features.values

    assert isinstance(v0["air_pressure"], list) and len(v0["air_pressure"]) == 3
    assert isinstance(v1["air_pressure"], list) and len(v1["air_pressure"]) == 3
    assert "air_pressure" not in wind_only
    assert "wind_speed" not in v1
    assert not isinstance(wind_only["wind_speed"], list)
    assert not isinstance(v0["wind_speed"], list)

    # Scaled values should have ~zero mean per feature across the whole stream
    ap_all = v0["air_pressure"] + v1["air_pressure"]
    ap_mean = sum(ap_all) / len(ap_all)
    assert abs(ap_mean) < 1e-6

    ws_all = [wind_only["wind_speed"], v0["wind_speed"]]
    ws_mean = sum(ws_all) / len(ws_all)
    assert abs(ws_mean) < 1e-6


def test_regression_fill_then_scale_with_missing_values(tmp_path) -> None:
    # air_pressure has a missing value in between two valid ones within the same hour
    series_high = [
        _record(_ts(0, 10), 1000.0),
        _record(_ts(0, 20), None),  # missing
        _record(_ts(0, 40), 1100.0),
    ]
    # wind_speed hourly with a missing at hour 1
    series_hourly = [
        _record(_ts(0, 0), 5.0),
        _record(_ts(1, 0), None),  # missing
    ]

    streams = {"ap": series_high, "ws": series_hourly}
    runtime = _runtime_with_streams(tmp_path, streams)
    group_by = "1h"

    # Fill then scale for both features
    stream_transforms = {
        "ap": [
            FillConfig(
                field="value",
                statistic="median",
                window=10,
                min_samples=1,
            ),
        ],
        "ws": [
            FillConfig(
                field="value",
                statistic="mean",
                window=10,
                min_samples=1,
            ),
        ],
    }

    runtime = _runtime_with_streams(
        tmp_path, streams, stream_transforms=stream_transforms
    )

    configs = [
        SeriesConfig(
            stream="ap",
            id="air_pressure",
            field="value",
            scale=True,
            collect=3,
        ),
        SeriesConfig(
            stream="ws",
            id="wind_speed",
            field="value",
            scale=True,
        ),
    ]

    _register_scaler(runtime, configs)
    register_series(runtime, configs, group_by)
    raw = open_samples(
        runtime,
        [config.id for config in configs],
    )
    out = list(
        SampleScaler(
            runtime.artifacts.load(SCALER_SPEC),
            scaled_feature_ids=[config.id for config in configs],
            scaled_target_ids=(),
        ).apply(raw)
    )

    assert [sample.key for sample in out] == [(_ts(0),), (_ts(1),)]

    v0 = out[0].features.values
    v1 = out[1].features.values
    assert "air_pressure" not in v0
    # air_pressure list length = 3 with middle filled (not None)
    assert isinstance(v1["air_pressure"], list) and len(v1["air_pressure"]) == 3
    assert all(isinstance(x, float) for x in v1["air_pressure"])  # filled and scaled

    # wind_speed hour 0 present and scaled; hour 1 present due to fill then scale
    v1 = out[1].features.values
    assert not isinstance(v0["wind_speed"], list)
    assert not isinstance(v1["wind_speed"], list)

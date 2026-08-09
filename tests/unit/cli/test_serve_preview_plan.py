from types import SimpleNamespace

from jerrythomas.pipelines.dataset.preview import preview_output_plan


def _cfg(id_: str, stream: str):
    return SimpleNamespace(id=id_, stream=stream)


def test_record_preview_plan_dedupes_shared_streams() -> None:
    preview_cfgs = [
        _cfg("closing_price", "equity.ohlcv"),
        _cfg("opening_price", "equity.ohlcv"),
        _cfg("linear_time", "time.ticks.linear"),
    ]

    plan = preview_output_plan(
        preview_cfgs,
        "input",
    )

    assert plan == (
        ("equity.ohlcv", preview_cfgs[0]),
        ("time.ticks.linear", preview_cfgs[2]),
    )


def test_series_preview_plan_keeps_each_series() -> None:
    preview_cfgs = [
        _cfg("closing_price", "equity.ohlcv"),
        _cfg("opening_price", "equity.ohlcv"),
    ]

    plan = preview_output_plan(
        preview_cfgs,
        "series",
    )

    assert plan == (
        ("closing_price", preview_cfgs[0]),
        ("opening_price", preview_cfgs[1]),
    )

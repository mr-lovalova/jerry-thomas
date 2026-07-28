from pathlib import Path

import pytest

from datapipeline.config.profiles.output import ServeOutputConfig
from datapipeline.io.output import resolve_output_target
from datapipeline.io.runs import get_run_paths


def test_resolve_output_target_uses_directory_and_profile_name(tmp_path):
    default_dir = tmp_path / "outputs"
    default_dir.mkdir()
    cfg = ServeOutputConfig(transport="fs", format="jsonl", directory=default_dir)

    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="train",
    )

    assert target.transport == "fs"
    assert target.format == "jsonl"
    assert target.view == "raw"
    assert target.encoding == "utf-8"
    assert target.destination == (default_dir / "train" / "train.jsonl").resolve()


def test_resolve_output_target_honors_custom_filename(tmp_path):
    base_dir = tmp_path / "outputs"
    base_dir.mkdir()
    cfg = ServeOutputConfig(
        transport="fs", format="jsonl", directory=base_dir, filename="custom"
    )

    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="train",
    )

    assert target.destination == (base_dir / "train" / "custom.jsonl").resolve()
    assert target.encoding == "utf-8"


def test_resolve_output_target_defaults_to_dataset_filename(tmp_path):
    config = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=tmp_path,
    )

    target = resolve_output_target(
        cli_output=None,
        config_output=config,
    )

    assert target.destination == (tmp_path / "dataset.jsonl").resolve()


@pytest.mark.parametrize("format_", ["jsonl", "csv"])
def test_resolve_output_target_adds_compressed_suffix(tmp_path, format_) -> None:
    config = ServeOutputConfig(
        transport="fs",
        format=format_,
        directory=tmp_path,
        filename="dataset",
        compression="gzip",
    )

    target = resolve_output_target(
        cli_output=None,
        config_output=config,
        profile_name="processed",
    )

    assert target.compression == "gzip"
    assert (
        target.destination
        == (tmp_path / "processed" / f"dataset.{format_}.gz").resolve()
    )


def test_resolve_output_target_allows_dotted_filename_stem(tmp_path):
    base_dir = tmp_path / "outputs"
    base_dir.mkdir()
    cfg = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=base_dir,
        filename="equity.price_aware_universe",
    )

    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="processed",
    )

    assert (
        target.destination
        == (base_dir / "processed" / "equity.price_aware_universe.jsonl").resolve()
    )


def test_routed_output_target_preserves_dotted_filename_stem(tmp_path):
    base_dir = tmp_path / "outputs"
    base_dir.mkdir()
    cfg = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=base_dir,
        filename="equity.price_aware_universe",
    )
    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="processed",
    )

    routed_target = target.for_output("holdout.train")

    assert (
        routed_target.destination
        == (
            base_dir / "processed" / "equity.price_aware_universe.holdout.train.jsonl"
        ).resolve()
    )


def test_compressed_routed_output_inserts_id_before_complete_suffix(tmp_path) -> None:
    config = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=tmp_path,
        filename="dataset.v1",
        compression="gzip",
    )
    target = resolve_output_target(
        cli_output=None,
        config_output=config,
        profile_name="processed",
    )

    routed = target.for_output("fold_0.train")

    assert routed.compression == "gzip"
    assert (
        routed.destination
        == (tmp_path / "processed" / "dataset.v1.fold_0.train.jsonl.gz").resolve()
    )


def test_preview_output_preserves_dotted_filename_stem(tmp_path):
    base_dir = tmp_path / "outputs"
    base_dir.mkdir()
    cfg = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=base_dir,
        filename="equity.price_aware_universe",
    )
    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="preview",
    )

    preview_target = target.for_output("equity.adv.daily.20")

    assert (
        preview_target.destination
        == (
            base_dir
            / "preview"
            / "equity.price_aware_universe.equity.adv.daily.20.jsonl"
        ).resolve()
    )


def test_resolve_output_target_rejects_filename_with_selected_extension(tmp_path):
    base_dir = tmp_path / "outputs"
    base_dir.mkdir()
    cfg = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=base_dir,
        filename="equity.price_aware_universe.jsonl",
    )

    with pytest.raises(ValueError, match="filename must omit the '.jsonl' extension"):
        resolve_output_target(
            cli_output=None,
            config_output=cfg,
            default=None,
            base_path=Path("."),
            profile_name="processed",
        )


@pytest.mark.parametrize(
    ("compression", "filename"),
    [
        ("gzip", "dataset.jsonl"),
        ("gzip", "dataset.jsonl.gz"),
        (None, "dataset.jsonl.gz"),
    ],
)
def test_output_rejects_filename_extensions(
    tmp_path,
    compression,
    filename,
) -> None:
    config = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=tmp_path,
        filename=filename,
        compression=compression,
    )

    with pytest.raises(ValueError, match="filename must omit"):
        resolve_output_target(
            cli_output=None,
            config_output=config,
            profile_name="processed",
        )


def test_resolve_output_target_sanitizes_derived_filename_stem(tmp_path):
    base_dir = tmp_path / "outputs"
    base_dir.mkdir()
    cfg = ServeOutputConfig(transport="fs", format="jsonl", directory=base_dir)

    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="train/val",
    )

    assert target.destination == (base_dir / "train_val" / "train_val.jsonl").resolve()


def test_routed_targets_keep_profile_names_in_shared_run(tmp_path):
    config = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=tmp_path / "outputs",
    )
    run = get_run_paths(tmp_path / "outputs", run_id="shared")
    first = resolve_output_target(
        cli_output=None,
        config_output=config,
        profile_name="first",
        run_paths=run,
    )
    second = resolve_output_target(
        cli_output=None,
        config_output=config,
        profile_name="second",
        run_paths=run,
    )
    normal = resolve_output_target(
        cli_output=None,
        config_output=config,
        profile_name="train",
        run_paths=run,
    )

    destinations = {
        first.for_output("train").destination,
        second.for_output("train").destination,
        normal.destination,
    }

    assert {path.name for path in destinations if path is not None} == {
        "first.train.jsonl",
        "second.train.jsonl",
        "train.jsonl",
    }


def test_resolve_output_target_honors_view(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cfg = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        view="flat",
        directory=out_dir,
    )

    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="train",
    )

    assert target.view == "flat"


def test_resolve_output_target_honors_encoding(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cfg = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        encoding="utf-8-sig",
        directory=out_dir,
    )

    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="train",
    )

    assert target.encoding == "utf-8-sig"


def test_stdout_jsonl_defaults_to_raw_view():
    target = resolve_output_target(
        cli_output=ServeOutputConfig(transport="stdout", format="jsonl"),
        config_output=None,
        default=None,
        base_path=Path("."),
        profile_name="train",
    )
    assert target.view == "raw"


def test_stdout_txt_is_supported():
    target = resolve_output_target(
        cli_output=ServeOutputConfig(transport="stdout", format="txt"),
        config_output=None,
        default=None,
        base_path=Path("."),
        profile_name="report",
    )
    assert target.transport == "stdout"
    assert target.format == "txt"


def test_resolve_output_target_html_uses_html_suffix(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cfg = ServeOutputConfig(
        transport="fs",
        format="html",
        directory=out_dir,
    )

    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="matrix",
    )

    assert target.destination == (out_dir / "matrix" / "matrix.html").resolve()


def test_resolve_output_target_txt_uses_txt_suffix(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cfg = ServeOutputConfig(
        transport="fs",
        format="txt",
        directory=out_dir,
    )

    target = resolve_output_target(
        cli_output=None,
        config_output=cfg,
        default=None,
        base_path=Path("."),
        profile_name="coverage",
    )

    assert target.destination == (out_dir / "coverage" / "coverage.txt").resolve()
    assert target.encoding == "utf-8"


def test_resolve_output_target_parquet_uses_flat_binary_output(tmp_path):
    out_dir = tmp_path / "outputs"
    config = ServeOutputConfig(
        transport="fs",
        format="parquet",
        directory=out_dir,
    )

    target = resolve_output_target(
        cli_output=None,
        config_output=config,
        default=None,
        base_path=Path("."),
        profile_name="research",
    )

    assert target.destination == (out_dir / "research" / "research.parquet").resolve()
    assert target.view == "flat"
    assert target.encoding is None
    assert (
        target.for_output("fold_0.train").destination
        == (out_dir / "research" / "research.fold_0.train.parquet").resolve()
    )


def test_parquet_output_rejects_raw_view_encoding_and_gzip(tmp_path):
    with pytest.raises(ValueError, match="parquet output supports only view='flat'"):
        ServeOutputConfig(
            transport="fs",
            format="parquet",
            view="raw",
            directory=tmp_path,
        )
    with pytest.raises(ValueError, match="parquet output does not support encoding"):
        ServeOutputConfig(
            transport="fs",
            format="parquet",
            encoding="utf-8",
            directory=tmp_path,
        )
    with pytest.raises(
        ValueError, match="gzip compression supports only jsonl and csv"
    ):
        ServeOutputConfig(
            transport="fs",
            format="parquet",
            compression="gzip",
            directory=tmp_path,
        )


def test_html_output_rejects_view_and_encoding(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="html output does not support view"):
        ServeOutputConfig(
            transport="fs",
            format="html",
            view="raw",
            directory=out_dir,
        )
    with pytest.raises(ValueError, match="html output does not support encoding"):
        ServeOutputConfig(
            transport="fs",
            format="html",
            encoding="utf-8",
            directory=out_dir,
        )


def test_txt_output_rejects_view(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="txt output does not support view"):
        ServeOutputConfig(
            transport="fs",
            format="txt",
            view="raw",
            directory=out_dir,
        )


def test_pickle_output_rejects_flat_view(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="pickle output supports only view='raw'"):
        ServeOutputConfig(
            transport="fs",
            format="pickle",
            view="flat",
            directory=out_dir,
        )


@pytest.mark.parametrize(
    ("transport", "format_", "message"),
    [
        ("stdout", "jsonl", "stdout outputs do not support compression"),
        ("fs", "pickle", "gzip compression supports only jsonl and csv"),
        ("fs", "txt", "gzip compression supports only jsonl and csv"),
        ("fs", "html", "gzip compression supports only jsonl and csv"),
    ],
)
def test_output_config_rejects_unsupported_compression(
    tmp_path,
    transport,
    format_,
    message,
) -> None:
    values = {
        "transport": transport,
        "format": format_,
        "compression": "gzip",
    }
    if transport == "fs":
        values["directory"] = tmp_path

    with pytest.raises(ValueError, match=message):
        ServeOutputConfig.model_validate(values)

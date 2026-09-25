import pytest

from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.config.options import OUTPUT_FORMATS, OUTPUT_INSPECT_FORMATS
from jerrythomas.config.profiles.build import ARTIFACT_MODES


@pytest.mark.parametrize("command", ["serve", "build", "inspect", "export"])
@pytest.mark.parametrize("mode", ARTIFACT_MODES)
def test_execution_commands_accept_the_same_artifact_modes(command, mode):
    args = build_parser().parse_args([command, "--artifact-mode", mode])

    assert args.artifact_mode == mode


@pytest.mark.parametrize("mode", ["force", "off", "AUTO", "invalid"])
def test_build_rejects_invalid_artifact_modes(mode):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["build", "--artifact-mode", mode])

    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["build", "--force"],
        ["export", "--output", "out.jsonl"],
        ["serve", "--pro", "training"],
        ["stream", "create", "--proj", "equities"],
    ],
)
def test_removed_or_abbreviated_flags_are_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["source", "create", "weather.daily", "--transport", "synthetic"],
        ["stream", "create", "--identity"],
        ["inflow", "create"],
        ["plugin", "create", "weather-plugin"],
        ["domain", "create", "weather"],
        ["dto", "create", "Observation"],
        ["parser", "create", "ObservationParser"],
        ["mapper", "create", "map_observation"],
        ["loader", "create", "observations"],
        ["demo", "create"],
        ["list", "datasets"],
        ["list", "domains"],
        ["version"],
        ["env"],
    ],
)
def test_logging_flags_work_after_each_nested_leaf(argv):
    args = build_parser().parse_args(
        [
            *argv,
            "--log-level",
            "debug",
            "--log-output",
            "stdout",
            "--log-output",
            "fs:run.log",
        ]
    )

    assert args.log_level == "DEBUG"
    assert args.log_output == ["stdout", "fs:run.log"]


@pytest.mark.parametrize(
    ("argv", "level", "outputs"),
    [
        (
            ["--log-level", "DEBUG", "--log-output", "stdout", "stream", "create"],
            "DEBUG",
            ["stdout"],
        ),
        (
            ["stream", "--log-level", "WARNING", "--log-output", "stderr", "create"],
            "WARNING",
            ["stderr"],
        ),
        (
            [
                "--log-level",
                "DEBUG",
                "--log-output",
                "stdout",
                "stream",
                "--log-level",
                "WARNING",
                "--log-output",
                "stderr",
                "create",
                "--log-level",
                "ERROR",
                "--log-output",
                "fs:leaf.log",
                "--log-output",
                "execution",
            ],
            "ERROR",
            ["fs:leaf.log", "execution"],
        ),
    ],
)
def test_logging_preserves_outer_defaults_and_deeper_explicit_overrides(
    argv, level, outputs
):
    args = build_parser().parse_args(argv)

    assert args.log_level == level
    assert args.log_output == outputs


@pytest.mark.parametrize(
    ("command", "formats"),
    [("serve", OUTPUT_FORMATS), ("inspect", OUTPUT_INSPECT_FORMATS)],
)
def test_shared_output_flags_keep_command_specific_format_choices(command, formats):
    for fmt in formats:
        args = build_parser().parse_args([command, "--output-format", fmt])
        assert args.output_format == fmt
    rejected = "html" if command == "serve" else "parquet"
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--output-format", rejected])
    assert exc.value.code == 2

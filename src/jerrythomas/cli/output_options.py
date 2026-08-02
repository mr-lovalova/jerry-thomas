import logging
from pathlib import Path

from pydantic import ValidationError

from jerrythomas.config.profiles.output import ServeOutputConfig
from jerrythomas.services.path_policy import resolve_workspace_path


logger = logging.getLogger(__name__)

_OUTPUT_MATRIX_HELP = (
    "Valid output combinations:\n"
    "  stdout: format=jsonl|txt\n"
    "          jsonl view=raw|flat, txt has no view\n"
    "          encoding is not supported\n"
    "  fs:     format=jsonl|csv|parquet|pickle|txt|html\n"
    "          encoding is supported only for jsonl/csv/txt (default utf-8)\n"
    "          gzip compression is supported only for jsonl/csv\n"
    "          jsonl supports view=raw|flat\n"
    "          csv supports view=flat\n"
    "          parquet supports view=flat and uses internal compression\n"
    "          pickle supports view=raw\n"
    "          txt/html output does not support view\n"
    "          html output support depends on the selected runtime operation\n"
)


def build_cli_output_config(
    transport: str | None,
    fmt: str | None,
    directory: str | None,
    output_encoding: str | None = None,
    output_compression: str | None = None,
    workspace_root: Path | None = None,
    view: str | None = None,
) -> ServeOutputConfig | None:
    if (
        transport is None
        and fmt is None
        and directory is None
        and view is None
        and output_encoding is None
        and output_compression is None
    ):
        return None

    if not transport or not fmt:
        logger.error("--output-transport and --output-format must be provided together")
        raise SystemExit(2)
    transport = transport.lower()
    fmt = fmt.lower()

    config: dict[str, object] = {
        "format": fmt,
        "view": view,
        "encoding": output_encoding,
        "compression": output_compression,
    }
    if transport == "fs":
        if not directory:
            logger.error("--output-directory is required when --output-transport=fs")
            raise SystemExit(2)
        config["transport"] = "fs"
        config["directory"] = resolve_workspace_path(directory, workspace_root)
    elif transport == "stdout":
        config["transport"] = "stdout"
    else:
        logger.error("--output-transport must be 'fs' or 'stdout'")
        raise SystemExit(2)
    if transport != "fs" and directory:
        logger.error("--output-directory is only valid when --output-transport=fs")
        raise SystemExit(2)
    try:
        return ServeOutputConfig.model_validate(config)
    except ValidationError as exc:
        logger.error("Invalid output configuration: %s", exc.errors()[0]["msg"])
        logger.error(_OUTPUT_MATRIX_HELP)
        raise SystemExit(2) from exc

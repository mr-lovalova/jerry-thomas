import shutil
from importlib.resources import as_file, files
from pathlib import Path

from jerrythomas.plugins import MAPPERS_EP, PARSERS_EP
from jerrythomas.services.scaffold.entrypoints import register_entry_point
from jerrythomas.services.scaffold.plugin import create_plugin_base

_DEMO_NAME = "demo"
_IGNORE_GENERATED_FILES = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
)


def _install_demo_files(plugin_root: Path, template_root: Path) -> None:
    shutil.copy2(template_root / "README.md", plugin_root / "README.md")
    shutil.copy2(template_root / "jerry.yaml", plugin_root / "jerry.yaml")
    shutil.copytree(
        template_root / "demo",
        plugin_root / "demo",
        ignore=_IGNORE_GENERATED_FILES,
    )

    demo_package = template_root / "src" / "demo"
    plugin_package = plugin_root / "src" / "demo"
    shutil.copytree(
        demo_package,
        plugin_package,
        dirs_exist_ok=True,
        ignore=_IGNORE_GENERATED_FILES,
    )

    pyproject = plugin_root / "pyproject.toml"
    register_entry_point(
        pyproject,
        PARSERS_EP,
        "sandbox_ohlcv_dto_parser",
        "demo.parsers.sandbox_ohlcv_dto_parser:SandboxOhlcvDTOParser",
    )
    register_entry_point(
        pyproject,
        MAPPERS_EP,
        "map_sandbox_ohlcv_dto_to_equity",
        "demo.mappers.map_sandbox_ohlcv_dto_to_equity:map_sandbox_ohlcv_dto_to_equity",
    )


def scaffold_demo(outdir: Path) -> Path:
    plugin_root = create_plugin_base(_DEMO_NAME, outdir)
    try:
        template_ref = files("jerrythomas") / "templates" / "demo_skeleton"
        with as_file(template_ref) as template_root:
            _install_demo_files(plugin_root, template_root)
    except BaseException:
        shutil.rmtree(plugin_root)
        raise

    return plugin_root

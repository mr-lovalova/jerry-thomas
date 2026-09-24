import keyword
import re
import shutil
import sys
from importlib.resources import as_file, files
from pathlib import Path

_RESERVED_DISTRIBUTION_NAMES = {"jerry-thomas"}
_RESERVED_PACKAGE_NAMES = {"jerrythomas", "test", "tests"}
_STDLIB_MODULE_NAMES = {name.casefold() for name in sys.stdlib_module_names}
_DISTRIBUTION_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")


def _normalized_package_name(dist_name: str) -> str:
    if _DISTRIBUTION_NAME.fullmatch(dist_name) is None:
        raise ValueError(f"'{dist_name}' is not a valid Python distribution name")
    normalized_dist_name = re.sub(r"[-_.]+", "-", dist_name).casefold()
    if normalized_dist_name in _RESERVED_DISTRIBUTION_NAMES:
        raise ValueError(f"'{dist_name}' conflicts with the Jerry Thomas distribution")

    package_name = dist_name.replace("-", "_").replace(".", "_")
    normalized_package_name = package_name.casefold()
    if normalized_package_name in _RESERVED_PACKAGE_NAMES:
        raise ValueError(f"'{package_name}' is reserved; choose another plugin name")
    if normalized_package_name in _STDLIB_MODULE_NAMES:
        raise ValueError(
            f"Plugin name '{dist_name}' conflicts with the Python standard library"
        )
    if not package_name.isidentifier() or keyword.iskeyword(package_name):
        raise ValueError(
            "Plugin names must form valid Python package names after replacing "
            "'-' and '.' with '_'"
        )
    return package_name


def create_plugin_base(name: str, outdir: Path) -> Path:
    package_name = _normalized_package_name(name)
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / name
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"{target} already exists") from exc

    try:
        skeleton_ref = files("jerrythomas") / "templates" / "plugin_skeleton"
        with as_file(skeleton_ref) as skeleton_dir:
            shutil.copytree(
                skeleton_dir,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "*.pyo",
                    ".DS_Store",
                ),
            )

        package_dir = target / "src" / "{{PACKAGE_NAME}}"
        package_dir.rename(target / "src" / package_name)

        pyproject = target / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        pyproject.write_text(text.replace("{{DIST_NAME}}", name), encoding="utf-8")
    except BaseException:
        shutil.rmtree(target)
        raise
    return target


def _install_dataset_files(plugin_root: Path, template_root: Path) -> None:
    shutil.copy2(template_root / "README.md", plugin_root / "README.md")
    shutil.copy2(template_root / "jerry.yaml", plugin_root / "jerry.yaml")
    shutil.copytree(
        template_root / "your-project",
        plugin_root / "your-project",
    )


def scaffold_plugin(name: str, outdir: Path) -> Path:
    target = create_plugin_base(name, outdir)
    try:
        template_ref = files("jerrythomas") / "templates" / "dataset_skeleton"
        with as_file(template_ref) as template_root:
            _install_dataset_files(target, template_root)
    except BaseException:
        shutil.rmtree(target)
        raise

    return target

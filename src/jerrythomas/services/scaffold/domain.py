from pathlib import Path

from jerrythomas.services.scaffold.layout import (
    DIR_DOMAINS,
    TPL_DOMAIN_RECORD,
    domain_record_class,
)
from jerrythomas.services.scaffold.locking import ScaffoldLock, acquire_scaffold_lock
from jerrythomas.services.scaffold.paths import (
    ensure_base_pkg_dir,
    pkg_root,
    resolve_base_pkg_dir,
)
from jerrythomas.services.scaffold.templates import render
from jerrythomas.services.scaffold.utils import (
    ensure_pkg_dir,
    is_python_identifier,
    rollback_new_scaffold_paths,
    write_new_file,
)


def domain_scaffold_paths(domain: str, root: Path | None) -> tuple[Path, ...]:
    root_dir, name, _ = pkg_root(root)
    base = resolve_base_pkg_dir(root_dir, name)
    domain_dir = base / DIR_DOMAINS / domain
    return base / "__init__.py", domain_dir / "__init__.py", domain_dir / "model.py"


def validate_domain_creation(domain: str, root: Path | None) -> None:
    if not is_python_identifier(domain):
        raise ValueError("Domain name must be a valid Python identifier")
    path = domain_scaffold_paths(domain, root)[-1]
    if path.exists():
        raise FileExistsError(f"{path} already exists")


def create_domain(
    domain: str,
    root: Path | None,
    scaffold_lock: ScaffoldLock | None = None,
) -> Path:
    root_dir, name, pyproject = pkg_root(root)
    with acquire_scaffold_lock(pyproject.parent, scaffold_lock):
        validate_domain_creation(domain, root)
        created_paths = domain_scaffold_paths(domain, root)
        with rollback_new_scaffold_paths(created_paths):
            base = ensure_base_pkg_dir(root_dir, name)
            package_name = base.name
            pkg_dir = ensure_pkg_dir(base / DIR_DOMAINS, domain)
            path = pkg_dir / "model.py"
            write_new_file(
                path,
                render(
                    TPL_DOMAIN_RECORD,
                    PACKAGE_NAME=package_name,
                    DOMAIN=domain,
                    CLASS_NAME=domain_record_class(domain),
                    PARENT_CLASS="TemporalRecord",
                    time_aware=True,
                ),
            )
            return path

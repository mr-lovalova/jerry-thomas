import os
import stat
from pathlib import Path


def pipeline_yaml_files(root: Path) -> tuple[Path, ...]:
    root_metadata = root.stat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise NotADirectoryError(
            f"Pipeline configuration path is not a directory: {root}"
        )

    paths: list[Path] = []

    def raise_scan_error(error: OSError) -> None:
        raise error

    for directory, directory_names, file_names in os.walk(
        root,
        onerror=raise_scan_error,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in directory_names:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError(
                    f"Pipeline configuration directories must not be symlinks: {path}"
                )
        for name in file_names:
            path = directory_path / name
            if path.suffix not in {".yaml", ".yml"}:
                continue
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"YAML config is not a regular file: {path}")
            paths.append(path)

    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))

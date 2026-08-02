import subprocess
import sys
import time
import tomllib
from pathlib import Path
from textwrap import dedent

import pytest
from tomlkit.exceptions import ParseError

from jerrythomas.plugins import MAPPERS_EP, PARSERS_EP
from jerrythomas.services.scaffold.entrypoints import (
    read_entry_points,
    register_entry_point,
)
from jerrythomas.services.scaffold.locking import acquire_scaffold_lock


def test_register_entry_point_serializes_concurrent_writers(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'example'\n", encoding="utf-8")
    ready = tmp_path / "ready"
    go = tmp_path / "go"
    blocked = tmp_path / "blocked"
    entered_update = tmp_path / "entered-update"
    source_root = Path(__file__).parents[3] / "src"
    script = dedent(
        """
        import sys
        import time
        from pathlib import Path

        sys.path.insert(0, sys.argv[1])

        import jerrythomas.services.scaffold.entrypoints as entrypoints
        import jerrythomas.services.execution_lock as execution_lock

        original_update = entrypoints._register_entry_point
        original_try_lock = execution_lock.try_acquire_file_lock

        def marked_update(*args, **kwargs):
            Path(sys.argv[6]).touch()
            return original_update(*args, **kwargs)

        def marked_try_lock(lock_file):
            acquired = original_try_lock(lock_file)
            if not acquired:
                Path(sys.argv[5]).touch()
            return acquired

        entrypoints._register_entry_point = marked_update
        execution_lock.try_acquire_file_lock = marked_try_lock
        Path(sys.argv[3]).touch()
        while not Path(sys.argv[4]).exists():
            time.sleep(0.01)
        entrypoints.register_entry_point(
            Path(sys.argv[2]),
            "jerrythomas.parsers",
            "second",
            "package:second",
        )
        """
    )
    command = [
        sys.executable,
        "-c",
        script,
        str(source_root),
        str(pyproject),
        str(ready),
        str(go),
        str(blocked),
        str(entered_update),
    ]

    with acquire_scaffold_lock(tmp_path) as scaffold_lock:
        process = subprocess.Popen(command)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists()
            go.touch()
            deadline = time.monotonic() + 10
            while not blocked.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert blocked.exists()
            assert not entered_update.exists()
            register_entry_point(
                pyproject,
                PARSERS_EP,
                "first",
                "package:first",
                scaffold_lock=scaffold_lock,
            )
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise

    try:
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    assert read_entry_points(pyproject, PARSERS_EP) == {
        "first": "package:first",
        "second": "package:second",
    }


def test_register_entry_point_preserves_unrelated_toml(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """# project comment
[project]
name = 'example'

[project.entry-points."jerrythomas.parsers"]
# parser comment
existing = 'package:old' # keep this inline comment
untouched = 'package:untouched'

[tool.example]
answer = 42 # unrelated comment
""",
        encoding="utf-8",
    )

    changed = register_entry_point(
        pyproject,
        PARSERS_EP,
        "existing",
        "package:new",
    )

    assert changed is True
    text = pyproject.read_text(encoding="utf-8")
    assert "# project comment" in text
    assert "# parser comment" in text
    assert 'existing = "package:new" # keep this inline comment' in text
    assert "untouched = 'package:untouched'" in text
    assert "answer = 42 # unrelated comment" in text
    parsed = tomllib.loads(text)
    assert parsed["project"]["entry-points"][PARSERS_EP]["existing"] == "package:new"


def test_register_entry_point_noop_leaves_file_byte_identical(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    original = (
        b"[project]\nname='example'\n\n"
        b"[project.entry-points.'jerrythomas.parsers']\nparser='package:parser'\n"
    )
    pyproject.write_bytes(original)

    changed = register_entry_point(
        pyproject,
        PARSERS_EP,
        "parser",
        "package:parser",
    )

    assert changed is False
    assert pyproject.read_bytes() == original


def test_register_entry_point_preserves_pyproject_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real.toml"
    target.write_text("[project]\nname = 'example'\n", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.symlink_to(target)

    register_entry_point(pyproject, PARSERS_EP, "parser", "package:parser")

    assert pyproject.is_symlink()
    assert read_entry_points(target, PARSERS_EP) == {"parser": "package:parser"}


def test_register_entry_point_adds_missing_tables(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\nname = 'example'\n\n# keep\n[tool.example]\nvalue = true\n",
        encoding="utf-8",
    )

    register_entry_point(pyproject, MAPPERS_EP, "map", "package:map")

    text = pyproject.read_text(encoding="utf-8")
    assert "# keep" in text
    assert text.index("# keep") < text.index("[tool.example]")
    assert text.index("[tool.example]") < text.index(
        '[project.entry-points."jerrythomas.mappers"]'
    )
    assert tomllib.loads(text)["project"]["entry-points"][MAPPERS_EP] == {
        "map": "package:map"
    }


def test_register_entry_point_preserves_crlf_newlines(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_bytes(
        b"[project]\r\nname = 'example'\r\n\r\n"
        b"[project.entry-points.'jerrythomas.parsers']\r\n"
        b"parser = 'package:old'\r\n"
    )

    register_entry_point(pyproject, PARSERS_EP, "parser", "package:new")

    content = pyproject.read_bytes()
    assert b"\r\n" in content
    assert b"\n" not in content.replace(b"\r\n", b"")


@pytest.mark.parametrize(
    "original",
    [
        b"[project]\r\nname = 'example'\r\n",
        (
            b"[project]\r\nname = 'example'\r\n\r\n"
            b"[project.entry-points.'jerrythomas.mappers']\r\n"
            b"mapper = 'package:mapper'\r\n"
        ),
        (
            b"[project]\r\nname = 'example'\r\n\r\n"
            b"[project.entry-points.'jerrythomas.parsers']\r\n"
            b"existing = 'package:existing'\r\n"
        ),
    ],
)
def test_register_entry_point_preserves_crlf_when_adding_entry(
    tmp_path: Path,
    original: bytes,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_bytes(original)

    register_entry_point(pyproject, PARSERS_EP, "parser", "package:parser")

    content = pyproject.read_bytes()
    assert b"\r\n" in content
    assert b"\n" not in content.replace(b"\r\n", b"")
    assert (
        tomllib.loads(content.decode("utf-8"))["project"]["entry-points"][PARSERS_EP][
            "parser"
        ]
        == "package:parser"
    )


def test_read_entry_points_supports_inline_tables(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """[project]
name = "example"
entry-points = { "jerrythomas.parsers" = { first = 'package:first', second = "package:second" } }
""",
        encoding="utf-8",
    )

    assert read_entry_points(pyproject, PARSERS_EP) == {
        "first": "package:first",
        "second": "package:second",
    }

    register_entry_point(pyproject, PARSERS_EP, "third", "package:third")
    assert (
        tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["entry-points"][
            PARSERS_EP
        ]["third"]
        == "package:third"
    )


@pytest.mark.parametrize(
    "content,error",
    [
        ("name = 'missing project'\n", "must define a \\[project\\] table"),
        (
            "[project]\nname = 'example'\nentry-points = []\n",
            "project.entry-points must be a TOML table",
        ),
        (
            "[project]\nname = 'example'\n"
            "[project.entry-points]\n'jerrythomas.parsers' = []\n",
            "project.entry-points.jerrythomas.parsers must be a TOML table",
        ),
        (
            "[project]\nname = 'example'\n"
            "[project.entry-points.'jerrythomas.parsers']\nparser = 3\n",
            "Entry point 'parser'.*must be a string",
        ),
    ],
)
def test_register_entry_point_rejects_invalid_structure(
    tmp_path: Path,
    content: str,
    error: str,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        register_entry_point(pyproject, PARSERS_EP, "parser", "package:parser")


def test_register_entry_point_rejects_malformed_toml(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project\n", encoding="utf-8")

    with pytest.raises(ParseError):
        register_entry_point(pyproject, PARSERS_EP, "parser", "package:parser")

    pyproject.write_text("[project]\nname = 'example'\n", encoding="utf-8")
    assert register_entry_point(
        pyproject,
        PARSERS_EP,
        "parser",
        "package:parser",
    )

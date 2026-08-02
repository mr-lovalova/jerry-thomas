import subprocess
import sys
from pathlib import Path
from textwrap import dedent

from jerrythomas.services.execution_lock import project_execution_lock


def test_project_execution_lock_excludes_another_process(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    source_root = Path(__file__).parents[3] / "src"
    script = dedent(
        """
        import sys
        from pathlib import Path

        sys.path.insert(0, sys.argv[1])

        from jerrythomas.services.execution_lock import (
            ProjectExecutionBusyError,
            project_execution_lock,
        )

        try:
            with project_execution_lock(Path(sys.argv[2])):
                pass
        except ProjectExecutionBusyError:
            print("busy")
        else:
            print("acquired")
        """
    )
    command = [
        sys.executable,
        "-c",
        script,
        str(source_root),
        str(artifacts_root),
    ]

    with project_execution_lock(artifacts_root):
        blocked = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    acquired = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert blocked.returncode == 0, blocked.stderr
    assert blocked.stdout.strip() == "busy"
    assert acquired.returncode == 0, acquired.stderr
    assert acquired.stdout.strip() == "acquired"

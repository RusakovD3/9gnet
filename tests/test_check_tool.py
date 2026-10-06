"""The verification entry point must report failures and isolate artifacts."""

from pathlib import Path
import sys

import pytest

from tools.check import (
    prepare_smoke_workspace,
    project_python_files,
    run_command,
    verify_smoke_exports,
)


def test_command_keeps_failure_code_and_log(tmp_path: Path) -> None:
    log = tmp_path / "failed.log"
    result = run_command(
        [sys.executable, "-c", "print('diagnostic: проверка'); raise SystemExit(7)"],
        cwd=tmp_path, log=log,
    )
    assert result["exit_code"] == 7
    assert "diagnostic: проверка" in log.read_text(encoding="utf-8")


def test_smoke_copy_excludes_user_data_and_preserves_existing_output(tmp_path: Path) -> None:
    root = tmp_path / "project"
    for name in ("main.py", "src/gnet9/model.py", "policies/l1_policies.d0sl",
                 "output/result.json", "private/notes.py", ".venv/library.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
    destination = tmp_path / "isolated"
    prepare_smoke_workspace(root, destination)
    assert sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()) == [
        "main.py", "policies/l1_policies.d0sl", "src/gnet9/model.py",
    ]
    (destination / "main.py").write_text("changed", encoding="utf-8")
    assert (root / "main.py").read_text(encoding="utf-8") == "original"
    assert (root / "output/result.json").read_text(encoding="utf-8") == "original"
    assert all("private" not in p.parts and ".venv" not in p.parts for p in project_python_files(root))
    with pytest.raises(FileExistsError):
        prepare_smoke_workspace(root, destination)


def test_smoke_rejects_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing or empty"):
        verify_smoke_exports(tmp_path)


def test_command_reports_timeout(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path, log=tmp_path / "timeout.log", timeout=1,
    )
    assert result["exit_code"] == 124
    assert result["timed_out"] is True

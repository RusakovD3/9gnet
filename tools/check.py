"""Portable local/CI checks; never overwrite a normal simulation run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "networkx", "matplotlib", "pytest")
QUICK_TESTS = (
    "tests/test_main_exports.py",
    "tests/test_telemetry_validation.py",
    "tests/test_check_tool.py",
)
SMOKE_FILES = (
    "baseline_topology.json", "baseline_topology.graphml", "baseline_summary.txt",
    "l1_d0sl_profiles.json", "l1_monitoring.csv", "l2_equipment_profiles.json",
    "l0_server_service_catalog.json", "ip_address_plan.json", "mitre_attack_catalog.json",
    "l1_d0sl_parsed.json", "l1_policies.d0sl", "network_dynamics.json",
    "algorithm_dialogue.json", "algorithm_dialogue.md", "standards_and_equipment_audit.json",
    "network_logic.png", "layer_scheme.png", "ip_address_map.png", "service_flows.png",
    "charts/dynamics_overview.png", "charts/capacity_bottlenecks.png",
    "step_log.csv",
)


def project_python_files(root: Path) -> list[Path]:
    """Only maintained code; exclude outputs, environments and user materials."""
    files = [root / "main.py"]
    for folder in ("src", "tests", "tools"):
        files.extend(sorted((root / folder).rglob("*.py")))
    return files


def check_syntax(root: Path) -> None:
    for path in project_python_files(root):
        compile(path.read_bytes(), str(path), "exec")


def run_command(command: list[str], *, cwd: Path, log: Path, timeout: int = 1800) -> dict:
    """Use an argument list, explicit interpreter and headless child environment."""
    env = os.environ.copy()
    env.update(
        MPLBACKEND="Agg", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
        MPLCONFIGDIR=str(log.parent / "matplotlib"),
    )
    started = time.monotonic()
    timed_out = False
    with log.open("w", encoding="utf-8") as stream:
        try:
            result = subprocess.run(
                command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            code = 124
            stream.write(f"\nTimeout after {timeout} seconds\n")
    print(log.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
    return {
        "command": command, "cwd": str(cwd), "exit_code": code,
        "timed_out": timed_out, "seconds": round(time.monotonic() - started, 3),
        "log": str(log),
    }


def prepare_smoke_workspace(root: Path, destination: Path) -> None:
    """Copy only executable model inputs; main.py resolves output relative to itself."""
    destination.mkdir(parents=True, exist_ok=False)
    inputs = [root / "main.py", root / "policies" / "l1_policies.d0sl"]
    inputs.extend(sorted((root / "src").rglob("*.py")))
    for source in inputs:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def verify_smoke_exports(output: Path) -> dict[str, str]:
    """Read actual exported formats and basic healthy-run contracts."""
    digests = {}
    for name in SMOKE_FILES:
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty artifact: {name}")
        data = path.read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        if path.suffix == ".json":
            json.loads(data)
        if path.suffix == ".png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Invalid PNG signature: {name}")

    topology = json.loads((output / "baseline_topology.json").read_text(encoding="utf-8"))
    if topology["level_summary"]["L1"] != 480 or topology["level_summary"]["L2"] != 30:
        raise ValueError("Unexpected baseline L1/L2 counts")
    if len(topology["services"]) != 6 or len(topology["servers"]) != 4:
        raise ValueError("Unexpected service/server counts")
    graphml = ET.parse(output / "baseline_topology.graphml")
    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
    if len(graphml.findall(".//g:node", namespace)) != len(topology["nodes"]):
        raise ValueError("GraphML/JSON node count mismatch")
    if len(graphml.findall(".//g:edge", namespace)) != len(topology["edges"]):
        raise ValueError("GraphML/JSON edge count mismatch")
    with (output / "l1_monitoring.csv").open(encoding="utf-8", newline="") as stream:
        if sum(1 for _ in csv.DictReader(stream)) != 480 * 30:
            raise ValueError("Unexpected monitoring row count")
    dynamics = json.loads((output / "network_dynamics.json").read_text(encoding="utf-8"))
    if not dynamics["health"]["ok"] or len(dynamics["snapshots"]) != 2:
        raise ValueError("Smoke requires healthy t0 and one subsequent snapshot")
    if [item["step_index"] for item in dynamics["snapshots"]] != [0, 1]:
        raise ValueError("Unexpected smoke snapshot sequence")
    with (output / "step_log.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["шаг"] for row in rows] != ["0", "1"]:
        raise ValueError("Журнал шагов не соответствует состояниям модели")
    expected_update = "да" if dynamics["snapshots"][1]["training"]["operator_updated"] else "нет"
    if rows[1]["модель_обновлена"] != expected_update:
        raise ValueError("Журнал неверно отражает обновление прогноза")
    if (output / "service_flows_remapped.png").exists():
        raise ValueError("Healthy smoke unexpectedly exported remapped flows")
    return digests


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("doctor", "quick", "full", "smoke"))
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = ROOT / "output" / "verification" / f"{stamp}-{args.profile}-{uuid.uuid4().hex[:8]}"
    report_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "profile": args.profile, "started_utc": stamp, "python": sys.version,
        "executable": sys.executable, "packages": {}, "checks": [], "passed": False,
    }
    print(f"Report: {report_dir}", flush=True)

    def command(name: str, arguments: list[str], cwd: Path = ROOT) -> None:
        print(f"Running {name}...", flush=True)
        result = run_command(arguments, cwd=cwd, log=report_dir / f"{name}.log")
        report["checks"].append({"name": name, **result})
        if result["exit_code"]:
            raise RuntimeError(f"{name} failed with exit code {result['exit_code']}")

    try:
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10+ is required")
        for package in PACKAGES:
            report["packages"][package] = metadata.version(package)
        inputs = project_python_files(ROOT) + [
            ROOT / "policies/l1_policies.d0sl", ROOT / "requirements.txt",
            ROOT / "pyproject.toml",
        ]
        report["inputs_sha256"] = {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        }
        print(f"Python: {sys.executable}\nPackages: {report['packages']}", flush=True)
        command("imports", [sys.executable, "-c", "import numpy, networkx, matplotlib, pytest"])
        command("dependencies", [sys.executable, "-m", "pip", "check"])
        if args.profile in {"quick", "full"}:
            check_syntax(ROOT)
            report["checks"].append({"name": "syntax", "exit_code": 0})
            selection = list(QUICK_TESTS) if args.profile == "quick" else ["tests"]
            command("pytest", [sys.executable, "-m", "pytest", "-q", *selection,
                               f"--junitxml={report_dir / 'pytest.xml'}"])
        elif args.profile == "smoke":
            workspace = report_dir / "workspace"
            prepare_smoke_workspace(ROOT, workspace)
            command("smoke", [sys.executable, "main.py", "--attack-scenario", "none",
                              "--dynamics-steps", "1", "--snapshot-detail", "summary",
                              "--packet-detail", "summary"], cwd=workspace)
            report["artifacts_sha256"] = verify_smoke_exports(workspace / "output")
            report["checks"].append({"name": "export_contract", "exit_code": 0})
        report["passed"] = True
    except (OSError, ValueError, SyntaxError, RuntimeError, metadata.PackageNotFoundError,
            KeyError, ET.ParseError) as error:
        report["error"] = str(error)
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
    finally:
        (report_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    print("PASSED" if report["passed"] else "FAILED", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path

from src.gnet9.arbitrator import aggregate_metric
from src.gnet9.debug_visualizer import PROCESS_BLOCKS, RuntimeCallTracer, export_debug_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_tracer_records_only_project_function_calls() -> None:
    tracer = RuntimeCallTracer(PROJECT_ROOT)
    with tracer:
        assert aggregate_metric({}, "L1", "sla_margin", "min", 1.0) == 1.0

    trace = tracer.to_dict()
    recorded = [node for node in trace["nodes"] if node["function"] == "aggregate_metric"]
    assert len(recorded) == 1
    assert recorded[0]["file"] == "src/gnet9/arbitrator.py"
    assert recorded[0]["calls"] == 1
    assert all(not node["file"].startswith("venv/") for node in trace["nodes"])


def test_debug_artifacts_include_diagrams_and_full_trace(tmp_path: Path) -> None:
    trace = {
        "elapsed_seconds": 0.125,
        "function_count": 2,
        "edge_count": 1,
        "nodes": [
            {"id": "main.py:main", "function": "main", "file": "main.py", "module": "main", "calls": 1, "total_time_ms": 125.0},
            {"id": "src/gnet9/dynamics.py:_snapshot", "function": "_snapshot", "file": "src/gnet9/dynamics.py", "module": "dynamics", "calls": 2, "total_time_ms": 80.0},
        ],
        "edges": [{"source": "main.py:main", "target": "src/gnet9/dynamics.py:_snapshot", "calls": 2}],
    }

    paths = export_debug_artifacts(trace, tmp_path)

    assert set(paths) == {"process", "calls", "trace", "process_json"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    saved_trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
    assert saved_trace["function_count"] == 2
    process = json.loads(paths["process_json"].read_text(encoding="utf-8"))
    assert len(process["blocks"]) == len(PROCESS_BLOCKS)
    assert any(block["block_id"] == "arb" for block in process["blocks"])

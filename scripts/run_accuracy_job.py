"""Run one local CPU experiment in the foreground and persist its exit status."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "final"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = ROOT / "outputs/accuracy-v1"
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    state = {"pid": os.getpid(), "phase": args.phase, "status": "running",
             "started_at": datetime.now(timezone.utc).isoformat()}
    status_file = output / f"worker-{args.phase}.json"
    status_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    started = time.perf_counter()
    sys.argv = [str(ROOT / "scripts/train_accuracy.py"), args.phase, "--precision", "dynamic_int8"]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        traceback.print_exc()
    finally:
        state["seconds"] = time.perf_counter() - started
        state["ended_at"] = datetime.now(timezone.utc).isoformat()
        status_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if state["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

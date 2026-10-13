"""Verify every sealed query's complete offline ranking through the live HTTP API.

This replays saved predictions; it neither scores labels nor changes selection.
Start serve_closure.py first. Only loopback HTTP servers are accepted.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import file_sha256, load_jsonl
from scripts import benchmark_closure as experiment


def request(url, payload=None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with urlopen(Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=60) as response:
        if response.status != 200:
            raise ValueError("server did not return HTTP 200")
        return json.load(response)


def verify(data, run, url, output):
    data, run, output = Path(data), Path(run), Path(output)
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in ("localhost", "127.0.0.1", "::1") or parts.path not in ("", "/") or parts.query or parts.fragment or parts.username:
        raise ValueError("URL must be a loopback HTTP origin")
    if output.exists():
        raise FileExistsError("verification output already exists; choose a new file")
    url = url.rstrip("/")
    report = experiment.read(run / "report.json")
    release = experiment.read(run / "release.json")
    if file_sha256(run / "test_transcripts.json") != report["prediction_sha256"]:
        raise ValueError("sealed predictions differ from the report")
    experiment.verify_selection(data, run)
    expected = {(row["query_id"], row["variant"]): row
                for row in experiment.read(run / "test_transcripts.json")}
    health = request(url + "/health")
    bindings = {"model_sha256": file_sha256(run / "models.json"),
                "report_sha256": file_sha256(run / "report.json"),
                "release_sha256": file_sha256(run / "release.json"),
                "service_sha256": file_sha256(ROOT / "scripts/serve_closure.py"),
                "serving_variant": release["serving_variant"], "pool": experiment.POOL}
    if any(health.get(key) != value for key, value in bindings.items()):
        raise ValueError("live server is not the bound release")
    if request(url + "/api/report") != report:
        raise ValueError("HTTP report differs from sealed report")
    rows = load_jsonl(data / "test.jsonl")
    if len(rows) != report["queries"]:
        raise ValueError("test query count differs")
    started = time.perf_counter()
    rankings, candidate_entries, score_entries = 0, 0, 0
    for number, row in enumerate(rows, 1):
        response = request(url + "/api/search", {"query": row["query"], "k": 10})
        if any(response.get(key) != release[key] for key in ("serving_variant", "selected_variant", "promoted")):
            raise ValueError("response changed serving decision")
        for column, variant in (("baseline", experiment.BASELINE), ("serving", release["serving_variant"])):
            actual, saved = response[column], expected[(row["query_id"], variant)]
            if actual["candidate_ids"] != saved["candidate_ids"] or actual["ranking"] != saved["ranking"]:
                raise ValueError(f"offline/HTTP ranking differs for {row['query_id']} / {column}")
            if [p["id"] for p in actual["results"]] != [p["id"] for p in saved["ranking"][:10]]:
                raise ValueError("displayed top10 differs from full ranking")
            rankings += 1
            candidate_entries += len(actual["candidate_ids"])
            score_entries += len(actual["ranking"])
        if number % 50 == 0:
            print(f"HTTP verified {number}/{len(rows)}", flush=True)
    if request(url + "/health") != health:
        raise ValueError("live server release changed during verification")
    value = {"protocol": experiment.PROTOCOL, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True, "transport": "real loopback HTTP", "queries": len(rows),
        "rankings": rankings, "candidate_id_entries": candidate_entries, "score_entries": score_entries,
        "candidate_order_exact": True, "ranking_ids_and_scores_exact": True, "displayed_top10_exact": True,
        "new_metrics_computed": False, "decision_changed": False,
        "seconds": time.perf_counter() - started, "bindings": bindings,
        "verification_source_sha256": file_sha256(Path(__file__)),
        "html_sha256": file_sha256(ROOT / "web/closure.html"),
        "test_transcripts_sha256": report["prediction_sha256"],
        "note": "Both baseline and deployed variant replayed for every sealed query; a rejected candidate is not deployed."}
    experiment.atomic_json(output, value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/esci-closure-v1"))
    parser.add_argument("--run", type=Path, default=Path("outputs/closure-v1"))
    parser.add_argument("--url", default="http://127.0.0.1:18868")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.run, args.url, args.output or args.run / "serving-verification.json"), indent=2))

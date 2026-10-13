"""Serve the exact frozen closure release, including a failed-gate fallback.

Only the release's serving variant is deployed. A selected model that failed the
independent test gate remains a report result, never an implicit serving change.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecom_search.closure import ClosureEngine
from ecom_search.data import file_sha256
from scripts import benchmark_closure as experiment

RELEASE_ARTIFACTS = ("protocol.json", "trained.json", "models.json", "training.json",
                     "selection.json", "validation.json", "products.sqlite", "report.json")
TRAINED_ARTIFACTS = ("protocol.json", "models.json", "training.json", "products.sqlite")
SELECTION_ARTIFACTS = tuple(name for name in RELEASE_ARTIFACTS
                            if name not in ("selection.json", "report.json"))


def _fingerprints(directory, values, names, label):
    # Requiring the complete name set prevents an empty/partial manifest from
    # silently skipping provenance checks. Names never come from HTTP input.
    if not isinstance(values, dict) or set(values) != set(names):
        raise ValueError(f"incomplete {label} fingerprint manifest")
    experiment.old._assert_fingerprints(directory, values, label)


def _comparison(value):
    if not isinstance(value, dict):
        raise ValueError("invalid comparison")
    bounds = value.get("ci95")
    if bounds is None:
        if value.get("queries") != 0 or value.get("delta") is not None:
            raise ValueError("invalid undefined comparison")
        return
    numbers = [value.get("delta"), *bounds] if isinstance(bounds, list) else []
    if len(numbers) != 3 or any(type(x) not in (float, int) or not math.isfinite(x)
                                or not -1 <= x <= 1 for x in numbers):
        raise ValueError("comparison requires finite recall delta and CI")
    if bounds[0] > bounds[1] or type(value.get("queries")) is not int or value["queries"] < 1:
        raise ValueError("invalid comparison interval or query count")
    if value.get("samples") != 2000 or value.get("seed") != 20261013:
        raise ValueError("comparison bootstrap differs from frozen protocol")


class ClosureSearch:
    def __init__(self, data, run):
        self.data, self.directory = Path(data), Path(run)
        self.engine = None
        try:
            release = experiment.read(self.directory / "release.json")
            if release.get("protocol") != experiment.PROTOCOL:
                raise ValueError("expected completed fresh_query_closure_v1 release")
            _fingerprints(self.directory, release.get("artifacts"), RELEASE_ARTIFACTS, "release artifact")
            _fingerprints(self.data, release.get("data_fingerprints"),
                          (*experiment.old.DATA_FILES, "selection.json"), "data")
            _fingerprints(ROOT, release.get("source_fingerprints"), experiment.SOURCES, "source")
            config, trained = experiment.verify_trained(self.data, self.directory)
            _fingerprints(self.directory, trained.get("artifacts"), TRAINED_ARTIFACTS, "training artifact")
            if (config.get("baseline") != experiment.BASELINE or config.get("primary") != experiment.PRIMARY
                or config.get("promotion_rule") != {"primary_ci95_lower_strictly_above": 0.0,
                    "candidate_recall_delta_ci95_lower_at_least": -.01}
                or config.get("bootstrap") != {"samples": 2000, "seed": 20261013}):
                raise ValueError("release rules differ from frozen protocol")
            for field in ("data_fingerprints", "source_fingerprints"):
                if release[field] != config[field]:
                    raise ValueError(f"release {field} differs from protocol")
            selection = experiment.read(self.directory / "selection.json")
            _fingerprints(self.directory, selection.get("artifacts"), SELECTION_ARTIFACTS, "selection artifact")
            validation = experiment.read(self.directory / "validation.json")
            selected, eligible = experiment.select_variant(validation["summary"])
            if (selected != selection.get("selected") or selected != validation.get("selected")
                or eligible != validation.get("eligible") or selection.get("variant") != experiment.VARIANTS[selected]
                or selection.get("baseline") != experiment.BASELINE or selection.get("pool") != experiment.POOL
                or selection.get("protocol") != experiment.PROTOCOL or selection.get("test_labels_decoded") is not False):
                raise ValueError("validation selection does not match its freeze")
            self.report = experiment.read(self.directory / "report.json")
            report = self.report
            if (report.get("protocol") != experiment.PROTOCOL
                or report.get("protocol_sha256") != file_sha256(self.directory / "protocol.json")
                or report.get("selection_sha256") != file_sha256(self.directory / "selection.json")
                or report.get("validation") != validation or report.get("no_test_retuning") is not True
                or report.get("data_fingerprints") != config["data_fingerprints"]
                or report.get("source_fingerprints") != config["source_fingerprints"]
                or set(report.get("summary", {})) != {experiment.BASELINE, selected}):
                raise ValueError("report does not match the frozen experiment")
            primary, coverage = report["primary_comparison"], report["candidate_coverage_comparison"]
            _comparison(primary)
            _comparison(coverage)
            decision = experiment.release_decision(selected, primary, coverage)
            if report.get("decision") != decision or any(release.get(key) != value for key, value in decision.items()):
                raise ValueError("release decision does not follow the frozen test gate")
            if release.get("pool") != experiment.POOL or release.get("variant") != experiment.VARIANTS[decision["serving_variant"]]:
                raise ValueError("release policy or pool differs from the gated serving variant")
            models = {name: value["weights"] for name, value in experiment.read(self.directory / "models.json").items()}
            if set(models) != {"judged", "retrieved"}:
                raise ValueError("release must bind both trained models")
            self.engine = ClosureEngine(self.data / "catalog.jsonl", self.directory / "products.sqlite", models=models)
            catalog_hash = self.engine.index.metadata["catalog_sha256"]
            products = len(self.engine.catalog)
            if (any(obj.get("catalog_sha256") != catalog_hash for obj in (trained, selection, release))
                or report.get("products") != products):
                raise ValueError("release catalog does not match the serving index")
            self.decision = decision
            self.metadata = {"status": "ok", "protocol": experiment.PROTOCOL, "products": products,
                **decision, "baseline_variant": experiment.BASELINE, "pool": experiment.POOL,
                "catalog_sha256": catalog_hash, "model_sha256": file_sha256(self.directory / "models.json"),
                "report_sha256": file_sha256(self.directory / "report.json"),
                "selection_sha256": file_sha256(self.directory / "selection.json"),
                "release_sha256": file_sha256(self.directory / "release.json"),
                "service_sha256": file_sha256(Path(__file__)), "source": config.get("dataset"),
                "synthetic": config.get("synthetic"), "primary_metric": experiment.PRIMARY,
                "scope": config["scope"], "candidate_policy": "Columns may retrieve different pools; each returns its own complete candidate_ids and ranking.",
                "constraint_note": "Soft policy removes natural-language hard filters. Learned rankers can use negation features; BM25 only uses lexical matching. Explicit -word remains a literal catalog filter, not a semantic guarantee."}
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    def compare(self, obj):
        if not isinstance(obj, dict) or set(obj) - {"query", "k", "pool"}:
            raise ValueError("request must be a JSON object with query, optional k and pool")
        query, k = obj.get("query"), obj.get("k", 10)
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must contain 1 to 2000 characters")
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError("k must be an integer in [1, 20]")
        if type(obj.get("pool", 100)) is not int or obj.get("pool", 100) != experiment.POOL:
            raise ValueError("pool is frozen at 100 for this release")
        baseline = self.engine.search(query, **experiment.VARIANTS[experiment.BASELINE], pool=experiment.POOL, k=k)
        serving = self.engine.search(query, **experiment.VARIANTS[self.decision["serving_variant"]], pool=experiment.POOL, k=k)
        return {"query": query, "pool": experiment.POOL, **self.decision,
                "baseline": baseline, "serving": serving,
                "note": "Each column has its own retrieval pool. Ranking scores are not relevance probabilities."}


def make_server(data, run, port=18868):
    application = ClosureSearch(data, run)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def send(self, status, payload, content_type="application/json; charset=utf-8"):
            body = (payload.encode("utf-8") if isinstance(payload, str)
                    else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.send(200, (ROOT / "web/closure.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
            elif path == "/health":
                self.send(200, application.metadata)
            elif path == "/api/report":
                self.send(200, application.report)
            else:
                self.send(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/search":
                self.send(404, {"error": "unknown endpoint"})
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is unsupported")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not lengths[0].isdigit():
                    raise ValueError("one valid Content-Length is required")
                length = int(lengths[0])
                if not 0 < length <= 8192:
                    raise ValueError("request body must be between 1 and 8192 bytes")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request body")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                def reject_constant(value):
                    raise ValueError(f"invalid JSON constant: {value}")
                self.send(200, application.compare(json.loads(raw, parse_constant=reject_constant)))
            except (ValueError, TypeError, UnicodeError, TimeoutError) as exc:
                self.send(400, {"error": str(exc)})

        def log_message(self, *_):
            pass

    try:
        server = HTTPServer(("127.0.0.1", port), Handler)
    except BaseException:
        application.close()
        raise
    return server, application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/esci-closure-v1"))
    parser.add_argument("--run", type=Path, default=Path("outputs/closure-v1"))
    parser.add_argument("--port", type=int, default=18868)
    args = parser.parse_args()
    server, application = make_server(args.data, args.run, args.port)
    print(f"Frozen closure release: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        application.close()


if __name__ == "__main__":
    main()

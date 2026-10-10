"""Inspect a real public benchmark run and compare retrieval with learned ranking.

Run from the source checkout. This loopback-only research server deliberately
uses only artifacts supplied in --experiment, never the historical report folder.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.cli import SearchEngine
from ecom_search.data import file_sha256


class PublicSearch:
    def __init__(self, experiment):
        self.directory = Path(experiment)
        self.lexical = self.learned = None
        try:
            freeze_path = self.directory / "freeze.json"
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if freeze.get("protocol") != "public_cpu_v1":
                raise ValueError("Expected a frozen public_cpu_v1 experiment")
            artifacts = freeze.get("artifact_fingerprints", {})
            required = ("model.json", "products.sqlite", "selection.json", "train_val_identities.json")
            for name in required:
                if not artifacts.get(name) or file_sha256(self.directory / name) != artifacts[name]:
                    raise ValueError(f"Frozen artifact fingerprint changed: {name}")
            self.lexical = SearchEngine(self.directory / "products.sqlite")
            self.learned = SearchEngine(self.directory / "products.sqlite", self.directory / "model.json")
            report_path = self.directory / "report.json"
            self.report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
            products = int(self.lexical.index.metadata["products"])
            if self.report is not None and (
                self.report.get("freeze_sha256") != file_sha256(freeze_path)
                or self.report.get("data_fingerprints") != freeze.get("data_fingerprints")
                or self.report.get("catalog_products") != products
                or self.report.get("dataset") != freeze.get("dataset")
                or self.report.get("synthetic") != freeze.get("synthetic")
            ):
                raise ValueError("Final report does not match the loaded frozen experiment")
            self.metadata = {
                "status": "ok", "products": products,
                "catalog_sha256": self.lexical.index.metadata["catalog_sha256"],
                "model_sha256": file_sha256(self.directory / "model.json"),
                "report_available": self.report is not None,
                "retrieval": "SQLite FTS BM25 over title and brand; shared candidates",
                "ranking": "validation-selected 12-feature pairwise model; epoch 0 is the untrained control",
                "source": freeze.get("dataset", "user_supplied"), "synthetic": freeze.get("synthetic"),
                "selected_epochs": json.loads((self.directory / "model.json").read_text(encoding="utf-8")).get("selected_epochs"),
                "freeze_sha256": file_sha256(freeze_path),
                "scope": "Research comparison; no live inventory, price or online business metrics",
            }
        except BaseException:
            self.close()
            raise

    def close(self):
        for engine in (self.lexical, self.learned):
            if engine is not None:
                engine.close()

    def compare(self, obj):
        if not isinstance(obj, dict):
            raise ValueError("request must be a JSON object")
        query, k, pool = obj.get("query"), obj.get("k", 10), obj.get("pool", 100)
        if not isinstance(query, str) or not query.strip() or len(query) > 256:
            raise ValueError("query must contain 1 to 256 characters")
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError("k must be an integer in [1, 20]")
        if type(pool) is not int or not k <= pool <= 200:
            raise ValueError("pool must be an integer in [k, 200]")
        baseline = self.lexical.search(query, k=k, pool=pool)
        ranked = self.learned.search(query, k=k, pool=pool)
        # Both engines use the same index and candidate policy. A score is a
        # ranking signal, never a calibrated probability of relevance.
        for product in ranked["products"]:
            values = product.pop("feature_values", {})
            contributions = [
                {"feature": name, "value": values[name], "weight": weight,
                 "contribution": weight * values[name]}
                for name, weight in zip(self.learned.extractor.names, self.learned.model.weights)
                if name in values
            ]
            product["contributions"] = sorted(contributions, key=lambda row: -abs(row["contribution"]))
        return {"query": query, "pool": pool, "baseline": baseline, "learned": ranked,
                "note": "Same lexical candidate pool. Scores across columns are not comparable probabilities."}


def make_server(experiment, port=18867):
    application = PublicSearch(experiment)

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
                self.send(200, (ROOT / "web/public.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
            elif path == "/health":
                self.send(200, application.metadata)
            elif path == "/api/report":
                self.send(200 if application.report else 404,
                          application.report or {"error": "Final report is not available; run final evaluation first."})
            else:
                self.send(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/search":
                self.send(404, {"error": "unknown endpoint"})
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is unsupported")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("request body must be between 1 and 8192 bytes")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request body")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                self.send(200, application.compare(json.loads(raw)))
            except (ValueError, TypeError, UnicodeError) as exc:
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
    parser.add_argument("--experiment", type=Path, default=Path("outputs/public-cpu-v1"))
    parser.add_argument("--port", type=int, default=18867)
    args = parser.parse_args()
    server, application = make_server(args.experiment, args.port)
    print(f"Public benchmark explorer: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        application.close()


if __name__ == "__main__":
    main()

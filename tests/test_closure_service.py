"""Synthetic artifacts exercise deployment binding; no benchmark test is run."""
from __future__ import annotations

import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ecom_search.closure import ClosureEngine
from ecom_search.data import file_sha256, save_json, save_jsonl
from ecom_search.index import build_index, catalog_digest
from ecom_search.ranking import PairwiseRanker
from scripts import benchmark_closure as b
from scripts import serve_closure as service


class ClosureServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data, self.run = self.directory / "data", self.directory / "run"
        self.catalog = [
            {"id": "a", "title": "wireless headphones blue", "brand": "Acme", "description": ""},
            {"id": "b", "title": "wired headphones red", "brand": "Beta", "description": ""},
            {"id": "c", "title": "headphones wireless black", "brand": "Gamma", "description": ""},
        ]
        save_jsonl(self.data / "catalog.jsonl", self.catalog)
        for split in ("train", "val", "test"):
            save_jsonl(self.data / f"{split}.jsonl", [{"query_id": split, "query": f"headphones {split}",
                "candidate_ids": ["a", "b", "c"], "qrels": {"a": 3, "b": 0, "c": 2}, "split": split}])
        save_json(self.data / "selection.json", {"synthetic_fixture": True})
        save_json(self.data / "manifest.json", {"protocol": "closure-esci-v1", "dataset": "synthetic-service-fixture",
            "synthetic": True, "selection_uses_esci_label": False, "source_files": {},
            "previous_experiment_exclusion": {"registry_content_sha256": "fixture-only", "selected_overlap_by_id": 0,
                                              "selected_overlap_by_normalized_query": 0},
            "files": {name: {"sha256": file_sha256(self.data / name), "bytes": (self.data / name).stat().st_size}
                      for name in b.old.DATA_FILES[:-1]}})
        self.config = b.prepare(self.data, self.run)
        build_index(self.catalog, self.run / "products.sqlite")
        save_json(self.run / "models.json", {name: PairwiseRanker().to_dict() for name in ("judged", "retrieved")})
        save_json(self.run / "training.json", {"synthetic_fixture": True})
        save_json(self.run / "trained.json", {"protocol": b.PROTOCOL, "test_labels_decoded": False,
            "catalog_sha256": catalog_digest(self.catalog), "artifacts": b.fingerprints(self.run, service.TRAINED_ARTIFACTS)})
        self.selected = "soft_judged"
        summary = {name: {b.PRIMARY: {"mean": .2, "queries": 20}, b.COVERAGE: {"mean": .6, "queries": 20}}
                   for name in b.VARIANTS}
        summary[self.selected][b.PRIMARY]["mean"] = .3
        self.validation = {"queries": 20, "summary": summary, "selected": self.selected, "eligible": list(b.VARIANTS)}
        save_json(self.run / "validation.json", self.validation)
        save_json(self.run / "selection.json", {"protocol": b.PROTOCOL, "test_labels_decoded": False,
            "selected": self.selected, "baseline": b.BASELINE, "pool": 100, "variant": b.VARIANTS[self.selected],
            "catalog_sha256": catalog_digest(self.catalog), "artifacts": b.fingerprints(self.run, service.SELECTION_ARTIFACTS)})
        primary = {"queries": 20, "delta": .05, "ci95": [.02, .08], "samples": 2000, "seed": 20261013}
        coverage = {"queries": 20, "delta": .01, "ci95": [-.001, .03], "samples": 2000, "seed": 20261013}
        self.report = {"protocol": b.PROTOCOL, "products": 3, "queries": 20, "queries_without_known_positives": 0,
            "summary": {b.BASELINE: {b.PRIMARY: {"mean": .2}}, self.selected: {b.PRIMARY: {"mean": .25}}},
            "primary_comparison": primary, "candidate_coverage_comparison": coverage,
            "decision": b.release_decision(self.selected, primary, coverage), "validation": self.validation,
            "data_fingerprints": self.config["data_fingerprints"], "source_fingerprints": self.config["source_fingerprints"],
            "protocol_sha256": file_sha256(self.run / "protocol.json"),
            "selection_sha256": file_sha256(self.run / "selection.json"), "no_test_retuning": True}
        self.write_release()

    def write_release(self):
        save_json(self.run / "report.json", self.report)
        self.release = {"protocol": b.PROTOCOL, **self.report["decision"], "pool": 100,
            "variant": b.VARIANTS[self.report["decision"]["serving_variant"]], "catalog_sha256": catalog_digest(self.catalog),
            "data_fingerprints": self.config["data_fingerprints"], "source_fingerprints": self.config["source_fingerprints"],
            "artifacts": b.fingerprints(self.run, service.RELEASE_ARTIFACTS)}
        save_json(self.run / "release.json", self.release)

    def open_app(self):
        application = service.ClosureSearch(self.data, self.run)
        self.addCleanup(application.close)
        return application

    def test_promoted_service_matches_offline_full_scores_and_its_own_candidate_pool(self):
        original = b.load_jsonl
        def forbid_labels(path):
            if Path(path).name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                raise AssertionError("serving must not decode experiment labels")
            return original(path)
        with patch.object(b, "load_jsonl", side_effect=forbid_labels):
            app = self.open_app()
        self.assertTrue(app.metadata["promoted"])
        self.assertEqual(app.metadata["serving_variant"], "soft_judged")
        self.assertEqual(app.metadata["report_sha256"], file_sha256(self.run / "report.json"))
        result = app.compare({"query": "headphones without Acme", "k": 1})
        self.assertNotIn("a", result["baseline"]["candidate_ids"])
        self.assertIn("a", result["serving"]["candidate_ids"])
        self.assertEqual(len(result["serving"]["results"]), 1)
        self.assertEqual(len(result["serving"]["ranking"]), 3)
        weights = {name: value["weights"] for name, value in b.read(self.run / "models.json").items()}
        with ClosureEngine(self.data / "catalog.jsonl", self.run / "products.sqlite", models=weights) as offline:
            for key, variant in (("baseline", b.BASELINE), ("serving", self.selected)):
                expected = offline.search("headphones without Acme", **b.VARIANTS[variant], pool=100, k=1)
                self.assertEqual(result[key], expected)
        explicit = app.compare({"query": "headphones -Acme"})
        for key in ("baseline", "serving"):
            self.assertNotIn("a", explicit[key]["candidate_ids"])

    def test_failed_gate_retains_baseline_with_no_hidden_selected_model(self):
        self.report["primary_comparison"]["ci95"] = [-.01, .08]
        self.report["decision"] = b.release_decision(self.selected, self.report["primary_comparison"], self.report["candidate_coverage_comparison"])
        self.write_release()
        app = self.open_app()
        self.assertFalse(app.metadata["promoted"])
        self.assertEqual(app.metadata["selected_variant"], self.selected)
        result = app.compare({"query": "headphones without Acme"})
        self.assertEqual(result["serving_variant"], b.BASELINE)
        self.assertEqual(result["baseline"], result["serving"])

    def test_model_data_and_report_byte_changes_are_rejected(self):
        for path in (self.run / "models.json", self.data / "catalog.jsonl", self.run / "report.json"):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                try:
                    with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                        service.ClosureSearch(self.data, self.run)
                finally:
                    path.write_bytes(before)

    def test_release_cannot_override_failed_gate_even_with_new_report_hash(self):
        self.report["primary_comparison"]["ci95"] = [-.01, .08]
        # Keep the forged promoted decision, and recompute its outer byte binding.
        self.write_release()
        with self.assertRaisesRegex(ValueError, "test gate"):
            service.ClosureSearch(self.data, self.run)

    def test_wrong_report_selection_or_catalog_is_rejected_after_refingerprinting(self):
        pristine = copy.deepcopy(self.report)
        for field, value in (("selection_sha256", "0" * 64), ("protocol_sha256", "0" * 64),
                             ("products", 99), ("no_test_retuning", False)):
            with self.subTest(field=field):
                self.report = dict(pristine, **{field: value})
                self.write_release()
                with self.assertRaises(ValueError):
                    service.ClosureSearch(self.data, self.run)

    def test_missing_fingerprints_and_changed_serving_policy_fail_closed(self):
        pristine = copy.deepcopy(self.release)
        for field, value in (("artifacts", {}), ("source_fingerprints", {}), ("data_fingerprints", {}),
                             ("variant", {"policy": "legacy", "ranker": "judged"}), ("pool", 50)):
            with self.subTest(field=field):
                save_json(self.run / "release.json", dict(pristine, **{field: value}))
                with self.assertRaises(ValueError):
                    service.ClosureSearch(self.data, self.run)

    def test_source_mismatch_is_checked_without_mutating_repo_files(self):
        fake_root = self.directory / "changed-source"
        fake_root.mkdir()
        with patch.object(service, "ROOT", fake_root):
            with self.assertRaisesRegex(ValueError, "source fingerprint changed"):
                service.ClosureSearch(self.data, self.run)

    def test_nonfinite_comparison_cannot_bypass_release_gate(self):
        self.report["primary_comparison"]["ci95"] = [float("nan"), .1]
        with self.assertRaisesRegex(ValueError, "finite"):
            service._comparison(self.report["primary_comparison"])

    def test_request_limits_keep_the_deployed_pool_frozen(self):
        app = self.open_app()
        for payload in ([], {}, {"query": " "}, {"query": "x" * 2001}, {"query": "x", "k": True},
                        {"query": "x", "k": 0}, {"query": "x", "k": 21}, {"query": "x", "pool": 99},
                        {"query": "x", "pool": True}, {"query": "x", "variant": "soft_judged"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    app.compare(payload)

    def test_real_http_routes_body_limits_and_response_contract(self):
        server, app = service.make_server(self.data, self.run, port=0)
        self.addCleanup(app.close)
        self.addCleanup(server.server_close)
        server.timeout = 3
        self.assertEqual(server.server_address[0], "127.0.0.1")

        def request(method, path, payload=None, headers=None):
            output = []
            def client():
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                try:
                    connection.request(method, path, payload, headers or {"Content-Type": "application/json"})
                    response = connection.getresponse()
                    body = response.read()
                    output.append((response.status, json.loads(body) if response.getheader("Content-Type").startswith("application/json") else body.decode()))
                except Exception as exc:
                    output.append(exc)
                finally:
                    connection.close()
            thread = threading.Thread(target=client, daemon=True)
            thread.start()
            server.handle_request()  # SQLite remains on its creating thread.
            thread.join(timeout=4)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(output), 1)
            if isinstance(output[0], Exception):
                raise output[0]
            return output[0]

        self.assertEqual(request("GET", "/health")[1]["serving_variant"], self.selected)
        self.assertEqual(request("GET", "/api/report")[1], self.report)
        self.assertIn("已知正例", request("GET", "/")[1])
        status, body = request("POST", "/api/search", json.dumps({"query": "headphones -Acme"}))
        self.assertEqual(status, 200)
        self.assertNotIn("a", body["serving"]["candidate_ids"])
        self.assertEqual(len(body["serving"]["ranking"]), len(body["serving"]["candidate_ids"]))
        for body in ("{", "[]", '{"query":"headphones","k":true}', '{"query":"headphones","pool":50}',
                     '{"query":"headphones","k":NaN}'):
            self.assertEqual(request("POST", "/api/search", body)[0], 400)
        # Reject a declared oversize body before waiting for or allocating it.
        self.assertEqual(request("POST", "/api/search", None,
                                 {"Content-Type": "application/json", "Content-Length": "8193"})[0], 400)
        self.assertEqual(request("POST", "/api/search", '{}', {"Content-Type": "text/plain"})[0], 400)
        self.assertEqual(request("POST", "/api/search", None, {"Transfer-Encoding": "chunked"})[0], 400)
        self.assertEqual(request("GET", "/missing")[0], 404)
        self.assertEqual(request("POST", "/missing", '{}')[0], 404)


if __name__ == "__main__":
    unittest.main()

"""The explorer must serve one bound experiment and compare identical candidates."""
import http.client
import json
from pathlib import Path
import shutil
import threading
import unittest
from uuid import uuid4

from ecom_search.data import file_sha256, save_json
from ecom_search.index import build_index, catalog_digest
from ecom_search.ranking import PairwiseRanker
from scripts.serve_public import PublicSearch, make_server

ROOT = Path(__file__).resolve().parents[1]


class PublicServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = (ROOT / "outputs/test-tmp" / uuid4().hex).resolve()
        self.directory.mkdir(parents=True)
        self.catalog = [
            {"id": "a", "title": "wireless headphones blue", "brand": "Acme", "description": ""},
            {"id": "b", "title": "wired headphones red", "brand": "Beta", "description": ""},
            {"id": "c", "title": "headphones wireless black", "brand": "Gamma", "description": ""},
        ]
        build_index(self.catalog, self.directory / "products.sqlite")
        save_json(self.directory / "model.json", {
            "ranker": PairwiseRanker().to_dict(), "catalog_sha256": catalog_digest(self.catalog),
            "selected_epochs": 0, "frozen_before_test": True, "protocol": "public_cpu_v1"})
        save_json(self.directory / "selection.json", {"selected_epochs": 0})
        save_json(self.directory / "train_val_identities.json", {"query_ids": [], "normalized_queries": []})
        self.freeze = {"protocol": "public_cpu_v1", "dataset": "synthetic-service-test",
                       "synthetic": True, "data_fingerprints": {"catalog.jsonl": "c" * 64},
                       "artifact_fingerprints": {name: file_sha256(self.directory / name) for name in
                           ("model.json", "products.sqlite", "selection.json", "train_val_identities.json")}}
        self.write_freeze_and_report()

    def write_freeze_and_report(self):
        save_json(self.directory / "freeze.json", self.freeze)
        self.report = {"protocol": "public_cpu_v1", "dataset": self.freeze["dataset"],
                       "synthetic": self.freeze["synthetic"], "catalog_products": len(self.catalog),
                       "freeze_sha256": file_sha256(self.directory / "freeze.json"),
                       "data_fingerprints": self.freeze["data_fingerprints"]}
        save_json(self.directory / "report.json", self.report)

    def tearDown(self):
        if not self.directory.is_relative_to((ROOT / "outputs/test-tmp").resolve()):
            raise RuntimeError("Unsafe test cleanup path")
        shutil.rmtree(self.directory)

    def test_current_metadata_candidate_identity_and_contributions(self):
        application = PublicSearch(self.directory)
        try:
            self.assertEqual(application.metadata["products"], 3)
            self.assertEqual(application.metadata["catalog_sha256"], catalog_digest(self.catalog))
            self.assertEqual(application.metadata["model_sha256"], file_sha256(self.directory / "model.json"))
            self.assertEqual(application.metadata["source"], "synthetic-service-test")
            self.assertIs(application.metadata["synthetic"], True)
            self.assertEqual(application.report, self.report)
            result = application.compare({"query": "wireless headphones", "k": 3, "pool": 3})
            self.assertEqual(result["baseline"]["ranking"], "retrieval_order")
            self.assertEqual(result["learned"]["ranking"], "trained_pairwise")
            self.assertEqual(result["baseline"]["candidate_count"], 3)
            self.assertEqual({p["id"] for p in result["baseline"]["products"]},
                             {p["id"] for p in result["learned"]["products"]})
            for product in result["learned"]["products"]:
                self.assertEqual(len(product["contributions"]), 12)
                self.assertAlmostEqual(sum(row["contribution"] for row in product["contributions"]), product["score"])
            excluded = application.compare({"query": "headphones without Acme", "k": 3, "pool": 3})
            for column in ("baseline", "learned"):
                self.assertNotIn("a", [p["id"] for p in excluded[column]["products"]])
        finally:
            application.close()

    def test_mismatched_model_catalog_is_rejected_even_if_refrozen(self):
        model_path = self.directory / "model.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["catalog_sha256"] = "f" * 64
        save_json(model_path, model)
        self.freeze["artifact_fingerprints"]["model.json"] = file_sha256(model_path)
        self.write_freeze_and_report()
        with self.assertRaises(ValueError):
            PublicSearch(self.directory)

    def test_report_from_another_frozen_experiment_is_rejected(self):
        for field, value in (("freeze_sha256", "f" * 64),
                             ("data_fingerprints", {"catalog.jsonl": "e" * 64}),
                             ("catalog_products", 99), ("dataset", "other-run"), ("synthetic", False)):
            with self.subTest(field=field):
                save_json(self.directory / "report.json", dict(self.report, **{field: value}))
                with self.assertRaises(ValueError):
                    PublicSearch(self.directory)

    def test_unfrozen_artifact_change_is_rejected(self):
        save_json(self.directory / "selection.json", {"selected_epochs": 30})
        with self.assertRaises(ValueError):
            PublicSearch(self.directory)

    def test_selection_can_be_explored_before_final_report(self):
        (self.directory / "report.json").unlink()
        application = PublicSearch(self.directory)
        try:
            self.assertIsNone(application.report)
            self.assertFalse(application.metadata["report_available"])
        finally:
            application.close()

    def test_actual_http_contract_in_sqlite_owner_thread(self):
        server, application = make_server(self.directory, port=0)
        server.timeout = 3

        def request(method, path, payload=None, content_type="application/json"):
            output = []

            def client():
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                try:
                    connection.request(method, path, payload, {"Content-Type": content_type})
                    response = connection.getresponse()
                    output.append((response.status, json.loads(response.read())))
                except Exception as error:
                    output.append(error)
                finally:
                    connection.close()

            thread = threading.Thread(target=client, daemon=True)
            thread.start()
            # SQLite objects are created and used in this same test thread.
            server.handle_request()
            thread.join(timeout=4)
            self.assertFalse(thread.is_alive(), "HTTP client did not complete")
            self.assertEqual(len(output), 1)
            if isinstance(output[0], Exception):
                raise output[0]
            return output[0]

        try:
            status, body = request("GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(body["products"], 3)
            status, body = request("POST", "/api/search", json.dumps({"query": "headphones", "k": 3, "pool": 3}))
            self.assertEqual(status, 200)
            self.assertEqual(body["learned"]["candidate_count"], 3)
            for payload in ("{", "[]", '{"query":"headphones","k":true}', '{"query":"headphones","pool":0}'):
                status, body = request("POST", "/api/search", payload)
                self.assertEqual(status, 400)
                self.assertIn("error", body)
            status, _ = request("POST", "/api/search", '{"query":"headphones"}', "text/plain")
            self.assertEqual(status, 400)
            status, _ = request("GET", "/unknown")
            self.assertEqual(status, 404)
        finally:
            server.server_close()
            application.close()


if __name__ == "__main__":
    unittest.main()

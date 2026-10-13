"""Protocol fixtures test safeguards, not ecommerce relevance claims."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ecom_search.data import file_sha256, save_json, save_jsonl
from scripts import benchmark_closure as b


class DecisionsTest(unittest.TestCase):
    def test_validation_tie_keeps_baseline_and_guard_filters_high_score(self):
        summary = {name: {b.PRIMARY: {"mean": .2}, b.COVERAGE: {"mean": .6}} for name in b.VARIANTS}
        summary["soft_judged"][b.PRIMARY]["mean"] = .9
        summary["soft_judged"][b.COVERAGE]["mean"] = .59
        chosen, eligible = b.select_variant(summary)
        self.assertEqual(chosen, b.BASELINE)
        self.assertNotIn("soft_judged", eligible)
        summary["soft_retrieved"][b.PRIMARY]["mean"] = .3
        self.assertEqual(b.select_variant(summary)[0], "soft_retrieved")

    def test_release_gate_is_conjunctive_and_conservative(self):
        self.assertTrue(b.release_decision("soft_bm25", {"ci95": [.001, .03]}, {"ci95": [-.01, .02]})["promoted"])
        for primary, coverage in (([0, .03], [0, .1]), ([.01, .03], [-.01001, .1]), (None, [0, .1])):
            result = b.release_decision("soft_bm25", {"ci95": primary}, {"ci95": coverage})
            self.assertFalse(result["promoted"])
            self.assertEqual(result["serving_variant"], b.BASELINE)
        self.assertFalse(b.release_decision(b.BASELINE, {"ci95": [.01, .02]}, {"ci95": [0, .1]})["promoted"])

    def test_unknown_is_not_a_negative_and_queries_without_pairs_stay(self):
        class FakeEngine:
            def retrieve(self, query, **kwargs):
                return [{"id": "unknown"}, {"id": "a"}, {"id": "b"}]
        records = [{"query": "camera", "candidate_ids": ["a", "b", "c"], "split": "train",
                    "qrels": {"a": 3, "b": 0, "c": 2}},
                   {"query": "shoe", "candidate_ids": ["c"], "split": "train", "qrels": {"c": 3}}]
        original = copy.deepcopy(records)
        judged, retrieved, coverage = b.training_records(records, FakeEngine())
        self.assertEqual(records, original)
        self.assertEqual(retrieved[0]["candidate_ids"], ["a", "b"])
        self.assertEqual(retrieved[1]["candidate_ids"], [])
        self.assertEqual(coverage["queries_without_pairs"], 1)
        self.assertEqual(coverage["unjudged_ignored"], 4)
        self.assertNotIn("unknown", retrieved[0]["qrels"])

    def test_candidate_coverage_is_not_reranked_top10_or_unknown_precision(self):
        result = {"ranking": [{"id": "unknown"}, {"id": "a"}], "candidate_ids": ["a", "b", "unknown"]}
        actual = b.metrics(result, {"a": 3, "b": 2, "c": 0})
        self.assertEqual(actual[b.PRIMARY], .5)
        self.assertEqual(actual[b.COVERAGE], 1.0)
        self.assertEqual(actual["judged_fraction@10"], .1)
        self.assertIsNone(b.metrics(result, {"c": 0})[b.PRIMARY])


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.output = self.root / "data", self.root / "run"
        products = [{"id": f"p{i:02}", "title": f"camera {i} plastic", "brand": "Acme", "description": ""} for i in range(20)]
        save_jsonl(self.data / "catalog.jsonl", products)
        for split, query in (("train", "camera 3"), ("val", "camera 5"), ("test", "camera 8")):
            save_jsonl(self.data / f"{split}.jsonl", [{"query_id": split, "query": query, "split": split,
                "candidate_ids": [p["id"] for p in products], "qrels": {p["id"]: i % 4 for i, p in enumerate(products)}}])
        save_json(self.data / "selection.json", {"fixture": True})
        save_json(self.data / "manifest.json", {"protocol": "closure-esci-v1", "synthetic": True,
            "selection_uses_esci_label": False, "source_files": {},
            "previous_experiment_exclusion": {"registry_content_sha256": "fixture",
                "selected_overlap_by_id": 0, "selected_overlap_by_normalized_query": 0},
            "files": {name: {"sha256": file_sha256(self.data / name), "bytes": (self.data / name).stat().st_size}
                      for name in b.old.DATA_FILES[:-1]}})

    def select(self):
        original = b.load_jsonl
        def no_test(path):
            if Path(path).name == "test.jsonl":
                raise AssertionError("test labels opened before freeze")
            return original(path)
        with patch.object(b, "load_jsonl", side_effect=no_test):
            b.prepare(self.data, self.output)
            b.train(self.data, self.output)
            b.select(self.data, self.output)

    def test_sealed_end_to_end_and_repeat_refused(self):
        self.select()
        frozen = b.read(self.output / "selection.json")
        self.assertFalse(frozen["test_labels_decoded"])
        original = b.load_jsonl
        def after_marker(path):
            if Path(path).name == "test.jsonl":
                self.assertTrue((self.output / "test_started.json").is_file())
            return original(path)
        with patch.object(b, "load_jsonl", side_effect=after_marker):
            report = b.final(self.data, self.output)
        self.assertEqual(report["queries"], 1)
        self.assertEqual(b.read(self.output / "release.json")["serving_variant"], report["decision"]["serving_variant"])
        with self.assertRaises(FileExistsError):
            b.final(self.data, self.output)
        with self.assertRaises(FileExistsError):
            b.prepare(self.data, self.output)

    def test_model_tampering_rejected_before_test_open(self):
        self.select()
        model = self.output / "models.json"
        model.write_text(model.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with patch.object(b, "load_jsonl", side_effect=AssertionError("must not open labels")):
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                b.final(self.data, self.output)
        self.assertFalse((self.output / "test_started.json").exists())

    def test_changed_selection_cannot_choose_another_valid_variant(self):
        self.select()
        selection = b.read(self.output / "selection.json")
        selection["selected"] = next(name for name in b.VARIANTS if name != selection["selected"])
        b.atomic_json(self.output / "selection.json", selection)
        with self.assertRaisesRegex(ValueError, "validation winner"):
            b.final(self.data, self.output)
        self.assertFalse((self.output / "test_started.json").exists())

    def test_changed_protocol_cannot_relabel_executed_gate(self):
        b.prepare(self.data, self.output)
        protocol = b.read(self.output / "protocol.json")
        protocol["promotion_rule"]["primary_ci95_lower_strictly_above"] = -.1
        b.atomic_json(self.output / "protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "release rules"):
            b.train(self.data, self.output)

    def test_old_data_is_rejected_before_training(self):
        manifest = b.read(self.data / "manifest.json")
        manifest["protocol"] = "public_cpu_v1"
        save_json(self.data / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "fresh-query"):
            b.prepare(self.data, self.output)

    def test_data_tamper_rejected_before_training(self):
        b.prepare(self.data, self.output)
        with (self.data / "test.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "data fingerprint"):
            b.train(self.data, self.output)
        self.assertFalse((self.output / "training_started.json").exists())


if __name__ == "__main__":
    unittest.main()

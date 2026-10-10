"""Protocol failures must fail closed; fixtures are synthetic, never evidence."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from ecom_search.cli import SearchEngine
from ecom_search.data import file_sha256, save_json, save_jsonl
from scripts import benchmark_public as benchmark


class PublicBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("outputs/test-tmp") / ("public-benchmark-" + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.data = cls.root / "data"
        catalog = [{"id": f"p{n:02d}", "title": f"Acme wireless camera {n} gb",
                    "brand": "Acme", "description": ""} for n in range(20)]
        save_jsonl(cls.data / "catalog.jsonl", catalog)
        for split, queries in {"train": ["camera 1 gb", "camera 2 gb"],
                               "val": ["Acme camera 3 gb"],
                               "test": ["Acme wireless camera 4 gb", "camera 5 gb -wired"]}.items():
            records = []
            for number, query in enumerate(queries):
                target = int(next(token for token in query.split() if token.isdigit()))
                qrels = {p["id"]: (3 if n == target else 2 if abs(n - target) == 1 else 0)
                         for n, p in enumerate(catalog)}
                records.append({"query_id": f"{split}-{number}", "query": query,
                                "split": split, "candidate_ids": list(qrels),
                                "qrels": qrels, "full_qrels": qrels})
            save_jsonl(cls.data / f"{split}.jsonl", records)
        cls._refresh_manifest(cls.data)
        cls.frozen = cls.root / "frozen"
        original = benchmark.load_jsonl

        def forbid_test(path):
            if Path(path).name == "test.jsonl":
                raise AssertionError("selection must never parse test labels")
            return original(path)

        with patch.object(benchmark, "load_jsonl", side_effect=forbid_test):
            benchmark.select(cls.data, cls.frozen)

    @staticmethod
    def _refresh_manifest(directory):
        save_json(directory / "manifest.json", {
            "dataset": "handwritten unit-test fixture", "synthetic": True,
            "files": {name: {"bytes": (directory / name).stat().st_size,
                             "sha256": file_sha256(directory / name)}
                      for name in benchmark.DATA_FILES[:-1]}})

    def setUp(self):
        self.case = self.root / uuid.uuid4().hex
        self.data_copy = self.case / "data"
        self.run = self.case / "run"
        shutil.copytree(self.data, self.data_copy)
        shutil.copytree(self.frozen, self.run)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    def test_selection_never_decodes_holdout_and_is_service_compatible(self):
        selection = json.loads((self.run / "selection.json").read_text())
        self.assertIs(selection["test_labels_decoded"], False)
        self.assertEqual(selection["grid"], [0, 5, 15, 30])
        engine = SearchEngine(self.run / "products.sqlite", self.run / "model.json")
        try:
            self.assertEqual(engine.search("camera 4 gb")["ranking"], "trained_pairwise")
        finally:
            engine.close()

    def test_final_metrics_and_live_ranking_match_and_repeat_is_rejected(self):
        report = benchmark.final(self.data_copy, self.run, per_query=True)
        rows = [json.loads(line) for line in (self.run / "per_query_test.jsonl").read_text().splitlines()]
        self.assertEqual(report["test_queries"], 2)
        self.assertEqual(report["fixed_pool"]["title_bm25"]["ndcg@10"]["queries"], 2)
        self.assertEqual(report["corpus_paired_delta"]["known_positive_recall@100"]["delta"], 0)
        engine = SearchEngine(self.run / "products.sqlite", self.run / "model.json")
        try:
            for row in rows:
                actual = [p["id"] for p in engine.search(row["query"], k=10, pool=100)["products"]]
                self.assertEqual(actual, row["rankings"]["reranked"][:10])
        finally:
            engine.close()
        with self.assertRaises(FileExistsError):
            benchmark.final(self.data_copy, self.run)

    def test_holdout_mutation_is_rejected_before_label_parse(self):
        with (self.data_copy / "test.jsonl").open("a") as handle:
            handle.write("\n")
        with patch.object(benchmark, "load_jsonl", side_effect=AssertionError("must not parse any labels")):
            with self.assertRaisesRegex(ValueError, "data fingerprint changed"):
                benchmark.final(self.data_copy, self.run)
        self.assertFalse((self.run / "final_started.json").exists())

    def test_model_tampering_and_unfinished_final_are_rejected(self):
        with (self.run / "model.json").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "artifact fingerprint changed"):
            benchmark.final(self.data_copy, self.run)
        shutil.copyfile(self.frozen / "model.json", self.run / "model.json")
        save_json(self.run / "final_started.json", {"simulated_interrupted_final": True})
        with self.assertRaises(FileExistsError):
            benchmark.final(self.data_copy, self.run)

    def test_source_change_is_rejected_before_labels_are_loaded(self):
        original = benchmark.file_sha256

        def changed(path):
            if Path(path).resolve() == (benchmark.ROOT / "ecom_search/ranking.py").resolve():
                return "0" * 64
            return original(path)

        with patch.object(benchmark, "file_sha256", side_effect=changed), \
                patch.object(benchmark, "load_jsonl", side_effect=AssertionError("must not parse labels")):
            with self.assertRaisesRegex(ValueError, "source fingerprint changed"):
                benchmark.final(self.data_copy, self.run)

    def test_normalized_query_overlap_cannot_reach_scoring(self):
        data = self.case / "overlap-data"
        shutil.copytree(self.data, data)
        records = benchmark.load_jsonl(data / "test.jsonl")
        records[0]["query"] = "  CAMERA   1 gb "
        save_jsonl(data / "test.jsonl", records)
        self._refresh_manifest(data)
        run = self.case / "overlap-run"
        benchmark.select(data, run)
        with self.assertRaisesRegex(ValueError, "holdout overlaps"):
            benchmark.final(data, run)

    def test_unknown_corpus_items_are_not_assigned_relevance_labels(self):
        metrics = benchmark._corpus_metrics(["unknown", "positive"], {"positive": 3, "irrelevant": 0})
        self.assertEqual(metrics["known_positive_recall@10"], 1)
        self.assertEqual(metrics["judged_fraction@10"], .1)
        self.assertNotIn("p@10_es", metrics)
        self.assertNotIn("ndcg@10", metrics)

    def test_bootstrap_is_paired_deterministic_and_handles_zero_denominator(self):
        result = benchmark.paired_bootstrap([0., None, .5], [.25, .9, .75], samples=100)
        self.assertEqual(result["queries"], 2)
        self.assertEqual(result["ci95"], [.25, .25])
        self.assertEqual(result, benchmark.paired_bootstrap([0., None, .5], [.25, .9, .75], samples=100))
        self.assertIsNone(benchmark.paired_bootstrap([None], [None], samples=100)["delta"])


if __name__ == "__main__":
    unittest.main()

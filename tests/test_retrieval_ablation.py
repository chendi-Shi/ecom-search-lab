"""Synthetic controls check the ablation intervention, not retrieval quality."""
from pathlib import Path
import json
import shutil
import unittest
from unittest.mock import patch
import uuid

from ecom_search.data import file_sha256, save_json, save_jsonl
from ecom_search.index import ProductIndex, build_index
from ecom_search.ranking import _tokens
from scripts import ablate_retrieval as ablation
from scripts import benchmark_public as benchmark


class RetrievalAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("outputs/test-tmp") / ("retrieval-ablation-" + uuid.uuid4().hex)
        cls.data, cls.reference = cls.root / "data", cls.root / "reference"
        cls.catalog = [{"id": f"p{n:02d}", "title": f"camera {n} gb " + ("wired" if n % 2 else "wireless"),
                        "brand": "Acme", "description": ""} for n in range(20)]
        save_jsonl(cls.data / "catalog.jsonl", cls.catalog)
        for split, queries in {"train": ["camera 1 gb", "camera 2 gb"], "val": ["camera 3 gb"],
                               "test": ["Acme camera 4 gb", "camera 5 gb -wired"]}.items():
            save_jsonl(cls.data / f"{split}.jsonl", [
                {"query_id": f"{split}-{i}", "query": query, "split": split,
                 "candidate_ids": [p["id"] for p in cls.catalog],
                 "qrels": {p["id"]: (3 if n == i else 2 if n % 3 == 0 else 0) for n, p in enumerate(cls.catalog)}}
                for i, query in enumerate(queries)])
        save_json(cls.data / "manifest.json", {"dataset": "synthetic test fixture", "synthetic": True,
            "files": {name: {"bytes": (cls.data / name).stat().st_size,
                             "sha256": file_sha256(cls.data / name)} for name in benchmark.DATA_FILES[:-1]}})
        benchmark.select(cls.data, cls.reference)
        benchmark.final(cls.data, cls.reference, per_query=True)
        cls.original = {p["id"]: p for p in cls.catalog}

    @classmethod
    def tearDownClass(cls):
        root = cls.root.resolve()
        assert root.is_relative_to(Path("outputs/test-tmp").resolve())
        shutil.rmtree(root)

    def new_run(self):
        return self.root / uuid.uuid4().hex

    def test_full_primitive_reproduces_live_index_with_constraints(self):
        with ProductIndex(self.reference / "products.sqlite", cache_size=0) as index:
            for query in ("camera", "camera -wired", "Acme 3 gb", "nothingmatches"):
                plan = ablation.make_plan(query, "full")
                self.assertEqual(ablation.retrieve(index, plan, 5, self.original), index.search(plan, 5))

    def test_brand_removed_only_from_retrieval_not_reranker_metadata(self):
        path = self.new_run() / "title.sqlite"
        build_index([dict(p, brand="") for p in self.catalog], path)
        with ProductIndex(path, cache_size=0) as index:
            self.assertEqual(ablation.retrieve(index, ablation.make_plan("Acme", "full"), 100, self.original), [])
            hits = ablation.retrieve(index, ablation.make_plan("camera", "full"), 100, self.original)
            self.assertEqual(len(hits), 20)
            self.assertTrue(all(p["brand"] == "Acme" for p in hits))
            self.assertEqual(ablation.retrieve(index, ablation.make_plan("camera -Acme", "full"), 100, self.original), [])

    def test_raw_parser_keeps_same_tokenizer_and_removes_exclusion(self):
        query = "a camera without wired 12.5 gb"
        raw = ablation.make_plan(query, "raw")
        self.assertEqual(raw.positive_terms, tuple(dict.fromkeys(_tokens(query))))
        self.assertEqual(raw.excluded_terms, ())
        self.assertIn("wired", ablation.make_plan(query, "full").excluded_terms)
        self.assertEqual(ablation.make_plan(query, "no_exclusion").excluded_terms, ())

    def test_candidate_recall_uses_all_returned_items_and_ignores_unjudged(self):
        ids = [f"x{i}" for i in range(150)]
        result = ablation.metrics(ids, {"x0": 3, "x149": 2, "missing": 2})
        self.assertEqual(result["candidate_known_positive_recall"], 2/3)
        self.assertEqual(result["known_positive_recall@100"], 1/3)
        self.assertIsNone(ablation.metrics(ids, {"x0": 0})["candidate_known_positive_recall"])

    def test_prepare_does_not_decode_test_and_evaluate_locks_and_reproduces(self):
        output = self.new_run()
        original = ablation.load_jsonl
        def no_test(path):
            if Path(path).name == "test.jsonl":
                raise AssertionError("prepare must not decode test")
            return original(path)
        with patch.object(ablation, "load_jsonl", side_effect=no_test):
            ablation.prepare(self.data, self.reference, output)
        report = ablation.evaluate(self.data, self.reference, output)
        self.assertEqual(report["baseline_reproduction"]["queries"], 2)
        self.assertTrue(report["baseline_reproduction"]["exact_historical_rankings_checked"])
        self.assertEqual(report["exclusion_subset"]["queries"], 1)
        with self.assertRaises(FileExistsError):
            ablation.evaluate(self.data, self.reference, output)

    def test_index_tampering_rejected_before_test_is_read(self):
        output = self.new_run()
        ablation.prepare(self.data, self.reference, output)
        with (output / "title_only.sqlite").open("ab") as handle:
            handle.write(b"tampered")
        with patch.object(ablation, "load_jsonl", side_effect=AssertionError("test was parsed")):
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                ablation.evaluate(self.data, self.reference, output)
        self.assertFalse((output / "evaluation_started.json").exists())


if __name__ == "__main__":
    unittest.main()

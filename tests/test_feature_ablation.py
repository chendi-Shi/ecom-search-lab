"""Synthetic fixtures test the diagnosis protocol, never establish accuracy."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from ecom_search.data import file_sha256, save_json, save_jsonl
from ecom_search.ranking import FEATURE_NAMES, FeatureExtractor, PairwiseRanker
from scripts import ablate_features as ablation
from scripts import benchmark_public as benchmark


class FeatureAblationUnitTest(unittest.TestCase):
    def setUp(self):
        self.products = [
            {"id": "a", "title": "Acme wireless camera 64 gb without plastic", "brand": "Acme", "description": "metal camera"},
            {"id": "b", "title": "Other camera 32 gb plastic case", "brand": "Other", "description": "wired camera"},
            {"id": "c", "title": "Acme camera 64 gb", "brand": "Acme", "description": ""},
        ]
        self.records = [{"query": "Acme camera 64 gb without plastic", "candidate_ids": ["a", "b", "c"],
                         "split": "train", "qrels": {"a": 3, "b": 0, "c": 2}}]
        self.extractor = FeatureExtractor(self.products)

    def test_cache_is_label_free_exact_and_never_mutates_cached_vectors(self):
        class ForbidLabels(dict):
            def __getitem__(self, key):
                if key in ("qrels", "split"):
                    raise AssertionError("cache must be label free")
                return super().__getitem__(key)
        cache = ablation.cache_features([ForbidLabels(self.records[0])], self.extractor)
        before = json.dumps(list(cache.items()), sort_keys=True)
        query = benchmark._ranking_query(self.records[0]["query"])
        for name, dropped in ablation.variants().items():
            wrapped = ablation.CachedExtractor(self.extractor.products, cache, dropped)
            for product in self.products:
                direct = self.extractor.features(query, product)
                actual = wrapped.features(query, product)
                self.assertEqual(actual, [0.0 if i in dropped else x for i, x in enumerate(direct)], name)
                actual[0] = 12345
        self.assertEqual(before, json.dumps(list(cache.items()), sort_keys=True))
        with self.assertRaises(KeyError):
            ablation.CachedExtractor(self.extractor.products, cache).features("unseen query", self.products[0])

    def test_cached_and_direct_masked_training_are_identical(self):
        cache = ablation.cache_features(self.records, self.extractor)
        normalized = [dict(r, query=benchmark._ranking_query(r["query"])) for r in self.records]
        base = self.extractor
        for dropped in ablation.variants().values():
            class DirectMask:
                names = list(FEATURE_NAMES)
                products = base.products

                def features(self, query, product):
                    return ablation.mask_vector(base.features(query, product), dropped)
            direct, cached = PairwiseRanker(), PairwiseRanker()
            result_direct = direct.fit(normalized, DirectMask(), epochs=4, seed=42)
            result_cached = cached.fit(normalized, ablation.CachedExtractor(base.products, cache, dropped), epochs=4, seed=42)
            self.assertEqual(result_direct, result_cached)
            self.assertEqual(direct.weights, cached.weights)

    def test_partition_and_activity_distinguish_constant_nonzero_from_rank_signal(self):
        self.assertEqual(sorted(i for group in ablation.GROUPS.values() for i in group), list(range(12)))
        constant = [1.0] + [0.0] * 11
        varying = list(constant)
        varying[4] = 1.0
        result = ablation.activity([[constant, list(constant)], [constant, varying]])
        self.assertEqual(result["groups"]["title_bm25"]["nonzero_queries"], 2)
        self.assertEqual(result["groups"]["title_bm25"]["varying_queries"], 0)
        self.assertEqual(result["groups"]["brand"]["varying_queries"], 1)
        self.assertEqual(result["features"]["description_token_coverage"]["nonzero_pairs"], 0)

    def test_paired_delta_direction_and_support(self):
        rows = [{"fixed_pool": {"full": {"ndcg@10": .8}, "without_brand": {"ndcg@10": .5}},
                 "varying_groups": {"fixed_pool": {"brand": True}}},
                {"fixed_pool": {"full": {"ndcg@10": .4}, "without_brand": {"ndcg@10": .4}},
                 "varying_groups": {"fixed_pool": {"brand": False}}}]
        self.assertAlmostEqual(ablation.delta_rows(rows, "fixed_pool", "without_brand", "ndcg@10")["delta"], .15)
        active = ablation.delta_rows(rows, "fixed_pool", "without_brand", "ndcg@10", group="brand")
        self.assertEqual(active["queries"], 1)
        self.assertAlmostEqual(active["delta"], .3)


class FeatureAblationProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("outputs/test-tmp") / ("feature-ablation-" + uuid.uuid4().hex)
        cls.data, cls.reference = cls.root / "data", cls.root / "reference"
        products = [{"id": f"p{i:02}", "title": f"Acme camera {i} gb wireless",
                     "brand": "Acme" if i % 2 == 0 else "Other", "description": ""} for i in range(20)]
        save_jsonl(cls.data / "catalog.jsonl", products)
        for split, queries in (("train", ["camera 4 gb", "camera wireless"]),
                               ("val", ["camera 5 gb"]), ("test", ["Acme camera 6 gb", "camera without wired"])):
            rows = [{"query_id": f"{split}-{i}", "query": query, "split": split,
                     "candidate_ids": [p["id"] for p in products],
                     "qrels": {p["id"]: 3 if j == 4 else 2 if j == 5 else 0 for j, p in enumerate(products)}}
                    for i, query in enumerate(queries)]
            save_jsonl(cls.data / (split + ".jsonl"), rows)
        save_json(cls.data / "manifest.json", {"dataset": "synthetic protocol test", "synthetic": True,
                  "files": {name: {"bytes": (cls.data / name).stat().st_size, "sha256": file_sha256(cls.data / name)}
                            for name in benchmark.DATA_FILES[:-1]}})
        with patch.object(benchmark, "EPOCH_GRID", (30,)):
            benchmark.select(cls.data, cls.reference)
        benchmark.final(cls.data, cls.reference)

    @classmethod
    def tearDownClass(cls):
        resolved = cls.root.resolve()
        resolved.relative_to(Path("outputs/test-tmp").resolve())
        shutil.rmtree(resolved)

    def setUp(self):
        self.output = self.root / uuid.uuid4().hex

    def test_frozen_pipeline_reproduces_and_test_is_not_opened_during_training(self):
        original_load = ablation.load_jsonl

        def forbid_test(path):
            if Path(path).name == "test.jsonl":
                raise AssertionError("test must not be opened before model freeze")
            return original_load(path)
        with patch.object(ablation, "load_jsonl", side_effect=forbid_test):
            protocol = ablation.prepare(self.data, self.reference, self.output)
            training = ablation.train(self.data, self.reference, self.output)
        self.assertFalse(protocol["test_labels_decoded_in_this_run"])
        self.assertIn("retrospective", protocol["study_type"])
        self.assertTrue(training["reproduction"]["passed"])
        self.assertEqual(len(training["cost"]), 8)
        self.assertEqual(len({v["pairs"] for v in training["cost"].values()}), 1)
        report = ablation.evaluate(self.data, self.reference, self.output)
        self.assertTrue(report["reference_reproduction"]["passed"])
        self.assertEqual(len(report["metrics"]["fixed_pool"]), 15)
        for method in report["metrics"]["corpus"].values():
            self.assertEqual(method["known_positive_recall@100"], report["metrics"]["corpus"]["full"]["known_positive_recall@100"])
        with self.assertRaises(FileExistsError):
            ablation.evaluate(self.data, self.reference, self.output)
        with self.assertRaises(FileExistsError):
            ablation.prepare(self.data, self.reference, self.output)

    def test_changed_reference_fails_before_any_dataset_parse(self):
        reference = self.root / ("changed-reference-" + uuid.uuid4().hex)
        shutil.copytree(self.reference, reference)
        ablation.prepare(self.data, reference, self.output)
        with (reference / "model.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with patch.object(ablation, "load_jsonl", side_effect=AssertionError("must not parse")):
            with self.assertRaisesRegex(ValueError, "reference fingerprint changed"):
                ablation.train(self.data, reference, self.output)

    def test_reproduction_failure_blocks_freeze(self):
        ablation.prepare(self.data, self.reference, self.output)
        with patch.object(ablation, "TOLERANCE", -1):
            with self.assertRaisesRegex(ValueError, "failed reproduction"):
                ablation.train(self.data, self.reference, self.output)
        self.assertFalse((self.output / "models_frozen.json").exists())
        self.assertFalse((self.output / "evaluation_started.json").exists())

    def test_trained_model_tamper_rejected_before_test(self):
        ablation.prepare(self.data, self.reference, self.output)
        ablation.train(self.data, self.reference, self.output)
        with (self.output / "models.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with patch.object(ablation, "load_jsonl", side_effect=AssertionError("must not parse")):
            with self.assertRaisesRegex(ValueError, "trained artifact fingerprint changed"):
                ablation.evaluate(self.data, self.reference, self.output)
        self.assertFalse((self.output / "evaluation_started.json").exists())


if __name__ == "__main__":
    unittest.main()

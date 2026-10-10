import importlib.util
import math
from pathlib import Path
import unittest

from ecom_search.accuracy import (ACCURACY_FEATURES, ACCURACY_SCHEMA_VERSION,
    CompactFeatureExtractor, TreeEnsemble, model_scores, complete_pool_features, build_pool_features)
from ecom_search.ranking import FeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("accuracy_experiment", ROOT / "scripts/train_accuracy.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class AccuracyContracts(unittest.TestCase):
    def test_compact_features_equal_full_catalog_statistics(self):
        catalog = [
            {"id": "a", "title": "Coleman Tumbler 20 oz stainless", "brand": "Coleman"},
            {"id": "b", "title": "YETI Tumbler 20 oz sugar free", "brand": "YETI"},
            {"id": "c", "title": "Nike women's shoes 42", "brand": "Nike"},
            {"id": "d", "title": "16 GB memory", "brand": ""},
        ]
        full, compact = FeatureExtractor(catalog), CompactFeatureExtractor(catalog)
        for query in ("coleman tumbler 20 oz", "without nike shoes", "sugar free tumbler", "32gb memory", ""):
            for product in catalog:
                self.assertEqual(full.features(query, product), compact.features(query, product))
        self.assertEqual(compact._stats_by_id.cache_info().maxsize, 512)

    def test_exported_numeric_tree_boundary_and_addition(self):
        ensemble = TreeEnsemble([
            {"feature": 12, "threshold": 0.5, "left": {"value": -2}, "right": {"value": 3}},
            {"value": 0.25}])
        rows = [[0] * 12 + [v, 0, 0] for v in (.49, .5, .51)]
        self.assertEqual(ensemble.predict(rows), [-1.75, -1.75, 3.25])

    def test_malformed_tree_and_nonfinite_feature_rejected(self):
        for trees in ([], [{"value": math.nan}], [{"feature": len(ACCURACY_FEATURES), "threshold": 0, "left": {"value": 0}, "right": {"value": 1}}]):
            with self.assertRaises(ValueError):
                TreeEnsemble(trees)
        ensemble = TreeEnsemble([{"value": 1}])
        for row in ([0] * 13, [0] * 14 + [float("nan")]):
            with self.assertRaises(ValueError):
                ensemble.predict([row])

    def test_same_pool_metrics_and_stable_ties(self):
        rows = [{"query_id": "q", "query": "lamp", "candidate_ids": ["z", "a"], "qrels": {"z": 0, "a": 3}}]
        result = experiment.metrics(rows, [1, 1])
        self.assertEqual(result["per_query"][0]["top10"], ["a", "z"])
        self.assertEqual(result["summary"]["ndcg10"], 1)
        self.assertEqual(result["summary"]["precision10"], .5)
        self.assertEqual(result["summary"]["exact_at1"], 1)
        with self.assertRaises(ValueError):
            experiment.metrics(rows, [1])
        with self.assertRaises(ValueError):
            experiment.metrics(rows, [1, math.inf])

    def test_ce_baseline_is_raw_logit(self):
        model = {"schema_version": ACCURACY_SCHEMA_VERSION, "kind": "cross_encoder", "feature_names": ACCURACY_FEATURES}
        self.assertEqual(model_scores(model, [[0] * 12 + [-5, 0, -1], [0] * 12 + [2, 1, 1]]), [-5, 2])
        with self.assertRaises(ValueError):
            model_scores(dict(model, schema_version=1), [])
        with self.assertRaises(ValueError):
            model_scores(dict(model, feature_names=list(reversed(ACCURACY_FEATURES))), [])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy not installed")
    def test_complete_pool_relative_features_and_order(self):
        rows = [[i] * 12 + [logit] for i, logit in enumerate([3., 1., 1., 7.])]
        actual = complete_pool_features(rows)
        self.assertEqual([r[:13] for r in actual], rows)
        self.assertEqual([r[13] for r in actual], [2/3, 1/6, 1/6, 1.])
        self.assertAlmostEqual(sum(r[14] for r in actual), 0.)
        self.assertAlmostEqual(sum(r[14] ** 2 for r in actual) / 4, 1.)
        reverse = complete_pool_features(rows[::-1])[::-1]
        for left, right in zip(actual, reverse):
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, b)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy not installed")
    def test_relative_features_empty_singleton_ties_and_invalid(self):
        self.assertEqual(complete_pool_features([]), [])
        self.assertEqual(complete_pool_features([[0.] * 12 + [3.]])[0][13:], [0., 0.])
        tied = complete_pool_features([[0.] * 12 + [3.]] * 3)
        self.assertEqual([r[13:] for r in tied], [[.5, 0.]] * 3)
        for bad in ([[0.] * 15], [[0.] * 12 + [math.nan]]):
            with self.assertRaises(ValueError):
                complete_pool_features(bad)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy not installed")
    def test_shared_pool_builder_preserves_products_and_checks_alignment(self):
        from unittest.mock import Mock
        extractor = Mock()
        extractor.features.return_value = [0.] * 12
        products = [{"id": "b"}, {"id": "a"}]
        built = build_pool_features("raw query", products, extractor, [5., -2.])
        self.assertEqual([r[12] for r in built], [5., -2.])
        self.assertEqual([r[13] for r in built], [1., 0.])
        self.assertEqual([call.args for call in extractor.features.call_args_list],
                         [("raw query", products[0]), ("raw query", products[1])])
        with self.assertRaises(ValueError):
            build_pool_features("q", products, extractor, [1.])
        with self.assertRaises(ValueError):
            build_pool_features("q", [products[0]] * 2, extractor, [1., 2.])

    def test_metric_denominators_exclude_zero_ideal_only(self):
        rows = [{"query_id": "none", "query": "none", "candidate_ids": ["a"], "qrels": {"a": 0}},
                {"query_id": "yes", "query": "yes", "candidate_ids": ["b"], "qrels": {"b": 3}}]
        result = experiment.metrics(rows, [0, 0])["summary"]
        self.assertEqual(result["ndcg10_denominator"], 1)
        self.assertEqual(result["precision10_denominator"], 2)
        self.assertEqual(result["precision10"], .5)

    def test_bootstrap_requires_query_pair_alignment(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("optional numpy not installed")
        a = [{"query_id": "a", "ndcg10": .8}, {"query_id": "b", "ndcg10": .5}]
        b = [{"query_id": "a", "ndcg10": .7}, {"query_id": "b", "ndcg10": .4}]
        result = experiment.paired_interval(a, b, repetitions=100)
        self.assertAlmostEqual(result["delta"], .1)
        self.assertTrue(all(abs(v - .1) < 1e-12 for v in result["ci95"]))
        with self.assertRaises(ValueError):
            experiment.paired_interval(a, b[::-1], repetitions=10)


if __name__ == "__main__":
    unittest.main()

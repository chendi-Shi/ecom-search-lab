"""Serving contracts use fake artifacts, no models, sockets, or held-out data."""
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ecom_search.cli import SearchEngine, main
from ecom_search.index import catalog_digest
from ecom_search.server import health_payload, search_request


CATALOG = [dict(id="a", title="Nike running shoes", brand="Nike", description=""),
           dict(id="b", title="Running shoes", brand="Other", description="")]


class MemoryIndex:
    def __init__(self):
        self.metadata = {"products": "2", "catalog_sha256": catalog_digest(CATALOG)}
        self.closed = False
        self.lexical_calls = 0

    def catalog(self):
        return [dict(row) for row in CATALOG]

    def get_products(self, ids):
        return {row["id"]: dict(row) for row in CATALOG if row["id"] in ids}

    def search(self, plan, k):
        self.lexical_calls += 1
        return [dict(row, retrieval_score=1 / (i + 1)) for i, row in enumerate(CATALOG[:k])]

    def close(self):
        self.closed = True


class FrozenPool:
    def __init__(self):
        self.metadata = {"catalog_sha256": catalog_digest(CATALOG)}
        self.selection = {"mode": "lexical_only"}
        self.calls = []
        self.hits = [("a", .02), ("b", .01)]
        self.closed = False

    def search(self, query, k=100, use_vector=True):
        self.calls.append((query, k, use_vector))
        return self.hits[:k]

    def close(self):
        self.closed = True


class CPUScorer:
    metadata = {"kind": "lambdamart"}

    def __init__(self):
        self.calls = []
        self.scores = {"a": -1.0, "b": 2.0}
        self.closed = False

    def score(self, query, products):
        self.calls.append((query, [p["id"] for p in products]))
        return {pid: value for pid, value in self.scores.items() if pid in {p["id"] for p in products}}

    def close(self):
        self.closed = True


class AccuracyServingTests(unittest.TestCase):
    def setUp(self):
        self.index, self.pool, self.scorer = MemoryIndex(), FrozenPool(), CPUScorer()
        self.pool_constructor = Mock(return_value=self.pool)
        self.compact_constructor = Mock(return_value=SimpleNamespace(products={p["id"]: p for p in CATALOG}))
        self.accuracy_constructor = Mock(return_value=self.scorer)
        self.patches = [patch("ecom_search.cli.ProductIndex", return_value=self.index),
            patch.dict("sys.modules", {
                "scripts.prepare_accuracy_retrieval": SimpleNamespace(RetrievalPool=self.pool_constructor),
                "ecom_search.accuracy": SimpleNamespace(AccuracyRanker=self.accuracy_constructor,
                                                        CompactFeatureExtractor=self.compact_constructor)})]
        for item in self.patches:
            item.start()
        self.engine = None

    def tearDown(self):
        if self.engine is not None:
            self.engine.close()
        for item in reversed(self.patches):
            item.stop()

    def create(self):
        self.engine = SearchEngine(Path("fixture.sqlite"), accuracy_path=Path("outputs/fixture/model.json"),
                                   strategy_path=Path("outputs/fixture/retrieval/pool_strategy.json"))
        return self.engine

    def test_default_precision_uses_only_frozen_candidates_and_raw_query(self):
        engine = self.create()
        query = "running shoes without Nike"
        result = engine.search(query, k=2)
        self.assertEqual(self.pool.calls, [(query, 100, True)])
        self.assertEqual(self.index.lexical_calls, 0)
        self.assertEqual(self.scorer.calls, [(query, ["a", "b"])])
        self.assertEqual([p["id"] for p in result["products"]], ["b", "a"])
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["ranking"], "cpu_lambdamart")
        self.assertTrue(result["accuracy"]["applied"])
        self.assertEqual(result["retrieval"], "fts_bm25")
        self.assertEqual(result["candidate_source"], "词法召回（BM25）")
        self.compact_constructor.assert_called_once_with(CATALOG)
        self.pool_constructor.assert_called_once_with(Path("outputs/fixture/retrieval"),
            strategy_path=Path("outputs/fixture/retrieval/pool_strategy.json"))

    def test_fast_switch_preserves_candidates_and_skips_cpu_scorer(self):
        result = search_request(self.create(), {"query": "running shoes", "k": 2, "use_accuracy": False})
        self.assertEqual([p["id"] for p in result["products"]], ["a", "b"])
        self.assertEqual(self.scorer.calls, [])
        self.assertFalse(result["accuracy"]["applied"])
        self.assertEqual(result["ranking"], "retrieval_order")

    def test_api_default_and_boolean_contract(self):
        engine = self.create()
        self.assertTrue(search_request(engine, {"query": "shoes"})["accuracy"]["applied"])
        for value in (None, "false", 0, 1, []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                search_request(engine, {"query": "shoes", "use_accuracy": value})

    def test_pool_guard_and_generative_guard_run_before_model(self):
        engine = self.create()
        with self.assertRaisesRegex(ValueError, "at most 200"):
            engine.search("shoes", pool=201)
        with self.assertRaisesRegex(ValueError, "frozen accuracy retrieval"):
            engine.search("shoes", use_generative=True)
        with self.assertRaisesRegex(ValueError, "boolean"):
            engine.search("shoes", use_accuracy="true")
        self.assertEqual(self.pool.calls, [])
        self.assertEqual(self.scorer.calls, [])
        self.assertEqual(engine.search("shoes", pool=200)["candidate_count"], 2)

    def test_unknown_duplicates_and_bad_model_scores_are_rejected(self):
        engine = self.create()
        self.pool.hits = [("missing", .1)]
        with self.assertRaisesRegex(ValueError, "unknown product IDs"):
            engine.search("shoes")
        self.pool.hits = [("a", .1), ("a", .05)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            engine.search("shoes")
        self.pool.hits = [("a", .1), ("b", .05)]
        self.pool.hits[0] = ("a", float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite scores"):
            engine.search("shoes", use_accuracy=False)
        self.pool.hits = [("a", .1), ("b", .05)]
        self.scorer.scores = {"a": 1.0}
        with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
            engine.search("shoes")
        self.scorer.scores = {"a": float("nan"), "b": 1.0}
        with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
            engine.search("shoes")

    def test_mismatched_strategy_closes_failed_constructor(self):
        self.pool.metadata["catalog_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "catalog fingerprints differ"):
            self.create()
        self.assertTrue(self.pool.closed)
        self.assertTrue(self.index.closed)
        self.accuracy_constructor.assert_not_called()

    def test_health_flags_and_no_model_legacy_default(self):
        engine = self.create()
        health = health_payload(engine, multimodal=object(), beauty=object())
        self.assertTrue(health["accuracy"])
        self.assertTrue(health["accuracy_default"])
        self.assertEqual(health["accuracy_kind"], "lambdamart")
        self.assertEqual(health["accuracy_pool_limit"], 200)
        self.assertFalse(health["generative"])
        self.assertEqual(health["retrieval_mode"], "lexical_only")
        self.assertEqual(health["candidate_source"], "词法召回（BM25）")
        self.assertFalse(health["disk"])
        self.assertTrue(health["multimodal"] and health["beauty"])
        engine.close()
        self.engine = SearchEngine(Path("fixture.sqlite"))
        legacy = self.engine.search("shoes")
        self.assertFalse(legacy["accuracy"]["applied"])
        with self.assertRaisesRegex(ValueError, "not configured"):
            self.engine.search("shoes", use_accuracy=True)
        self.assertFalse(health_payload(self.engine)["accuracy"])

    def test_cli_routes_accuracy_paths_and_fast_flag(self):
        fake = Mock()
        fake.search.return_value = {"products": []}
        with patch("ecom_search.cli.SearchEngine", return_value=fake) as constructor, redirect_stdout(io.StringIO()):
            main(["search", "shoes", "--accuracy-model", "model.json", "--retrieval-strategy", "pool_strategy.json", "--no-accuracy"])
        self.assertEqual(constructor.call_args.kwargs["accuracy_path"], Path("model.json"))
        self.assertEqual(constructor.call_args.kwargs["strategy_path"], Path("pool_strategy.json"))
        self.assertFalse(fake.search.call_args.kwargs["use_accuracy"])
        fake.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

import math
from pathlib import Path
import tempfile
import unittest

from ecom_search.closure import ClosureEngine, query_plan, ranking_query
from ecom_search.data import save_jsonl
from ecom_search.index import ProductIndex, build_index
from ecom_search.query import plan_query
from ecom_search.ranking import FEATURE_NAMES


class ClosureEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog = [
            {"id": "a", "title": "Sugar snack", "brand": "Acme"},
            {"id": "b", "title": "Sugar free snack", "brand": "Acme"},
            {"id": "c", "title": "Milk snack", "brand": "Acme"},
            {"id": "d", "title": "Plain snack", "brand": "Acme"},
        ]
        self.catalog_path = self.root / "catalog.jsonl"
        self.index_path = self.root / "index.sqlite"
        save_jsonl(self.catalog_path, self.catalog)
        build_index(self.catalog, self.index_path)

    def tearDown(self):
        self.temporary.cleanup()

    def engine(self, models=None):
        return ClosureEngine(self.catalog_path, self.index_path, models)

    def test_soft_only_removes_natural_hard_exclusion(self):
        with self.engine() as engine:
            legacy = engine.retrieve("snack without sugar", "legacy")
            soft = engine.retrieve("snack without sugar", "soft")
        self.assertEqual({hit["id"] for hit in legacy}, {"b", "c", "d"})
        self.assertEqual({hit["id"] for hit in soft}, {"a", "b", "c", "d"})
        original = query_plan("snack without sugar", "legacy")
        changed = query_plan("snack without sugar", "soft")
        self.assertEqual(original.positive_terms, changed.positive_terms)
        self.assertEqual(original.ranking_query, changed.ranking_query)
        self.assertEqual(changed.excluded_terms, ())

    def test_explicit_minus_stays_hard_and_keeps_negated_catalog_terms(self):
        with self.engine() as engine:
            explicit = engine.retrieve("snack -sugar", "soft")
            mixed = engine.retrieve("snack without milk -sugar", "soft")
        self.assertEqual({hit["id"] for hit in explicit}, {"b", "c", "d"})
        self.assertEqual({hit["id"] for hit in mixed}, {"b", "c", "d"})
        self.assertEqual(query_plan("snack -ＳＵＧＡＲ -sugar", "soft").excluded_terms,
                         ("sugar",))
        self.assertEqual(query_plan("snack -1.5", "soft").excluded_terms, ("1.5",))
        self.assertEqual(query_plan("snack sugar-free", "soft").excluded_terms, ())

    def test_legacy_matches_existing_index(self):
        with self.engine() as engine, ProductIndex(self.index_path) as legacy:
            for query in ("snack", "snack without sugar", "snack -milk", "Acme snack"):
                self.assertEqual(engine.retrieve(query, "legacy", 3), legacy.search(query, 3))
                self.assertEqual(query_plan(query, "legacy"), plan_query(query))

    def test_empty_positive_queries_remain_empty_and_invalid_input_fails(self):
        with self.engine() as engine:
            for query in ("the", "without sugar", "-sugar", "!!!"):
                response = engine.search(query, "soft", pool=10)
                self.assertEqual(response["candidate_ids"], [])
                self.assertEqual(response["ranking"], [])
                self.assertEqual(response["results"], [])
            for query in (None, 123, "", " ", "a" * 2001):
                with self.assertRaises(ValueError):
                    engine.retrieve(query)
                with self.assertRaises(ValueError):
                    ranking_query(query)

    def test_search_and_offline_ranking_share_exact_scores_and_order(self):
        weights = [0.0] * len(FEATURE_NAMES)
        weights[FEATURE_NAMES.index("negation_conflict")] = -2.0
        with self.engine({"judged": weights, "retrieved": weights}) as engine:
            weights[-2] = 1000.0  # Constructor takes a snapshot.
            query = "snack without sugar"
            hits = engine.retrieve(query, "soft", 4)
            expected = engine.rank(query, hits, "judged")
            response = engine.search(query, "soft", "judged", pool=4, k=2)
            self.assertEqual(response["ranking"], expected)
            self.assertEqual(response["candidate_ids"], [hit["id"] for hit in hits])
            self.assertEqual([row["id"] for row in response["results"]],
                             [row["id"] for row in expected[:2]])
            self.assertEqual(expected[-1], {"id": "a", "score": -2.0})
            self.assertEqual(engine.rank(query, hits, "retrieved"), expected)
            self.assertEqual(engine.rank(query, hits, "bm25"),
                             [{"id": hit["id"], "score": hit["retrieval_score"]} for hit in hits])

    def test_tie_order_and_hits_contract(self):
        with self.engine() as engine:
            hits = [{"id": "b", "retrieval_score": 1.0},
                    {"id": "a", "retrieval_score": 1.0}]
            self.assertEqual([row["id"] for row in engine.rank("snack", hits)], ["a", "b"])
            for invalid in ([{"id": "x", "retrieval_score": 1}], hits + hits,
                            [{"id": "a", "retrieval_score": math.nan}],
                            [{"id": "a", "retrieval_score": True}]):
                with self.assertRaises(ValueError):
                    engine.rank("snack", invalid)

    def test_policy_ranker_and_limit_validation(self):
        with self.engine() as engine:
            for limit in (0, 1001, 1.5, True):
                with self.assertRaises(ValueError):
                    engine.retrieve("snack", pool=limit)
            for k in (0, 11, True):
                with self.assertRaises(ValueError):
                    engine.search("snack", pool=10, k=k)
            with self.assertRaises(ValueError):
                engine.retrieve("snack", policy="raw")
            for ranker in ("other", "judged", "retrieved"):
                with self.assertRaises(ValueError):
                    engine.rank("snack", [], ranker)

    def test_model_and_catalog_binding_rejects_mismatch(self):
        for models in ({"bm25": [0.0] * 12}, {"judged": [0.0] * 11},
                       {"retrieved": [math.inf] * 12}, {"judged": "bad"}):
            with self.assertRaises(ValueError):
                self.engine(models)
        changed = [dict(row) for row in self.catalog]
        changed[0]["title"] = "changed"
        save_jsonl(self.catalog_path, changed)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.engine()


if __name__ == "__main__":
    unittest.main()

"""Optional vector correctness tests: normalization, invalid inputs and persistence."""
import importlib.util
from contextlib import contextmanager
import json
import shutil
import unittest
from pathlib import Path
import uuid

from ecom_search.vectors import VectorIndex, build, exact_cosine_search, normalize_vectors, model_fingerprint, _evaluate, _query_rows


@contextmanager
def workspace_temporary_directory():
    root = Path(__file__).resolve().parents[1]
    target = (root / (".vector-test-" + uuid.uuid4().hex)).resolve()
    if not target.is_relative_to(root):
        raise ValueError("temporary directory escaped project")
    # Windows sandbox ACLs cannot reopen Python's mode-0700 TemporaryDirectory.
    target.mkdir()
    try:
        yield str(target)
    finally:
        shutil.rmtree(target)


class RetrievalMetricTests(unittest.TestCase):
    def test_fixed_candidate_judgments_cannot_claim_corpus_recall(self):
        with workspace_temporary_directory() as directory:
            query_path = Path(directory) / "test.jsonl"
            query_path.write_text(json.dumps({"query_id": "q1", "query": "test", "candidate_ids": ["a"], "qrels": {"a": 3}}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _query_rows(query_path)

    def test_complements_excluded_and_unknowns_are_not_negatives(self):
        class FakeIndex:
            def search_vector(self, query, k):
                return [("exact", 1), ("complement", .9), ("unjudged", .8)]
        report = _evaluate(FakeIndex(), [{"qrels": {"exact": 3, "substitute": 2, "complement": 1}}], [[1, 0]],
                           [["exact", "complement", "unjudged"]], repeats=1)
        self.assertEqual(report["known_positive_recall_at_100"], .5)
        self.assertAlmostEqual(report["judged_coverage_at_100"], 2 / 3)
        self.assertEqual(report["neighbor_recall_at_10"], 1)
        self.assertEqual(report["queries_with_known_positives"], 1)

    def test_no_known_positive_is_not_scored_as_perfect_recall(self):
        class FakeIndex:
            def search_vector(self, query, k):
                return []
        report = _evaluate(FakeIndex(), [{"qrels": {"complement": 1}}], [[1, 0]], [[]], repeats=1)
        self.assertIsNone(report["known_positive_recall_at_100"])
        self.assertIsNone(report["judged_coverage_at_100"])
        self.assertIsNone(report["neighbor_recall_at_10"])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy unavailable")
class VectorTests(unittest.TestCase):
    def test_exact_is_cosine_not_vector_magnitude(self):
        result = exact_cosine_search(["large", "aligned", "opposite"], [[100, 100], [2, 0], [-2, 0]], [3, 0], 10)
        self.assertEqual([row[0] for row in result], ["aligned", "large", "opposite"])
        self.assertAlmostEqual(result[0][1], 1)
        self.assertAlmostEqual(result[1][1], 2 ** -.5, places=6)
        self.assertAlmostEqual(result[2][1], -1)

    def test_boundary_k_empty_and_ties(self):
        self.assertEqual(exact_cosine_search([], __import__("numpy").empty((0, 2)), [1, 0], 5), [])
        self.assertEqual(exact_cosine_search(["a"], [[1, 0]], [1, 0], 0), [])
        self.assertEqual(exact_cosine_search(["b", "a"], [[1, 0], [1, 0]], [1, 0], 5), [("a", 1.0), ("b", 1.0)])
        with self.assertRaises(ValueError):
            exact_cosine_search(["a"], [[1, 0]], [1, 0], -1)
        with self.assertRaises(ValueError):
            exact_cosine_search(["a"], [[1, 0]], [1, 0, 0], 1)

    def test_rejects_duplicate_nonfinite_and_zero(self):
        with self.assertRaises(ValueError):
            exact_cosine_search(["a", "a"], [[1, 0], [0, 1]], [1, 0], 1)
        with self.assertRaises(ValueError):
            exact_cosine_search([None], [[1, 0]], [1, 0], 1)
        for matrix in ([[float("nan"), 1]], [[float("inf"), 1]], [[0, 0]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                normalize_vectors(matrix)

    def test_model_identifier_never_triggers_download(self):
        with self.assertRaises(ValueError):
            model_fingerprint("sentence-transformers/paraphrase-MiniLM-L3-v2")

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "optional FAISS unavailable")
    def test_persistence_matches_reference_and_checks_integrity(self):
        vectors = [[1, 0], [1, 1], [-1, 0]]
        with workspace_temporary_directory() as directory:
            report = build(directory, ["a", "b", "c"], vectors)
            self.assertEqual(report["count"], 3)
            index = VectorIndex(directory)
            actual = index.search_vector([9, 0], 100)
            reference = exact_cosine_search(["a", "b", "c"], vectors, [9, 0], 100)
            self.assertEqual([row[0] for row in actual], [row[0] for row in reference])
            for observed, expected in zip(actual, reference):
                self.assertAlmostEqual(observed[1], expected[1], places=6)
            self.assertEqual(index.search_vector([1, 0], 0), [])
            (Path(directory) / "ids.json").write_text('["tampered"]', encoding="utf-8")
            with self.assertRaises(ValueError):
                VectorIndex(directory)

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "optional FAISS unavailable")
    def test_empty_index_and_invalid_build(self):
        import numpy as np
        with workspace_temporary_directory() as directory:
            build(directory, [], np.empty((0, 2)))
            index = VectorIndex(directory)
            self.assertEqual(index.search_vector([1, 0], 10), [])
            self.assertEqual(index.search("anything", 10), [])
            with self.assertRaises(ValueError):
                index.search_vector([1, 0], -1)
            with self.assertRaises(ValueError):
                build(directory, ["a", "a"], [[1, 0], [0, 1]])
            with self.assertRaises(ValueError):
                build(directory, ["a"], [[float("nan"), 0]])


if __name__ == "__main__":
    unittest.main()

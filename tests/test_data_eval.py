"""Data leakage, independent metric values, and held-out selection checks."""
import copy
import json
import math
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

from ecom_search.data import file_sha256, load_dataset, save_json, save_jsonl, validate_dataset
from ecom_search.evaluate import evaluate_ranking, ndcg, recall, train_and_evaluate

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    catalog = [{"id": "a", "title": "wireless blue headphones", "brand": "", "description": ""},
               {"id": "b", "title": "wired red headphones", "brand": "", "description": ""}]
    splits = {split: [{"query_id": f"q-{split}", "query": f"{split} wireless headphones",
                       "candidate_ids": ["a", "b"], "qrels": {"a": 3, "b": 0}, "split": split}]
              for split in ("train", "val", "test")}
    return catalog, splits


class MetricTests(unittest.TestCase):
    def test_graded_gain_and_cutoff(self):
        # Reversing grades 3/1 yields gain 1+7/log2(3), not a binary DCG.
        expected = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg(["low", "high"], {"low": 1, "high": 3}, 2), expected)
        self.assertEqual(ndcg(["high", "low"], {"low": 1, "high": 3}, 1), 1.0)
        self.assertEqual(recall(["low", "high"], {"low": 1, "high": 3}, 1), 0.0)

    def test_no_relevant_query_has_separate_denominators(self):
        records = [
            {"query_id": "one", "query": "one two three four five", "candidate_ids": ["a", "b"], "qrels": {"a": 3, "b": 0}},
            {"query_id": "zero", "query": "zero", "candidate_ids": ["a", "b"], "qrels": {"a": 0, "b": 0}},
            {"query_id": "complement", "query": "complement", "candidate_ids": ["a", "b"], "qrels": {"a": 1, "b": 0}},
        ]
        result = evaluate_ranking(records, lambda row: row["candidate_ids"], k=1)
        self.assertEqual(result["metrics"]["ndcg@1"], 1.0)
        self.assertEqual(result["metrics"]["ndcg@1_denominator"], 2)
        self.assertEqual(result["metrics"]["candidate_recall@1_denominator"], 1)
        self.assertEqual(result["slices"]["long_query_ge_5_tokens"]["queries"], 1)

    def test_invalid_ranker_cannot_improve_by_omitting_candidates(self):
        row = {"query_id": "q", "query": "q", "candidate_ids": ["a", "b"], "qrels": {"a": 3, "b": 0}}
        for ranking in (["a"], ["a", "a"], ["a", "outside"]):
            with self.subTest(ranking=ranking), self.assertRaises(ValueError):
                evaluate_ranking([row], lambda _: ranking)


class DataTests(unittest.TestCase):
    def test_rejects_duplicates_leakage_and_invalid_judgments(self):
        catalog, splits = fixture()
        invalid = copy.deepcopy(splits)
        invalid["val"][0]["query"] = "  ＴＲＡＩＮ   wireless HEADPHONES "
        with self.assertRaises(ValueError):
            validate_dataset(catalog, invalid)
        invalid = copy.deepcopy(splits)
        invalid["test"][0]["query_id"] = invalid["train"][0]["query_id"]
        with self.assertRaises(ValueError):
            validate_dataset(catalog, invalid)
        with self.assertRaises(ValueError):
            validate_dataset(catalog + [catalog[0]], splits)
        for grade in (-1, 4, True, 2.5):
            invalid = copy.deepcopy(splits)
            invalid["train"][0]["qrels"]["a"] = grade
            with self.subTest(grade=grade), self.assertRaises(ValueError):
                validate_dataset(catalog, invalid)
        invalid = copy.deepcopy(splits)
        invalid["train"][0]["candidate_ids"][1] = "unknown"
        with self.assertRaises(ValueError):
            validate_dataset(catalog, invalid)


class FileAndSelectionTests(unittest.TestCase):
    def setUp(self):
        self.work = (ROOT / "outputs/test-tmp" / uuid4().hex).resolve()
        self.work.mkdir(parents=True)

    def tearDown(self):
        if not self.work.is_relative_to((ROOT / "outputs/test-tmp").resolve()):
            raise RuntimeError("Unsafe cleanup path")
        shutil.rmtree(self.work)

    def write_fixture(self, root, test_grade=3):
        catalog, splits = fixture()
        splits["test"][0]["qrels"] = {"a": test_grade, "b": 3 - test_grade}
        save_jsonl(root / "catalog.jsonl", catalog)
        for split, rows in splits.items():
            save_jsonl(root / f"{split}.jsonl", rows)
        return catalog

    def test_saved_data_bytes_are_checked(self):
        root = self.work / "data"
        self.write_fixture(root)
        path = root / "train.jsonl"
        save_json(root / "manifest.json", {"files": {"train.jsonl": {
            "sha256": file_sha256(path), "bytes": path.stat().st_size}}})
        self.assertEqual(len(load_dataset(root)[0]), 2)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_dataset(root)

    def test_test_labels_do_not_select_epochs_or_weights(self):
        left, right = self.work / "left-data", self.work / "right-data"
        self.write_fixture(left, 3)
        self.write_fixture(right, 0)
        first = train_and_evaluate(left, self.work / "first-output")
        second = train_and_evaluate(right, self.work / "second-output")
        self.assertEqual(first["selected_epochs"], second["selected_epochs"])
        self.assertEqual(first["model_weights"], second["model_weights"])
        frozen = json.loads((self.work / "first-output/model.json").read_text(encoding="utf-8"))
        self.assertTrue(frozen["frozen_before_test"])
        self.assertIn("ranker", frozen)
        self.assertEqual(len(frozen["catalog_sha256"]), 64)
        with self.assertRaises(FileExistsError):
            train_and_evaluate(left, self.work / "first-output")


if __name__ == "__main__":
    unittest.main()

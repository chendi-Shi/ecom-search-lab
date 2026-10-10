"""Strict event time, user isolation, cold start and seeded BPR checks."""
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

from ecom_search.personalization import (
    BPRModel, BeautyRecommender, PersonalizationStore, ProfileRanker, chronological_split,
    evaluate_preferences, sampled_cases,
)
from ecom_search.data import file_sha256, save_json, save_jsonl

ROOT = Path(__file__).resolve().parents[1]


class LocalProfileTests(unittest.TestCase):
    def setUp(self):
        self.work = (ROOT / "outputs/test-tmp" / uuid4().hex).resolve()
        self.work.mkdir(parents=True)
        self.store = PersonalizationStore(self.work / "profile.sqlite")
        self.catalog = [{"id": "a", "title": "wireless headphones", "brand": "", "description": ""},
                        {"id": "b", "title": "garden fertilizer", "brand": "", "description": ""}]

    def tearDown(self):
        self.store.close()
        if not self.work.is_relative_to((ROOT / "outputs/test-tmp").resolve()):
            raise RuntimeError("Unsafe cleanup path")
        shutil.rmtree(self.work)

    def test_isolation_strict_time_and_latest_hide(self):
        self.store.record("one", "a", "like", 100)
        self.store.record("one", "a", "hide", 200)
        self.store.record("one", "a", "history", 250)
        self.store.record("two", "b", "like", 50)
        self.assertEqual(self.store.events("one", 100), [])
        self.assertEqual(self.store.hidden("one", 200), set())
        self.assertEqual(self.store.hidden("one", 201), {"a"})
        self.assertEqual(self.store.profile("two")["liked_ids"], ["b"])
        self.store.clear("one")
        self.assertEqual(self.store.events("one"), [])
        self.assertEqual(len(self.store.events("two")), 1)

    def test_profile_uses_only_prior_likes_and_zero_cold_start(self):
        ranker = ProfileRanker(self.store, self.catalog, vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]})
        self.assertEqual(ranker.score("headphones", self.catalog, "new", 100), {"a": 0.0, "b": 0.0})
        self.store.record("one", "a", "like", 100)
        self.store.record("one", "b", "like", 300)
        scores = ranker.score("query", self.catalog, "one", 200)
        self.assertGreater(scores["a"], scores["b"])
        self.assertEqual(scores["b"], 0.0)

    def test_rejects_invalid_explicit_signals(self):
        for timestamp in (True, float("nan"), -1):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.store.record("one", "a", "like", timestamp)
        with self.assertRaises(ValueError):
            self.store.record("one", "a", "click")

    def test_stale_preference_decays_even_with_one_liked_item(self):
        self.store.record("one", "a", "like", 100)
        ranker = ProfileRanker(self.store, self.catalog, vectors={"a": [10.0, 0.0], "b": [0.0, 1.0]})
        fresh = ranker.score("query", self.catalog, "one", 101)["a"]
        stale = ranker.score("query", self.catalog, "one", 100 + 60 * 86400)["a"]
        self.assertLessEqual(fresh, 1.0)
        self.assertAlmostEqual(stale / fresh, 0.25, places=4)

    def test_beauty_serving_reads_only_bound_train_history_and_rejects_model_change(self):
        train = [{"user_id": "u", "product_id": "a", "rating": 5, "timestamp": 1, "split": "train"},
                 {"user_id": "other", "product_id": "b", "rating": 5, "timestamp": 2, "split": "train"}]
        model = BPRModel()
        model.fit(train, epochs=5)
        directory = self.work / "beauty"
        save_jsonl(directory / "catalog.jsonl", self.catalog)
        save_jsonl(directory / "train.jsonl", train)
        fingerprints = {name: {"sha256": file_sha256(directory / name), "bytes": (directory / name).stat().st_size}
                        for name in ("catalog.jsonl", "train.jsonl")}
        save_json(directory / "model.json", {"model": model.to_dict(), "selection": {"epochs": 5, "personal_weight": 0.25},
                  "data_fingerprints": fingerprints, "source": {"dataset": "synthetic test fixture"},
                  "cutoffs": {"train_end_unix_seconds_exclusive": 3}, "frozen_before_test": True})
        fingerprints["model.json"] = {"sha256": file_sha256(directory / "model.json"), "bytes": (directory / "model.json").stat().st_size}
        save_json(directory / "manifest.json", {"files": fingerprints})
        # A malformed future file must not be read or used to suppress candidate b.
        (directory / "test.jsonl").write_text("future file is deliberately not valid JSON", encoding="utf-8")
        serving = BeautyRecommender(directory)
        self.assertEqual([row["id"] for row in serving.recommend("u")], ["b"])
        self.assertTrue(serving.recommend("cold")[0]["cold_user"])
        with (directory / "model.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaises(ValueError):
            BeautyRecommender(directory)


class OfflinePreferenceTests(unittest.TestCase):
    def test_equal_timestamps_never_cross_global_boundary(self):
        events = [{"user_id": "u", "product_id": str(index), "rating": 5, "timestamp": index // 2}
                  for index in range(30)]
        splits, cutoffs = chronological_split(events)
        self.assertLess(max(row["timestamp"] for row in splits["train"]), min(row["timestamp"] for row in splits["val"]))
        self.assertLess(max(row["timestamp"] for row in splits["val"]), min(row["timestamp"] for row in splits["test"]))

    def test_target_sampling_never_uses_same_timestamp_or_future_history(self):
        target = {"user_id": "u", "product_id": "target", "rating": 5, "timestamp": 200}
        all_events = [{"user_id": "u", "product_id": "old", "rating": 5, "timestamp": 100}, target,
                      {"user_id": "u", "product_id": "same_time", "rating": 5, "timestamp": 200},
                      {"user_id": "u", "product_id": "future", "rating": 5, "timestamp": 300}]
        cases, counts = sampled_cases([target], all_events, {"old", "target", "same_time", "future", "other"}, candidate_count=5)
        self.assertEqual(cases[0]["history_events"], 1)
        self.assertEqual(cases[0]["history_max_timestamp"], 100)
        self.assertNotIn("old", cases[0]["candidate_ids"])
        self.assertIn("future", cases[0]["candidate_ids"])
        repeated = sampled_cases([dict(target, product_id="old")], all_events, {"old", "target"})[0]
        self.assertEqual(repeated, [])

    def test_train_contract_seed_roundtrip_and_cold_user_fallback(self):
        rows = [{"user_id": "u", "product_id": "a", "rating": 5, "timestamp": 1, "split": "train"},
                {"user_id": "other", "product_id": "b", "rating": 5, "timestamp": 2, "split": "train"}]
        with self.assertRaises(ValueError):
            BPRModel().fit([dict(rows[0], split="test")])
        first, second = BPRModel(), BPRModel()
        first.fit(rows, epochs=5)
        second.fit(rows, epochs=5)
        self.assertEqual(first.to_dict(), second.to_dict())
        restored = BPRModel.from_dict(first.to_dict())
        self.assertEqual(restored.score("u", "a"), first.score("u", "a"))
        cases = [{"user_id": "cold", "target_id": "a", "candidate_ids": ["b", "a"], "timestamp": 3}]
        baseline = evaluate_preferences(cases, first, 0.0)
        personal = evaluate_preferences(cases, first, 1.0)
        self.assertEqual(baseline["metrics"], personal["metrics"])
        self.assertEqual(personal["cold_users"]["events"], 1)


if __name__ == "__main__":
    unittest.main()

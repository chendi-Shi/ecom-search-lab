import copy
import math
import unittest

from ecom_search.ranking import FeatureExtractor, PairwiseRanker, analyze_lexical_query


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [
            {"id": "correct", "title": "Acme black phone 64 GB", "brand": "Acme", "description": "without red coating"},
            {"id": "wrong", "title": "Other red phone 128GB", "brand": "Other", "description": "red coating"},
            {"id": "unknown", "title": "phone", "brand": "", "description": ""},
        ]
        self.extractor = FeatureExtractor(self.catalog)

    def as_named(self, query, product_id):
        return dict(zip(self.extractor.names, self.extractor.features(query, self.extractor.products[product_id])))

    def test_numeric_and_negated_attributes(self):
        query = "Acme phone 64gb not red"
        correct = self.as_named(query, "correct")
        wrong = self.as_named(query, "wrong")
        unknown = self.as_named(query, "unknown")
        self.assertEqual(correct["number_match"], 1)
        self.assertEqual(wrong["number_conflict"], 1)
        self.assertEqual(correct["brand_match"], 1)
        self.assertEqual(wrong["brand_conflict"], 1)
        self.assertEqual(correct["negation_conflict"], 0)
        self.assertEqual(correct["negation_satisfied"], 1)
        self.assertEqual(wrong["negation_conflict"], 1)
        self.assertEqual(unknown["number_conflict"], 0)
        self.assertEqual(unknown["brand_conflict"], 0)
        for product in self.catalog:
            self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in self.extractor.features(query, product)))

    def test_public_query_analysis_matches_existing_feature_rules(self):
        self.assertEqual(analyze_lexical_query("shoes no red shoes"),
                         {"positive_terms": ("shoes",), "excluded_terms": ("red",)})
        self.assertEqual(analyze_lexical_query("snack sugar-free"),
                         {"positive_terms": ("snack",), "excluded_terms": ("sugar",)})
        before = self.extractor.features("case not for iphone", self.catalog[0])
        self.assertEqual(analyze_lexical_query("case not for iphone")["excluded_terms"], ("iphone",))
        self.assertEqual(before, self.extractor.features("case not for iphone", self.catalog[0]))

    def test_title_only_baseline_and_typo(self):
        self.assertEqual(self.extractor.bm25("coating", self.catalog[0]), 0)
        typo = self.as_named("phnoe", "unknown")
        self.assertEqual(typo["title_token_coverage"], 0)
        self.assertGreater(typo["char_trigram_similarity"], 0)
        self.assertTrue(all(x == 0 for x in self.extractor.features("", self.catalog[2])))

    def test_decimal_number_canonicalization_and_excluded_numbers(self):
        extractor = FeatureExtractor([{"id": "zero", "title": "phone 0 GB"}])
        features = dict(zip(extractor.names, extractor.features("phone 000.00gb", extractor.products["zero"])))
        self.assertEqual(features["number_match"], 1)
        features = dict(zip(self.extractor.names, self.extractor.features("phone not 128gb", self.catalog[1])))
        self.assertEqual(features["number_match"], 0)
        self.assertEqual(features["number_conflict"], 0)
        extractor = FeatureExtractor([{"id": "screen", "title": "phone 6 in screen"}])
        features = dict(zip(extractor.names, extractor.features("6inch phone", extractor.products["screen"])))
        self.assertEqual(features["number_match"], 1)

    def test_gradient_moves_in_pairwise_direction(self):
        query = "phone"
        model = PairwiseRanker([0.0] * len(self.extractor.names))
        record = {"query": query, "candidate_ids": ["correct", "wrong"], "qrels": {"correct": 3, "wrong": 0}}
        positive = self.extractor.features(query, self.catalog[0])
        negative = self.extractor.features(query, self.catalog[1])
        expected = [0.05 * 0.5 * (a - b) for a, b in zip(positive, negative)]
        result = model.fit([record], self.extractor, epochs=1, learning_rate=0.05, l2=0)
        for actual, target in zip(model.weights, expected):
            self.assertAlmostEqual(actual, target)
        self.assertGreater(model.score(positive), model.score(negative))
        self.assertLess(result["final_loss"], result["initial_loss"])

    def test_training_learns_attribute_conflicts(self):
        query = "Acme phone 64gb not red"
        records = [{"query": query, "split": "train", "candidate_ids": ["correct", "wrong"],
                    "qrels": {"correct": 3, "wrong": 0}}]
        first = PairwiseRanker()
        result = first.fit(records, self.extractor, epochs=30)
        second = PairwiseRanker()
        second.fit(records, self.extractor, epochs=30)
        self.assertEqual(first.weights, second.weights)
        self.assertLess(first.weights[self.extractor.names.index("brand_conflict")], 0)
        self.assertLess(first.weights[self.extractor.names.index("number_conflict")], 0)
        self.assertLess(first.weights[self.extractor.names.index("negation_conflict")], 0)
        self.assertLess(result["final_loss"], result["initial_loss"])
        self.assertEqual(first.to_dict(), PairwiseRanker.from_dict(first.to_dict()).to_dict())

    def test_equal_grades_and_unjudged_candidates_do_not_train(self):
        model = PairwiseRanker()
        before = model.weights[:]
        result = model.fit([{"query": "phone", "candidate_ids": ["correct", "wrong", "unknown"],
                             "qrels": {"correct": 1, "wrong": 1}}], self.extractor)
        self.assertEqual(result["pairs"], 0)
        self.assertEqual(model.weights, before)
        # A labelled product outside candidate_ids must not create a pair.
        result = model.fit([{"query": "phone", "candidate_ids": ["correct", "unknown"],
                             "qrels": {"correct": 3, "wrong": 0}}], self.extractor)
        self.assertEqual(result["pairs"], 0)

    def test_duplicate_products_and_nonfinite_models_rejected(self):
        with self.assertRaises(ValueError):
            FeatureExtractor([self.catalog[0], self.catalog[0]])
        with self.assertRaises(ValueError):
            FeatureExtractor([{"id": 1}, {"id": "1"}])
        for bad in (math.nan, math.inf, -math.inf):
            weights = [0.0] * len(self.extractor.names)
            weights[0] = bad
            with self.assertRaises(ValueError):
                PairwiseRanker(weights)
        obj = PairwiseRanker().to_dict()
        changed = copy.deepcopy(obj)
        changed["feature_names"] = list(reversed(changed["feature_names"]))
        with self.assertRaises(ValueError):
            PairwiseRanker.from_dict(changed)
        for name in ("feature_version", "model_version"):
            changed = copy.deepcopy(obj)
            changed[name] = True
            with self.assertRaises(ValueError):
                PairwiseRanker.from_dict(changed)
        changed = copy.deepcopy(obj)
        changed["weights"][0] = math.nan
        with self.assertRaises(ValueError):
            PairwiseRanker.from_dict(changed)

    def test_held_out_training_rejected(self):
        for split in ("test", "val", "validation"):
            with self.assertRaises(ValueError):
                PairwiseRanker().fit([{"query": "phone", "split": split, "candidate_ids": ["correct", "wrong"],
                                      "qrels": {"correct": 3, "wrong": 0}}], self.extractor)

    def test_pair_sampling_limit(self):
        catalog = [{"id": str(i), "title": f"item {i}"} for i in range(40)]
        extractor = FeatureExtractor(catalog)
        record = {"query": "item", "candidate_ids": [p["id"] for p in catalog],
                  "qrels": {p["id"]: int(p["id"]) % 4 for p in catalog}}
        model = PairwiseRanker()
        before = model.weights[:]
        result = model.fit([record], extractor, epochs=0)
        self.assertEqual(result["pairs"], 100)
        self.assertEqual(model.weights, before)
        result = model.fit([record, record], extractor, epochs=0)
        self.assertEqual(result["pairs"], 100)
        self.assertEqual(result["queries_with_pairs"], 1)


if __name__ == "__main__":
    unittest.main()

"""Statistical alignment and cache integrity contracts for neural ablations."""
import importlib.util
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import uuid
from types import SimpleNamespace

from scripts import ablate_neural as experiment

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
ROOT = Path(__file__).resolve().parents[1]


class NeuralAblationProtocol(unittest.TestCase):
    def test_no_ce_really_removes_all_neural_information(self):
        self.assertEqual(experiment.VARIANTS["without_ce_all"], list(range(12)))
        self.assertEqual(experiment.VARIANTS["without_pool_stats"], list(range(13)))
        self.assertNotIn(12, experiment.VARIANTS["without_ce_raw"])
        self.assertTrue({13, 14}.issubset(experiment.VARIANTS["without_ce_raw"]))
        self.assertEqual(len(set(map(tuple, experiment.VARIANTS.values()))), len(experiment.VARIANTS))

    @unittest.skipUnless(HAS_NUMPY, "optional numpy not installed")
    def test_paired_intervals_keep_query_pairing_and_correction(self):
        left = [{"query_id": str(i), "ndcg10": v} for i, v in enumerate([.9, .6, .3, .8])]
        right = [{"query_id": str(i), "ndcg10": v} for i, v in enumerate([.6, .7, .2, .5])]
        result = experiment.paired_contrast(left, right, "ndcg10", family_tests=13, repetitions=2000)
        self.assertAlmostEqual(result["delta"], .15)
        self.assertLessEqual(result["familywise_ci"][0], result["ci95"][0])
        self.assertGreaterEqual(result["familywise_ci"][1], result["ci95"][1])
        self.assertEqual((result["wins"], result["losses"]), (3, 1))
        with self.assertRaisesRegex(ValueError, "identity/order"):
            experiment.paired_contrast(left, right[::-1], "ndcg10", repetitions=10)
        self.assertEqual(experiment.paired_contrast(left, left, "ndcg10", repetitions=100)["ci95"], [0., 0.])

    @unittest.skipUnless(HAS_NUMPY, "optional numpy not installed")
    def test_feature_activity_distinguishes_constant_from_rankable(self):
        import numpy as np
        rows = np.zeros((4, 15))
        rows[:, 10] = 1.  # Nonzero but no within-pool variation.
        rows[0, 4] = 1.
        records = [{"candidate_ids": ["a", "b"]}, {"candidate_ids": ["c", "d"]}]
        result = experiment.feature_activity(records, rows)
        self.assertEqual(result["negation"]["nonzero_queries"], 2)
        self.assertEqual(result["negation"]["varying_queries"], 0)
        self.assertEqual(result["brand"]["varying_queries"], 1)
        self.assertEqual(result["brand"]["nonzero_pairs"], 1)

    @unittest.skipUnless(HAS_NUMPY, "optional numpy not installed")
    def test_cache_rejects_reorder_corruption_and_wrong_pool_statistics(self):
        import numpy as np
        work = ROOT / "outputs/test-tmp" / uuid.uuid4().hex
        work.mkdir(parents=True)
        records = [{"query_id": "q", "query": "test", "candidate_ids": ["a", "b"], "qrels": {"a": 3, "b": 0}}]
        products = [{"id": "a", "title": "A", "brand": ""}, {"id": "b", "title": "B", "brand": ""}]
        extractor = Mock(products={p["id"]: p for p in products})
        ce = Mock(fingerprint="frozen-test-config")
        binding = {"records_sha256": experiment.fingerprint(records),
                   "catalog_sha256": experiment.catalog_digest(products), "ce": ce.fingerprint,
                   "features": experiment.ACCURACY_FEATURES,
                   "ranking_code": experiment.file_sha256(ROOT / "ecom_search/ranking.py"),
                   "accuracy_code": experiment.file_sha256(ROOT / "ecom_search/accuracy.py")}
        rows = np.asarray(experiment.complete_pool_features([[0.] * 12 + [1.], [0.] * 12 + [3.]]))
        array, meta = work / "features-test.npy", work / "features-test.json"

        def write_cache():
            np.save(array, rows)
            meta.write_text(json.dumps({"binding": binding, "key": experiment.fingerprint(binding),
                                       "sha256": experiment.file_sha256(array)}), encoding="utf-8")
        try:
            write_cache()
            np.testing.assert_array_equal(experiment.verified_features(records, extractor, ce, work, "test", work), rows)
            changed = [dict(records[0], candidate_ids=["b", "a"])]
            with self.assertRaisesRegex(ValueError, "binding"):
                experiment.verified_features(changed, extractor, ce, work, "test", work)
            with array.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                experiment.verified_features(records, extractor, ce, work, "test", work)
            rows[0, 13] = .2
            write_cache()  # Valid bytes but invalid full-pool semantics.
            with self.assertRaisesRegex(ValueError, "pool-relative"):
                experiment.verified_features(records, extractor, ce, work, "test", work)
        finally:
            array.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            work.rmdir()

    @unittest.skipUnless(HAS_NUMPY, "optional numpy not installed")
    def test_test_records_are_not_read_until_every_model_is_fit_and_locked(self):
        import numpy as np

        class StopAtTestRead(Exception):
            pass

        train = [{"query_id": f"train-{i}", "query": "q", "candidate_ids": ["p"], "qrels": {"p": 3}}
                 for i in range(500)]
        ordered = sorted(train, key=lambda r: experiment.fingerprint([experiment.SEED, "accuracy_train", r["query_id"]]))
        val = [dict(train[0], query_id=f"val-{i}") for i in range(200)]
        fitted, locked = [], []
        lgb = Mock(__version__="4.7.0")
        lgb.train.side_effect = lambda *a, **kw: (fitted.append(True) or Mock())
        protocol = {"lightgbm_version": "4.7.0", "encoder_fingerprint": "frozen"}

        def records(path):
            if path.name == "test.jsonl":
                self.assertEqual(len(fitted), len(experiment.VARIANTS))
                self.assertEqual(locked, [True])
                raise StopAtTestRead
            return {"catalog.jsonl": [{"id": "p", "title": "P"}], "train.jsonl": train, "val.jsonl": val}[path.name]

        def metadata(path):
            if path.parent.name == "fake-reference":
                return {"train_query_ids": [r["query_id"] for r in ordered]} if path.name == "protocol.json" else {}
            return dict(protocol, frozen_at="test")

        args = SimpleNamespace(phase="run", data="fake-data", reference="fake-reference",
                               output="outputs/fake-neural-test", ce_model="fake-model",
                               feature_cache=None, lightgbm_site=None)
        with ExitStack() as stack:
            mocks = {"protocol_binding": Mock(return_value=protocol), "read_json": Mock(side_effect=metadata),
                     "load_lightgbm": Mock(return_value=lgb), "load_jsonl": Mock(side_effect=records),
                     "validate_dataset": Mock(), "CompactFeatureExtractor": Mock(),
                     "LocalCrossEncoder": Mock(return_value=Mock(fingerprint="frozen")),
                     "verified_features": Mock(side_effect=lambda records, *a: np.zeros((len(records), 15))),
                     "file_sha256": Mock(return_value="sha"),
                     "save_json": Mock(side_effect=lambda path, obj: locked.append(obj["before_test_read"]))}
            for name, value in mocks.items():
                stack.enter_context(patch.object(experiment, name, value))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises(StopAtTestRead):
                experiment.run(args)


if __name__ == "__main__":
    unittest.main()

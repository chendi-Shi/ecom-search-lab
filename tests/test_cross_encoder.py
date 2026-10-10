"""Content/cache/ordering contracts, without model download or optional imports."""
from contextlib import contextmanager, closing
import json
from pathlib import Path
import shutil
import sqlite3
import unittest
from unittest import mock
import uuid

from ecom_search.cross_encoder import LocalCrossEncoder, product_text, model_fingerprint, MODEL_ID, MODEL_REVISION, MODEL_FILES, _key, _sha, _save


@contextmanager
def fixture():
    root = Path(__file__).resolve().parents[1]
    directory = (root / (".cross-encoder-test-" + uuid.uuid4().hex)).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("temporary fixture escaped workspace")
    directory.mkdir()
    try:
        model = directory / "model"
        model.mkdir()
        names = MODEL_FILES + ("README.md",)
        for name in names:
            (model / name).write_bytes(("unit-test artifact, not real pretrained data: " + name).encode())
        records = [{"path": name, "bytes": (model / name).stat().st_size, "sha256": _sha(model / name)} for name in names]
        manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": "apache-2.0", "files": records,
                    "artifact_fingerprint": _key({"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": records})}
        _save(model / "model_manifest.json", manifest)
        # The production constant always pins official weights. Fixture bytes
        # replace only that constant, and prediction is explicitly injected.
        with mock.patch("ecom_search.cross_encoder.MODEL_WEIGHT_SHA256", _sha(model / "model.safetensors")), mock.patch("ecom_search.cross_encoder._versions", return_value={"fixture": "1"}):
            yield directory, model
    finally:
        shutil.rmtree(directory)


class CrossEncoderTests(unittest.TestCase):
    def test_exact_content_cache_deduplicates_and_preserves_input_order(self):
        calls = []
        def predict(pairs, batch_size):
            calls.append((list(pairs), batch_size))
            return [float(len(query) + 2 * len(document)) for query, document in pairs]
        with fixture() as (directory, model):
            ce = LocalCrossEncoder(model, cache_dir=directory / "cache", predictor=predict)
            pairs = [("query", "one"), ("query", "longer"), ("query", "one")]
            self.assertEqual(ce.score_pairs(pairs), [11., 17., 11.])
            self.assertEqual(len(calls[0][0]), 2)
            self.assertEqual(calls[0][1], 32)
            self.assertEqual(ce.last_report["newly_predicted_pairs"], 2)
            self.assertEqual(ce.predict(pairs), [11., 17., 11.])
            self.assertEqual(len(calls), 1)
            self.assertEqual(ce.last_report["newly_predicted_pairs"], 0)
            self.assertEqual(ce.score_pairs([("query!", "one"), ("query", "one!")]), [12., 13.])
            self.assertEqual(len(calls), 2)

    def test_persistent_cache_and_inference_configuration_are_bound(self):
        with fixture() as (directory, model):
            first = LocalCrossEncoder(model, cache_dir=directory / "cache", predictor=lambda pairs, **kwargs: [3.] * len(pairs))
            first.score_pairs([("query", "document")])
            def must_not_predict(*args, **kwargs):
                raise AssertionError("a valid cached row should skip model loading")
            repeated = LocalCrossEncoder(model, cache_dir=directory / "cache", predictor=must_not_predict)
            self.assertEqual(repeated.score_pairs([("query", "document")]), [3.])
            repeated.close()
            self.assertEqual(repeated.score_pairs([("query", "document")]), [3.])
            length = LocalCrossEncoder(model, cache_dir=directory / "cache", max_length=256, predictor=must_not_predict)
            batch = LocalCrossEncoder(model, cache_dir=directory / "cache", batch_size=16, predictor=must_not_predict)
            self.assertNotEqual(first.fingerprint, length.fingerprint)
            self.assertNotEqual(first.fingerprint, batch.fingerprint)
            self.assertNotEqual(first.cache_directory, length.cache_directory)

    def test_modified_artifact_and_untracked_tokenizer_are_rejected(self):
        with fixture() as (_, model):
            expected = model_fingerprint(model)["artifact_fingerprint"]
            original = (model / "tokenizer.json").read_bytes()
            (model / "tokenizer.json").write_bytes(b"changed tokenizer")
            with self.assertRaises(ValueError):
                LocalCrossEncoder(model, predictor=lambda pairs, **kwargs: [])
            (model / "tokenizer.json").write_bytes(original)
            self.assertEqual(model_fingerprint(model)["artifact_fingerprint"], expected)
            (model / "added_tokens.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                model_fingerprint(model)

    def test_rewritten_manifest_cannot_unpin_official_weight(self):
        with fixture() as (_, model):
            (model / "model.safetensors").write_bytes(b"different model weights")
            manifest = json.loads((model / "model_manifest.json").read_text(encoding="utf-8"))
            for row in manifest["files"]:
                if row["path"] == "model.safetensors":
                    row.update(bytes=(model / row["path"]).stat().st_size, sha256=_sha(model / row["path"]))
            manifest["artifact_fingerprint"] = _key({"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": manifest["files"]})
            _save(model / "model_manifest.json", manifest)
            with self.assertRaises(ValueError):
                model_fingerprint(model)

    def test_cached_score_tampering_is_rejected(self):
        with fixture() as (directory, model):
            ce = LocalCrossEncoder(model, cache_dir=directory / "cache", predictor=lambda pairs, **kwargs: [1.] * len(pairs))
            ce.score_pairs([("query", "document")])
            with closing(sqlite3.connect(ce.cache_directory / "scores.sqlite3")) as database:
                database.execute("UPDATE scores SET score=999")
                database.commit()
            with self.assertRaises(ValueError):
                ce.score_pairs([("query", "document")])

    def test_invalid_inputs_outputs_and_empty_batch(self):
        with fixture() as (directory, model):
            ce = LocalCrossEncoder(model, cache_dir=directory / "cache", predictor=lambda pairs, **kwargs: [float("nan")] * len(pairs))
            self.assertEqual(ce.score_pairs([]), [])
            for pairs in ([('', 'document')], [('query', None)], ['query document']):
                with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                    ce.score_pairs(pairs)
            with self.assertRaises(ValueError):
                ce.score_pairs([("query", "document")])
            with closing(sqlite3.connect(ce.cache_directory / "scores.sqlite3")) as database:
                self.assertEqual(database.execute("SELECT count(*) FROM scores").fetchone()[0], 0)
            wrong_count = LocalCrossEncoder(model, predictor=lambda pairs, **kwargs: [])
            with self.assertRaises(ValueError):
                wrong_count.score_pairs([("query", "document")])
            for kwargs in ({"threads": 0}, {"max_length": 513}, {"batch_size": True}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    LocalCrossEncoder(model, **kwargs)

    def test_document_contract_reranking_ties_and_boundaries(self):
        self.assertEqual(product_text({"title": " Shoes ", "brand": "Brand", "description": "ignored"}), "Shoes. Brand: Brand")
        with fixture() as (_, model):
            ce = LocalCrossEncoder(model, predictor=lambda pairs, **kwargs: [1.] * len(pairs))
            documents = [{"id": "b", "title": "shoe", "brand": "brand"}, {"id": "a", "title": "boot", "brand": "brand"}]
            self.assertEqual(ce.rerank("query", documents, k=99), [("a", 1.), ("b", 1.)])
            self.assertEqual(ce.rerank("query", documents, k=0), [])
            self.assertEqual(ce.rerank("query", [], k=10), [])
            with self.assertRaises(ValueError):
                ce.rerank("query", [documents[0], documents[0]])
            with self.assertRaises(ValueError):
                ce.rerank("query", documents, k=-1)

    def test_int8_complete_call_context_order_and_duplicates_are_preserved(self):
        calls = []
        def predict(pairs, batch_size):
            calls.append(list(pairs))
            context_score = float(sum(len(document) for _, document in pairs))
            return [context_score] * len(pairs)
        with fixture() as (directory, model):
            ce = LocalCrossEncoder(model, cache_dir=directory / "cache", precision="dynamic_int8", predictor=predict)
            original = [("query", "d"), ("query", "x")]
            changed_neighbor = [("query", "d"), ("query", "yyyy")]
            self.assertEqual(ce.score_pairs(original), [2., 2.])
            first_key = ce.last_report["call_sha256"]
            self.assertEqual(ce.score_pairs(changed_neighbor), [5., 5.])
            self.assertFalse(ce.last_report["complete_call_cache_hit"])
            self.assertEqual(ce.last_report["newly_predicted_pairs"], 2)
            self.assertNotEqual(first_key, ce.last_report["call_sha256"])
            # The first exact text pair is shared, but its old score cannot be
            # reused in a different dynamic activation/batch context.
            self.assertEqual(len(calls), 2)
            self.assertEqual(ce.score_pairs(original), [2., 2.])
            self.assertTrue(ce.last_report["complete_call_cache_hit"])
            self.assertEqual(len(calls), 2)
            ce.score_pairs(list(reversed(original)))
            self.assertFalse(ce.last_report["complete_call_cache_hit"])
            self.assertEqual(len(calls), 3)
            duplicated = [("query", "d"), ("query", "d")]
            self.assertEqual(ce.score_pairs(duplicated), [2., 2.])
            self.assertEqual(calls[-1], duplicated)
            self.assertEqual(ce.last_report["newly_predicted_pairs"], 2)
            with closing(sqlite3.connect(ce.cache_directory / "scores.sqlite3")) as database:
                self.assertEqual(database.execute("SELECT count(*) FROM scores").fetchone()[0], 0)
                self.assertEqual(database.execute("SELECT count(*) FROM complete_calls").fetchone()[0], 4)

    def test_int8_precision_engine_and_cache_configuration_are_bound(self):
        with fixture() as (directory, model):
            predictor = lambda pairs, **kwargs: [1.] * len(pairs)
            fp32 = LocalCrossEncoder(model, cache_dir=directory / "cache", predictor=predictor)
            int8 = LocalCrossEncoder(model, cache_dir=directory / "cache", precision="dynamic_int8", predictor=predictor)
            self.assertNotEqual(fp32.fingerprint, int8.fingerprint)
            self.assertNotEqual(fp32.cache_directory, int8.cache_directory)
            self.assertEqual(int8.config["quantized_engine"], "onednn")
            self.assertEqual(int8.config["cache_policy"], "ordered_complete_call")
            with self.assertRaises(ValueError):
                LocalCrossEncoder(model, precision="float16")

    def test_int8_call_cache_checks_full_checksum_shape_and_finiteness(self):
        with fixture() as (directory, model):
            ce = LocalCrossEncoder(model, cache_dir=directory / "cache", precision="dynamic_int8", predictor=lambda pairs, **kwargs: [1.] * len(pairs))
            pairs = [("query", "one"), ("query", "two")]
            ce.score_pairs(pairs)
            key = ce.last_report["call_sha256"]
            database_file = ce.cache_directory / "scores.sqlite3"
            with closing(sqlite3.connect(database_file)) as database:
                database.execute("UPDATE complete_calls SET scores_json='[999.0,999.0]'")
                database.commit()
            with self.assertRaises(ValueError):
                ce.score_pairs(pairs)
            for payload in ("[1.0]", "[NaN,2.0]", '["1.0",2.0]'):
                with closing(sqlite3.connect(database_file)) as database:
                    database.execute("UPDATE complete_calls SET scores_json=?,checksum=?", (payload, ce._call_checksum(key, payload)))
                    database.commit()
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    ce.score_pairs(pairs)

    def test_int8_invalid_prediction_never_commits_partial_call(self):
        with fixture() as (directory, model):
            ce = LocalCrossEncoder(model, cache_dir=directory / "cache", precision="dynamic_int8", predictor=lambda pairs, **kwargs: [1., float("inf")])
            with self.assertRaises(ValueError):
                ce.score_pairs([("query", "one"), ("query", "two")])
            with closing(sqlite3.connect(ce.cache_directory / "scores.sqlite3")) as database:
                self.assertEqual(database.execute("SELECT count(*) FROM complete_calls").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()

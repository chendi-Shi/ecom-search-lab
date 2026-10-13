"""Fresh-cohort identity exclusion, immutable registry and real Parquet replay."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import random
import shutil
import unittest
from unittest.mock import patch
import uuid

from ecom_search.data import load_dataset
from scripts import prepare_closure_esci as closure
from scripts import prepare_public_esci as base
from tests.test_public_esci import fixture_rows


def registry_for(*identities):
    hashes = [closure.identity_hashes(qid, query) for qid, query in identities]
    return closure.seal({"protocol": closure.REGISTRY_PROTOCOL, "sources": [],
                         "query_id_sha256": sorted({pair[0] for pair in hashes}),
                         "normalized_query_sha256": sorted({pair[1] for pair in hashes})})


def select(rows, registry):
    return closure.select_fresh_metadata(rows, registry, train_queries=1, val_queries=1,
                                         test_queries=1, min_candidates=3, seed=42)


class FreshIdentityTests(unittest.TestCase):
    def test_identity_normalization_and_separate_id_and_text_exclusions(self):
        self.assertEqual(closure.identity_hashes("esci-us-0001", "ＲＥＤ　ＳＨＯＥＳ"),
                         closure.identity_hashes(1, "red  shoes"))
        # ID excludes query 2 despite a different registered text; NFKC text
        # excludes both 1 and 7 despite different IDs. Grades cannot rescue them.
        registry = registry_for((2, "previous wording"), (999, "ＲＥＤ　ＳＨＯＥＳ"))
        selected, _, _, selection = select(fixture_rows(), registry)
        self.assertFalse({"1", "2", "7"} & set(selected))
        self.assertEqual(selection["previous_experiment_exclusion"]["excluded_source_query_ids_union"], 3)
        self.assertEqual(selection["previous_experiment_exclusion"]["selected_overlap_by_id"], 0)
        self.assertEqual(selection["previous_experiment_exclusion"]["selected_overlap_by_normalized_query"], 0)

    def test_grade_blind_order_independent_and_fail_when_fresh_capacity_insufficient(self):
        class ForbiddenGrades(dict):
            def __getitem__(self, key):
                if key in {"esci_label", "qrels"}:
                    raise AssertionError("Identity selection read a grade")
                return super().__getitem__(key)
        registry = registry_for((2, "previous wording"), (999, "red shoes"))
        rows = fixture_rows()
        expected = select(rows, registry)[3]
        random.Random(4).shuffle(rows)
        self.assertEqual(expected, select([ForbiddenGrades(row) for row in rows], registry)[3])
        with self.assertRaisesRegex(ValueError, "Insufficient eligible"):
            select(rows, registry_for((1, "red shoes"), (2, "blue shoes"), (3, "green shoes")))

    def test_bad_official_id_rejected(self):
        for value in (True, None, -1, "esci-uk-1", "1.0", "x"):
            with self.assertRaises(ValueError):
                closure.canonical_id(value)


class RegistryFilesystemTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[1] / "outputs/test-tmp" / uuid.uuid4().hex
        self.directory.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def old_dataset(self, name="old"):
        directory = self.directory / name
        directory.mkdir()
        files = {}
        for position, split in enumerate(("train", "val", "test")):
            path = directory / (split + ".jsonl")
            base.write_jsonl(path, [{"query_id": f"esci-us-{100 + position}",
                                    "query": "old " + split, "qrels": {"never-publish": 3}}])
            files[path.name] = base.fingerprint(path)
        base.write_json(directory / "manifest.json", {"official_commit": base.COMMIT, "files": files})
        return directory

    def test_registry_is_hash_only_covers_all_files_and_rejects_tampering(self):
        old = self.old_dataset()
        output = self.directory / "exclusions.json"
        result = closure.build_registry([("historical", old)], output)
        self.assertEqual(result["unique_query_ids"], 3)
        content = output.read_text(encoding="utf-8")
        self.assertNotIn("never-publish", content)
        self.assertNotIn("old train", content)
        self.assertNotIn(str(self.directory), content)
        self.assertNotIn("esci-us-100", content)
        closure.read_registry(output)
        with self.assertRaises(FileExistsError):
            closure.build_registry([("historical", old)], output)
        altered = json.loads(content)
        altered["query_id_sha256"].pop()
        base.write_json(output, altered)
        with self.assertRaisesRegex(ValueError, "content hash"):
            closure.read_registry(output)

    def test_every_input_verified_before_extracting_any_identity_and_extra_files_fail_closed(self):
        old = self.old_dataset()
        # train contains malformed JSON, but its expected bytes are updated.
        # val is corrupt relative to the manifest. It must fail the global
        # fingerprint pass before attempting to parse the malformed train.
        (old / "train.jsonl").write_text("invalid json", encoding="utf-8")
        manifest = json.loads((old / "manifest.json").read_text())
        manifest["files"]["train.jsonl"] = base.fingerprint(old / "train.jsonl")
        base.write_json(old / "manifest.json", manifest)
        (old / "val.jsonl").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Historical identity file missing or changed"):
            closure.build_registry([("historical", old)], self.directory / "no-registry.json")
        clean = self.old_dataset("clean")
        base.write_jsonl(clean / "additional_queries.jsonl", [{"query_id": 999, "query": "extra"}])
        with self.assertRaisesRegex(ValueError, "additional_queries"):
            closure.build_registry([("historical", clean)], self.directory / "no-registry.json")

    def test_output_rejected_before_registry_raw_or_network_access(self):
        with patch.object(closure, "read_registry", side_effect=AssertionError("read too early")):
            with self.assertRaises(FileExistsError):
                closure.prepare(Namespace(output=self.directory))

    def test_additional_verified_query_files_are_registered(self):
        old = self.old_dataset()
        extra = old / "additional_queries.jsonl"
        base.write_jsonl(extra, [{"query_id": "esci-us-999", "query": "extra identity"}])
        manifest = json.loads((old / "manifest.json").read_text())
        manifest["files"][extra.name] = base.fingerprint(extra)
        base.write_json(old / "manifest.json", manifest)
        output = self.directory / "registry.json"
        result = closure.build_registry([("historical", old)], output)
        self.assertEqual(result["unique_query_ids"], 4)
        self.assertIn(closure.identity_hashes(999, "extra identity")[0], closure.read_registry(output)["query_id_sha256"])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "real Parquet replay requires optional pyarrow")
    def test_parquet_replay_exact_hashes_and_selection_precedes_grades(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        raw = self.directory / "raw"
        raw.mkdir()
        rows = fixture_rows()
        pq.write_table(pa.Table.from_pylist(rows), raw / base.EXAMPLES)
        pq.write_table(pa.Table.from_pylist([
            {"product_id": row["product_id"], "product_locale": "us", "product_title": row["product_id"],
             "product_brand": "Brand"} for row in rows]), raw / base.PRODUCTS)
        specs = {name: {**base.fingerprint(raw / name), "source_url": "fixture", "pointer_url": "fixture"}
                 for name in (base.EXAMPLES, base.PRODUCTS)}
        registry_path = self.directory / "registry.json"
        base.write_json(registry_path, registry_for((2, "previous wording"), (999, "red shoes")))
        outputs = [self.directory / "one", self.directory / "two"]
        original_rows = base.us_small_rows
        for output in outputs:
            def checked_rows(path, columns):
                if "esci_label" in columns:
                    self.assertTrue((output / "selection.json").is_file())
                    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
                    self.assertFalse(selection["selection_uses_esci_label"])
                    self.assertTrue(selection["previous_experiment_exclusion"]["registry_content_sha256"])
                yield from original_rows(path, columns)
            with patch.object(base, "SOURCES", specs), patch.object(base, "us_small_rows", checked_rows):
                closure.prepare(Namespace(raw=raw, output=output, exclusions=registry_path, download=False,
                    train_queries=1, val_queries=1, test_queries=1, min_candidates=3, seed=42))
        self.assertEqual({p.name: p.read_bytes() for p in outputs[0].iterdir()},
                         {p.name: p.read_bytes() for p in outputs[1].iterdir()})
        _, train, val, test = load_dataset(outputs[0])
        self.assertEqual([len(train), len(val), len(test)], [1, 1, 1])
        for row in train + val + test:
            self.assertFalse({row["query_id"]} & {"esci-us-1", "esci-us-2", "esci-us-7"})
            self.assertEqual(set(row["candidate_ids"]), set(row["qrels"]))
            self.assertEqual(set(row["qrels"].values()), {0, 2, 3})
        manifest = json.loads((outputs[0] / "manifest.json").read_text())
        self.assertNotIn(str(self.directory), json.dumps(manifest))
        self.assertEqual(manifest["raw_files"], specs)
        self.assertEqual(manifest["exclusion_registry_file"], base.fingerprint(registry_path))


if __name__ == "__main__":
    unittest.main()

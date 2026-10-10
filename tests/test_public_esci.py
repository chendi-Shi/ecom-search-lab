"""Public import contract: grade blindness, leakage exclusion and fail-closed IO."""
from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import random
import shutil
import unittest
from unittest.mock import patch
import uuid

from ecom_search.data import load_dataset, normalized_query
from scripts import prepare_public_esci as importer


def fixture_rows():
    rows = []
    queries = [(1, "red shoes", "train"), (2, "blue shoes", "train"),
               (3, "green shoes", "train"), (4, "white shoes", "train"),
               (5, "boots", "test"), (6, "sandals", "test"),
               (7, "ＲＥＤ　ＳＨＯＥＳ", "train"),  # normalized duplicate
               (8, "Case collision", "train"), (9, "case  collision", "test")]
    for qid, query, split in queries:
        for position in range(3):
            rows.append({"query_id": qid, "query": query, "split": split,
                         "product_id": f"p{qid}-{position}", "product_locale": "us",
                         "small_version": 1, "esci_label": ("E", "S", "I")[position]})
    return rows


def select(rows, **overrides):
    settings = dict(train_queries=2, val_queries=1, test_queries=1, min_candidates=3, seed=42)
    settings.update(overrides)
    return importer.select_metadata(rows, **settings)


class SelectionTests(unittest.TestCase):
    def test_selection_is_order_independent_and_does_not_read_labels(self):
        class ForbiddenLabels(dict):
            def __getitem__(self, key):
                if key == "esci_label":
                    raise AssertionError("selection inspected a relevance label")
                return super().__getitem__(key)
        rows = fixture_rows()
        expected = select(rows)[3]
        random.Random(9).shuffle(rows)
        observed = select([ForbiddenLabels(row) for row in rows])[3]
        self.assertEqual(expected, observed)

    def test_normalized_duplicates_and_cross_official_partition_are_excluded(self):
        selected, chosen, _, selection = select(fixture_rows(), train_queries=3)
        normals = [normalized_query(row["query"]) for row in selected.values()]
        self.assertEqual(len(normals), len(set(normals)))
        self.assertNotIn("case collision", normals)
        self.assertEqual(selection["summary"]["exclusions"]["cross_official_partition"], 2)
        self.assertEqual(selection["summary"]["exclusions"]["normalized_duplicate"], 1)
        for qid in chosen["test"]:
            self.assertEqual(selected[qid]["official_split"], "test")
        for split in ("train", "val"):
            self.assertTrue(all(selected[qid]["official_split"] == "train" for qid in chosen[split]))

    def test_count_uses_distinct_candidates_and_retains_all_labels(self):
        rows = fixture_rows()
        rows.extend([rows[0]] * 12)
        selected, _, _, _ = select(rows)
        labels = importer.copy_labels(rows, selected)
        for qid, item in selected.items():
            self.assertEqual(len(item["product_ids"]), 3)
            self.assertEqual(set(labels[qid]), item["product_ids"])
            self.assertIn(0, labels[qid].values())
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select(rows, min_candidates=4)

    def test_conflicting_identity_and_labels_are_rejected(self):
        rows = fixture_rows()
        with self.assertRaisesRegex(ValueError, "conflicting text"):
            select(rows + [{**rows[0], "query": "changed"}])
        selected, _, _, _ = select(rows)
        qid = next(iter(selected))
        original = next(row for row in rows if str(row["query_id"]) == qid)
        with self.assertRaisesRegex(ValueError, "Conflicting official judgments"):
            importer.copy_labels(rows + [{**original, "esci_label": "C"}], selected)
        with self.assertRaisesRegex(ValueError, "frozen candidate identities"):
            importer.copy_labels([row for row in rows if row != original], selected)


class ImporterFilesystemTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[1] / "outputs/test-tmp" / uuid.uuid4().hex
        self.directory.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_existing_output_rejected_before_source_or_network_access(self):
        args = Namespace(output=self.directory, raw=self.directory / "missing", download=True)
        with patch.object(importer, "verify_sources", side_effect=AssertionError("should reject first")):
            with self.assertRaises(FileExistsError):
                importer.prepare(args)

    def test_corrupt_source_rejected_without_automatic_redownload(self):
        (self.directory / importer.EXAMPLES).write_bytes(b"invalid")
        with patch.object(importer.urllib.request, "urlopen", side_effect=AssertionError("must not redownload")):
            with self.assertRaisesRegex(ValueError, "pinned official"):
                importer.verify_sources(self.directory, download=True)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "real Parquet importer test requires optional pyarrow")
    def test_parquet_roundtrip_determinism_complete_qrels_and_manifest(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        raw = self.directory / "raw"
        raw.mkdir()
        rows = fixture_rows()
        pq.write_table(pa.Table.from_pylist(rows), raw / importer.EXAMPLES)
        products = [{"product_id": row["product_id"], "product_locale": "us",
                     "product_title": "Official " + row["product_id"], "product_brand": "Brand"} for row in rows]
        pq.write_table(pa.Table.from_pylist(products), raw / importer.PRODUCTS)
        specs = {name: {**importer.fingerprint(raw / name), "source_url": "fixture", "pointer_url": "fixture"}
                 for name in (importer.EXAMPLES, importer.PRODUCTS)}
        outputs = [self.directory / "one", self.directory / "two"]
        with patch.object(importer, "SOURCES", specs):
            for output in outputs:
                importer.prepare(Namespace(raw=raw, output=output, download=False, train_queries=2,
                    val_queries=1, test_queries=1, min_candidates=3, seed=42))
        self.assertEqual({p.name: p.read_bytes() for p in outputs[0].iterdir()},
                         {p.name: p.read_bytes() for p in outputs[1].iterdir()})
        catalog, train, val, test = load_dataset(outputs[0])
        self.assertTrue(catalog)
        self.assertEqual([len(train), len(val), len(test)], [2, 1, 1])
        for record in train + val + test:
            self.assertEqual(set(record["candidate_ids"]), set(record["qrels"]))
            self.assertEqual(len(record["candidate_ids"]), 3)
        manifest = json.loads((outputs[0] / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["raw_files"], specs)
        self.assertFalse(manifest["selection_uses_esci_label"])
        self.assertNotIn(str(self.directory), json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()

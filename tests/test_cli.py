import json
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import unittest
import uuid
from unittest.mock import patch
from types import SimpleNamespace

from ecom_search.cli import SearchEngine
from ecom_search.index import build_index,catalog_digest
from ecom_search.ranking import PairwiseRanker


class ServiceContractTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]
        self.folder=self.root/"outputs"/"test-tmp"/uuid.uuid4().hex
        self.folder.mkdir(parents=True)
        self.catalog=[dict(id="free",title="Sugar free snack",brand="Healthy",description=""),dict(id="sugar",title="Sugar snack",brand="Sweet",description="")]
        self.index=self.folder/"index.sqlite"
        build_index(self.catalog,self.index)
        self.model=self.folder/"model.json"
        self.model.write_text(json.dumps({"ranker":PairwiseRanker().to_dict(),"catalog_sha256":catalog_digest(self.catalog)}),encoding="utf-8")

    def tearDown(self):
        if not self.folder.resolve().is_relative_to(self.root):
            raise ValueError("cleanup outside project")
        shutil.rmtree(self.folder)

    def test_api_keeps_free_variant_and_rejects_illegal_input(self):
        engine=SearchEngine(self.index,self.model)
        try:
            result=engine.search("snack without sugar",k=2)
            self.assertEqual([p["id"] for p in result["products"]],["free"])
            self.assertEqual(result["ranking"],"trained_pairwise")
            self.assertIn("feature_values",result["products"][0])
            with self.assertRaises(ValueError): engine.search("snack",k=True)
            with self.assertRaises(ValueError): engine.search("snack",pool="100")
        finally:engine.close()

    def test_catalog_tampering_cannot_reuse_frozen_model(self):
        with closing(sqlite3.connect(self.index)) as connection:
            with connection:
                connection.execute("UPDATE products SET title='different' WHERE id='free'")
        with self.assertRaisesRegex(ValueError,"catalog changed"):
            SearchEngine(self.index,self.model)

    def test_model_and_index_bind_same_catalog(self):
        obj=json.loads(self.model.read_text())
        obj["catalog_sha256"]="wrong"
        self.model.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,"fingerprints differ"):
            SearchEngine(self.index,self.model)

    def test_disk_catalog_text_mismatch_closes_failed_constructor(self):
        class Disk:
            ids=['free','sugar']
            metadata={'model':{'sha256':'fixture'},'input_fingerprint':'stale title fingerprint'}
            closed=False
            def close(self):self.closed=True
        disk=Disk()
        with patch.dict('sys.modules',{'ecom_search.disk_vectors':SimpleNamespace(DiskVectorIndex=lambda *a,**kw:disk)}):
            with self.assertRaisesRegex(ValueError,'disk/lexical catalog'):
                SearchEngine(self.index,self.model,disk_path=self.folder)
        self.assertTrue(disk.closed)

    def test_generative_catalog_mismatch_rejected_without_optional_import(self):
        fake=SimpleNamespace(metadata={'catalog_sha256':'stale catalog'})
        with patch.dict('sys.modules',{'ecom_search.generative':SimpleNamespace(GenerativeIndex=lambda path:fake)}):
            with self.assertRaisesRegex(ValueError,'generative/lexical catalog'):
                SearchEngine(self.index,self.model,generative_path=self.folder)

    def test_vector_union_cannot_bypass_negative_filter_or_known_ids(self):
        engine=SearchEngine(self.index,self.model)
        engine.vector=SimpleNamespace(metadata={'algorithm':'fixture'},search=lambda *a:[('sugar',1),('free',.9)])
        try:
            self.assertEqual([p['id'] for p in engine.search('snack without sugar',k=2)['products']],['free'])
            engine.vector.search=lambda *a:[('unknown',1)]
            with self.assertRaisesRegex(ValueError,'unknown product IDs'):engine.search('snack')
        finally:engine.close()

    def test_profiles_are_real_explicit_signals_and_can_be_cleared(self):
        engine=SearchEngine(self.index,self.model,profile_path=self.folder/'profiles.sqlite')
        try:
            cold=engine.search('snack',k=2,user_id='demo')
            self.assertFalse(cold['personalization']['applied'])
            engine.record_preference('demo','free','like')
            self.assertTrue(engine.search('snack',k=2,user_id='demo')['personalization']['applied'])
            engine.record_preference('demo','free','hide')
            self.assertEqual([p['id'] for p in engine.search('snack',k=2,user_id='demo')['products']],['sugar'])
            engine.profile_store.clear('demo')
            self.assertEqual(len(engine.search('snack',k=2,user_id='demo')['products']),2)
            with self.assertRaisesRegex(ValueError,'boolean'):engine.search('snack',use_generative='false')
        finally:engine.close()

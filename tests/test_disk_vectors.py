import importlib.util
import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

from ecom_search.disk_vectors import DiskVectorIndex,build_disk_store
from ecom_search.data import file_sha256,save_json


@unittest.skipUnless(importlib.util.find_spec("numpy"),"optional numpy unavailable")
class DiskVectorTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]
        self.folder=self.root/"outputs"/"test-tmp"/uuid.uuid4().hex
        self.folder.mkdir(parents=True)
    def tearDown(self):
        assert self.folder.resolve().is_relative_to(self.root)
        shutil.rmtree(self.folder)
    def source(self,ids,values):
        import numpy as np
        directory=self.folder/"source"
        directory.mkdir()
        np.save(directory/"vectors.npy",np.asarray(values,dtype=np.float32),allow_pickle=False)
        save_json(directory/"ids.json",ids)
        save_json(directory/"metadata.json",{"vectors_sha256":file_sha256(directory/"vectors.npy"),"ids_sha256":file_sha256(directory/"ids.json"),"model":None,"input_fingerprint":"synthetic-unit-test"})
        return directory
    def test_chunk_boundaries_and_quantization_keep_cosine(self):
        import numpy as np
        from ecom_search.vectors import exact_cosine_search
        values=np.random.default_rng(44).normal(size=(43,16)).astype(np.float32)
        ids=[str(i) for i in range(len(values))]
        self.source(ids,values)
        query=values[4]
        reference=exact_cosine_search(ids,values,query,7)
        for mode in ("float32","int8"):
            build_disk_store(self.folder/"source",self.folder/mode,mode=mode,chunk=9)
            for chunk in (1,8,100):
                with DiskVectorIndex(self.folder/mode,chunk=chunk) as index:
                    hits=index.search_vector(query,7)
                    self.assertEqual(hits[0][0],"4")
                    self.assertEqual({pid for pid,_ in hits},{pid for pid,_ in reference})
                    self.assertLess(abs(hits[0][1]-1),1e-5)
                    self.assertEqual(len(index.search_vector(query,100)),len(ids))
                    self.assertEqual(index.search_vector([],0),[])
    def test_corrupt_store_rejected_and_output_not_overwritten(self):
        import numpy as np
        self.source(["x"],np.array([[1.,2.]]))
        build_disk_store(self.folder/"source",self.folder/"q")
        with self.assertRaises(FileExistsError):build_disk_store(self.folder/"source",self.folder/"q")
        (self.folder/"q"/"ids.json").write_text('["tampered"]')
        with self.assertRaisesRegex(ValueError,"checksum"):DiskVectorIndex(self.folder/"q")

    def test_tied_topk_is_identical_across_chunk_sizes(self):
        self.source(["row-z","row-a","row-m","other"],[[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
        for mode in ("float32","int8"):
            build_disk_store(self.folder/"source",self.folder/mode,mode=mode)
            for chunk in (1,2,3,100):
                with DiskVectorIndex(self.folder/mode,chunk=chunk) as index:
                    self.assertEqual([pid for pid,_ in index.search_vector([1.,0.],2)],["row-z","row-a"])
                    self.assertGreater(index.workspace_estimate()["explicit_array_workspace_upper_bound_bytes"],min(chunk,4)*2*4)

    def test_explicit_close_releases_windows_files_without_gc(self):
        self.source(["x"],[[1.,2.]])
        build_disk_store(self.folder/"source",self.folder/"q")
        # Builder has closed its input mapping, independently of object lifetime.
        (self.folder/"source"/"vectors.npy").rename(self.folder/"source"/"released.npy")
        with DiskVectorIndex(self.folder/"q") as index:
            matrix_mapping=index.matrix._mmap
            scale_mapping=index.scales._mmap
            self.assertFalse(matrix_mapping.closed)
        self.assertTrue(matrix_mapping.closed)
        self.assertTrue(scale_mapping.closed)
        index.close()  # idempotent
        with self.assertRaisesRegex(ValueError,"closed"):
            index.search_vector([1.,2.])
        (self.folder/"q"/"quantized.npy").rename(self.folder/"q"/"released.npy")
        (self.folder/"q"/"scales.npy").unlink()

    def test_mapping_release_on_constructor_and_build_failure(self):
        import numpy as np
        self.source(["x"],[[1.,2.]])
        build_disk_store(self.folder/"source",self.folder/"q")
        np.save(self.folder/"q"/"scales.npy",np.array([1.,2.],dtype=np.float32),allow_pickle=False)
        metadata=json.loads((self.folder/"q"/"metadata.json").read_text(encoding="utf-8"))
        metadata["files"]["scales.npy"]={"sha256":file_sha256(self.folder/"q"/"scales.npy"),"bytes":(self.folder/"q"/"scales.npy").stat().st_size}
        save_json(self.folder/"q"/"metadata.json",metadata)
        with self.assertRaisesRegex(ValueError,"scale shape"):
            DiskVectorIndex(self.folder/"q")
        (self.folder/"q"/"quantized.npy").rename(self.folder/"q"/"released.npy")
        (self.folder/"q"/"scales.npy").unlink()
        # A zero source row fails normalization after opening output mappings.
        np.save(self.folder/"source"/"vectors.npy",np.array([[0.,0.]],dtype=np.float32),allow_pickle=False)
        source_metadata=json.loads((self.folder/"source"/"metadata.json").read_text(encoding="utf-8"))
        source_metadata["vectors_sha256"]=file_sha256(self.folder/"source"/"vectors.npy")
        save_json(self.folder/"source"/"metadata.json",source_metadata)
        with self.assertRaisesRegex(ValueError,"zero vectors"):
            build_disk_store(self.folder/"source",self.folder/"failure")
        (self.folder/"source"/"vectors.npy").rename(self.folder/"source"/"released.npy")
        (self.folder/"failure"/"quantized.npy").rename(self.folder/"failure"/"released.npy")
        (self.folder/"failure"/"scales.npy").unlink()

    def test_moved_text_model_uses_same_sha_packaged_copy(self):
        import numpy as np
        from ecom_search.vectors import model_fingerprint
        self.source(["x"],[[1.,2.]])
        packaged=self.folder/"outputs/text-assets/minilm"
        packaged.mkdir(parents=True)
        (packaged/"modules.json").write_text("[]",encoding="utf-8")
        (packaged/"model.safetensors").write_bytes(b"synthetic checkpoint bytes for fingerprint-only test")
        fingerprint=model_fingerprint(packaged)
        missing=self.folder/"original-machine/minilm"
        metadata=json.loads((self.folder/"source"/"metadata.json").read_text(encoding="utf-8"))
        metadata["model"]={**fingerprint,"path":str(missing)}
        save_json(self.folder/"source"/"metadata.json",metadata)
        build_disk_store(self.folder/"source",self.folder/"q")
        class Encoder:
            def encode(self,*args,**kwargs):return np.array([[1.,2.]],dtype=np.float32)
        with patch("ecom_search.disk_vectors.ROOT",self.folder),patch("ecom_search.vectors._load_encoder",return_value=Encoder()) as load:
            with DiskVectorIndex(self.folder/"q") as index:
                before=json.dumps(index.metadata,sort_keys=True)
                self.assertEqual(index.search("test",1)[0][0],"x")
                self.assertEqual(json.dumps(index.metadata,sort_keys=True),before)
            load.assert_called_once_with(packaged)
            (packaged/"model.safetensors").write_bytes(b"different checkpoint")
            with DiskVectorIndex(self.folder/"q") as index,self.assertRaisesRegex(ValueError,"model changed"):
                index.search("test",1)
            self.assertEqual(load.call_count,1)

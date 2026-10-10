"""Meaningful optional multimodal tests without downloading any model/images."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock
import uuid

from ecom_search.multimodal import fuse, normalize, evaluate_pairs, MultimodalIndex, _key, _sha, _save, _validate_split, _cache_embeddings, _model_info


@contextmanager
def temporary_workspace():
    root = Path(__file__).resolve().parents[1]
    target = (root / (".multimodal-test-" + uuid.uuid4().hex)).resolve()
    if not target.is_relative_to(root):
        raise ValueError("temporary directory escaped project")
    target.mkdir()
    try:
        yield target
    finally:
        shutil.rmtree(target)


class SplitTests(unittest.TestCase):
    def test_product_leakage_is_rejected(self):
        rows = [{"query_id": "query", "target_id": "item", "product_id": "product"}]
        self.assertEqual(_validate_split(rows, {"item"}), {"product"})
        with self.assertRaises(ValueError):
            _validate_split(rows, {"item"}, {"product"})
        with self.assertRaises(ValueError):
            _validate_split(rows, {"another-item"})


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy unavailable")
class FusionTests(unittest.TestCase):
    def test_normalization_and_missing_modality(self):
        import numpy as np
        np.testing.assert_allclose(fuse([[3., 0.]], None), [[1., 0.]])
        np.testing.assert_allclose(fuse(None, [[0., 2.]]), [[0., 1.]])
        np.testing.assert_allclose(fuse([[1., 0.]], [[0., 1.]], gate=.5), [[2 ** -.5, 2 ** -.5]], atol=1e-6)
        with self.assertRaises(ValueError):
            fuse(None, None)
        with self.assertRaises(ValueError):
            fuse([[1., 0.]], [[0., 1.]], gate=float("nan"))
        with self.assertRaises(ValueError):
            normalize([[float("inf"), 0.]])
        with self.assertRaises(ValueError):
            normalize([[0., 0.]])

    def test_pair_recall_and_missing_target(self):
        query = {"text": [[1., 0.], [0., 1.]], "image": [[0., 1.], [1., 0.]]}
        gallery = {"text": [[1., 0.], [0., 1.]], "image": [[1., 0.], [0., 1.]]}
        rows = [{"target_id": "a"}, {"target_id": "b"}]
        text = evaluate_pairs(query, gallery, rows, ["a", "b"], mode="text")
        image = evaluate_pairs(query, gallery, rows, ["a", "b"], mode="image")
        self.assertEqual(text["pair_recall_at_1"], 1)
        self.assertEqual(image["pair_recall_at_1"], 0)
        self.assertEqual(image["pair_recall_at_5"], 1)
        with self.assertRaises(ValueError):
            evaluate_pairs(query, gallery, [{"target_id": "missing"}, {"target_id": "b"}], ["a", "b"], mode="text")

    def test_projection_changes_scores_and_has_finite_shape_contract(self):
        import numpy as np
        base = fuse([[1., 1.]], None)
        projected = fuse([[1., 1.]], None, text_projection=[1., -1.])
        self.assertGreater(projected[0, 0], base[0, 0])
        self.assertLess(projected[0, 1], base[0, 1])
        with self.assertRaises(ValueError):
            fuse([[1., 1.]], None, text_projection=[0.])

    def test_persisted_index_boundaries_rerank_and_tamper(self):
        import numpy as np
        class FakeEncoder:
            def query(self, text, image):
                return {"text": np.array([[1., 0.]], dtype=np.float32)}
        with temporary_workspace() as directory:
            _save(directory / "catalog.json", [{"id": "a", "title": "first"}, {"id": "b", "title": "second"}])
            _save(directory / "model.json", {"gate": .5, "text_projection": [0., 0.], "image_projection": [0., 0.]})
            np.savez(directory / "gallery.npz", text=[[1., 0.], [0., 1.]], image=[[1., 0.], [0., 1.]])
            _save(directory / "metadata.json", {"artifacts": {name: _sha(directory / name) for name in ("catalog.json", "model.json", "gallery.npz")}})
            index = MultimodalIndex(directory, encoder=FakeEncoder())
            self.assertEqual([row[0] for row in index.search("query", k=99)], ["a", "b"])
            self.assertEqual(index.search("query", k=0), [])
            self.assertEqual(index.rerank(["b"], "query"), [("b", 0.)])
            with self.assertRaises(ValueError):
                index.rerank(["unknown"], "query")
            with self.assertRaises(ValueError):
                index.search("query", k=-1)
            with self.assertRaises(ValueError):
                index.search_vectors({"text": [[1., 0.]]}, mode="image")
            (directory / "model.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                MultimodalIndex(directory)

    def test_cache_hit_still_checks_actual_image_bytes(self):
        import numpy as np
        class FakeEncoder:
            info = {"sha256": "fake-local-encoder"}
            def encode(self, texts, images, batch_size):
                return {"text": np.array([[1., 0.]], dtype=np.float32), "image": np.array([[0., 1.]], dtype=np.float32)}, {"count": 1}
        with temporary_workspace() as root:
            image = root / "image.jpg"
            image.write_bytes(b"fake bytes used only for cache integrity test")
            rows = [{"query": "original text", "image_path": "image.jpg", "image_sha256": _sha(image)}]
            _, first = _cache_embeddings(rows, root, FakeEncoder(), root / "cache", text_field="query", batch_size=1)
            _, second = _cache_embeddings(rows, root, FakeEncoder(), root / "cache", text_field="query", batch_size=1)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            image.write_bytes(b"changed input bytes")
            with self.assertRaises(ValueError):
                _cache_embeddings(rows, root, FakeEncoder(), root / "cache", text_field="query", batch_size=1)

    def test_moved_project_recovers_paths_but_still_verifies_frozen_content(self):
        import numpy as np
        class FakeEncoder:
            def query(self, text, image):
                return {"text": np.array([[1., 0.]], dtype=np.float32)}
        with temporary_workspace() as parent:
            moved_root = parent / "moved-project"
            data_root = moved_root / "data" / "multimodal"
            image_path = data_root / "images" / "real-fixture.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"unit-test bytes; not a claimed real dataset image")
            _save(data_root / "manifest.json", {"fixture": "frozen-content"})
            model_path = moved_root / "outputs" / "multimodal-assets" / "tinyclip"
            model_path.mkdir(parents=True)
            for name in ("config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors"):
                (model_path / name).write_bytes(b"unit-test model artifact; encoder mocked")
            model_info = _model_info(model_path)
            old_root = parent / "missing-original-project"
            model_info["path"] = str(old_root / "outputs" / "multimodal-assets" / "tinyclip")
            selected = moved_root / "outputs" / "multimodal" / "selected"
            selected.mkdir(parents=True)
            _save(selected / "catalog.json", [{"id": "a", "title": "fixture", "image_path": str(old_root / "data" / "multimodal" / "images" / image_path.name), "image_sha256": _sha(image_path)}])
            _save(selected / "model.json", {"gate": .5, "text_projection": [0., 0.], "image_projection": [0., 0.]})
            np.savez(selected / "gallery.npz", text=[[1., 0.]], image=[[1., 0.]])
            _save(selected / "metadata.json", {"artifacts": {name: _sha(selected / name) for name in ("catalog.json", "model.json", "gallery.npz")}, "model": model_info, "data_root": str(old_root / "data" / "multimodal"), "data_manifest_sha256": _sha(data_root / "manifest.json"), "threads": 2})
            frozen_model_sha = _sha(selected / "model.json")
            with mock.patch("ecom_search.multimodal.ROOT", moved_root), mock.patch("ecom_search.multimodal.ClipEncoder", return_value=FakeEncoder()) as loader:
                index = MultimodalIndex(selected)
                self.assertEqual(index.catalog["a"]["image_path"], str(image_path.resolve()))
                self.assertEqual(index.search("query"), [("a", 1.)])
                self.assertEqual(loader.call_args.args[0], str(model_path.resolve()))
                self.assertEqual(_sha(selected / "model.json"), frozen_model_sha)
                (model_path / "model.safetensors").write_bytes(b"tampered relocated weights")
                with self.assertRaises(ValueError):
                    MultimodalIndex(selected).search("query")
                image_path.write_bytes(b"tampered relocated image")
                with self.assertRaises(ValueError):
                    MultimodalIndex(selected)

    def test_copied_project_uses_local_assets_even_when_original_still_exists(self):
        import numpy as np
        class FakeEncoder:
            def query(self, text, image):
                return {"text": np.array([[1., 0.]], dtype=np.float32)}
        with temporary_workspace() as parent:
            old_root, copied_root = parent / "original-project", parent / "copied-project"
            old_data = old_root / "data" / "multimodal"
            old_image = old_data / "images" / "fixture.jpg"
            old_image.parent.mkdir(parents=True)
            old_image.write_bytes(b"unit-test image bytes; not a real dataset image")
            _save(old_data / "manifest.json", {"fixture": "frozen-content"})
            old_model = old_root / "outputs" / "multimodal-assets" / "tinyclip"
            old_model.mkdir(parents=True)
            for name in ("config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors"):
                (old_model / name).write_bytes(b"unit-test model artifact; encoder mocked")
            local_data = copied_root / "data" / "multimodal"
            local_model = copied_root / "outputs" / "multimodal-assets" / "tinyclip"
            shutil.copytree(old_data, local_data)
            shutil.copytree(old_model, local_model)
            local_image = local_data / "images" / old_image.name
            selected = copied_root / "outputs" / "multimodal" / "selected"
            selected.mkdir(parents=True)
            _save(selected / "catalog.json", [{"id": "a", "title": "fixture", "image_path": str(old_image),
                                                "image_relative_path": "images/fixture.jpg", "image_sha256": _sha(old_image)}])
            _save(selected / "model.json", {"gate": .5, "text_projection": [0., 0.], "image_projection": [0., 0.]})
            np.savez(selected / "gallery.npz", text=[[1., 0.]], image=[[1., 0.]])
            _save(selected / "metadata.json", {"artifacts": {name: _sha(selected / name) for name in ("catalog.json", "model.json", "gallery.npz")},
                                                "model": _model_info(old_model), "data_root": str(old_data),
                                                "data_manifest_sha256": _sha(old_data / "manifest.json"), "threads": 2})
            with mock.patch("ecom_search.multimodal.ROOT", copied_root), mock.patch("ecom_search.multimodal.ClipEncoder", return_value=FakeEncoder()) as loader:
                index = MultimodalIndex(selected)
                self.assertEqual(index.data_root, local_data.resolve())
                self.assertEqual(index.catalog["a"]["image_path"], str(local_image.resolve()))
                self.assertEqual(index.search("query"), [("a", 1.)])
                self.assertEqual(loader.call_args.args[0], str(local_model.resolve()))
                explicit = MultimodalIndex(selected, data_root=old_data, model_path=old_model)
                self.assertEqual(explicit.catalog["a"]["image_path"], str(old_image.resolve()))
                explicit.search("query")
                self.assertEqual(loader.call_args.args[0], str(old_model.resolve()))

                original_manifest = (local_data / "manifest.json").read_bytes()
                (local_data / "manifest.json").write_bytes(b"tampered copied manifest")
                with self.assertRaises(ValueError):
                    MultimodalIndex(selected)
                (local_data / "manifest.json").write_bytes(original_manifest)
                local_image.write_bytes(b"tampered copied image")
                with self.assertRaises(ValueError):
                    MultimodalIndex(selected)
                local_image.write_bytes(old_image.read_bytes())
                (local_model / "model.safetensors").write_bytes(b"tampered copied weights")
                with self.assertRaises(ValueError):
                    MultimodalIndex(selected).search("query")
                # The accessible, valid old copy must not silently mask any of
                # these local failures. Only an explicit override may use it.
                self.assertEqual(MultimodalIndex(selected, data_root=old_data, model_path=old_model).search("query"), [("a", 1.)])
                local_image.unlink()
                with self.assertRaises(ValueError):
                    MultimodalIndex(selected)


if __name__ == "__main__":
    unittest.main()

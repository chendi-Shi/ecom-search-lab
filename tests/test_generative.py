import importlib.util
import json
from unittest.mock import patch
import unittest

OPTIONAL_AVAILABLE = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("torch") is not None
if OPTIONAL_AVAILABLE:
    import numpy as np
    import torch
    from ecom_search.generative import GenerativeIndex, QuerySIDDecoder, SIDRetriever, assign_semantic_ids, fit_residual_codebooks


@unittest.skipUnless(OPTIONAL_AVAILABLE, "optional CPU numpy/torch unavailable")
class GenerativeTests(unittest.TestCase):
    def test_persisted_weights_tamper_rejected_before_loading(self):
        metadata = json.dumps({"schema_version": 1, "files": {"decoder.pt": "recorded fingerprint"}})
        with patch("pathlib.Path.read_text", return_value=metadata), \
                patch("ecom_search.generative.file_sha256", return_value="changed weights fingerprint"):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                GenerativeIndex("unused-test-artifact")

    def test_codebook_fit_and_assignment_are_deterministic(self):
        training = np.asarray([[1., 0., 0.], [0.9, 0.1, 0.], [0., 1., 0.], [0.1, 0.9, 0.]], dtype=np.float32)
        books = fit_residual_codebooks(training, levels=2, codebook_size=2, seed=3)
        repeated = fit_residual_codebooks(training, levels=2, codebook_size=2, seed=3)
        np.testing.assert_array_equal(books, repeated)
        inferred = assign_semantic_ids(np.asarray([[0., 0., 1.]], dtype=np.float32), books)
        self.assertEqual(inferred.shape, (1, 2))
        self.assertTrue(np.all((inferred >= 0) & (inferred < 2)))

    def test_constraints_prevent_invalid_greedy_sid(self):
        torch.manual_seed(1)
        model = QuerySIDDecoder(dimension=3, levels=2, codebook_size=3, hidden=8, token_dim=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output.bias[2] = 20
        vectors = np.asarray([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        retriever = SIDRetriever(model, ["a", "b"], vectors, np.asarray([[0, 0], [1, 1]]), beam_width=2)
        hits, info = retriever.search_vector([1., 0., 0.], k=2)
        self.assertEqual({pid for pid, _ in hits}, {"a", "b"})
        self.assertTrue(info["unconstrained_greedy_invalid_id"])
        self.assertEqual(info["constrained_invalid_id_count"], 0)

    def test_collision_disambiguation_only_inside_generated_bucket(self):
        model = QuerySIDDecoder(dimension=3, levels=1, codebook_size=2, hidden=8, token_dim=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output.bias[0] = 10
        vectors = np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
        retriever = SIDRetriever(model, ["a", "b", "dense-nearest-unselected"], vectors,
                                 np.asarray([[0], [0], [1]]), beam_width=1)
        hits, info = retriever.search_vector([0.1, 0.9, 1.], k=3)
        self.assertEqual([pid for pid, _ in hits], ["b", "a"])
        self.assertNotIn("dense-nearest-unselected", [pid for pid, _ in hits])
        self.assertEqual(info["collision_dot_products"], 2)

    def test_decoder_really_learns_conditional_sid_tokens(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = QuerySIDDecoder(dimension=2, levels=2, codebook_size=2, hidden=12, token_dim=4)
        x = torch.tensor([[1., 0.], [0., 1.]])
        y = torch.tensor([[0, 0], [1, 1]])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        criterion = torch.nn.CrossEntropyLoss()
        initial = float(criterion(model(x, y).reshape(-1, 2), y.reshape(-1)).detach())
        for _ in range(60):
            optimizer.zero_grad()
            loss = criterion(model(x, y).reshape(-1, 2), y.reshape(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        final = float(criterion(model(x, y).reshape(-1, 2), y.reshape(-1)).detach())
        self.assertLess(final, initial * 0.1)
        retriever = SIDRetriever(model, ["a", "b"], x.numpy(), y.numpy(), beam_width=1)
        self.assertEqual(retriever.search_vector([1., 0.], 1)[0][0][0], "a")
        self.assertEqual(retriever.search_vector([0., 1.], 1)[0][0][0], "b")


if __name__ == "__main__":
    unittest.main()

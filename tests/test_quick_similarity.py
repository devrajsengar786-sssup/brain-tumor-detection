from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from retrieval.quick_ui import _rank_similar_images, quick_similarity_feature


class QuickSimilarityTests(unittest.TestCase):
    def _synthetic_mri(self, shift: int = 0) -> Image.Image:
        image = np.zeros((96, 96), dtype=np.uint8)
        image[24 + shift : 72 + shift, 30:66] = 180
        image[38 + shift : 58 + shift, 42:56] = 245
        return Image.fromarray(image, mode="L").convert("RGB")

    def test_visual_feature_is_normalized_and_deterministic(self):
        image = self._synthetic_mri()
        feature = quick_similarity_feature(image)

        self.assertGreater(feature.size, 32)
        self.assertAlmostEqual(float(np.linalg.norm(feature)), 1.0, places=6)
        np.testing.assert_allclose(feature, quick_similarity_feature(image))

    def test_identical_reference_ranks_first(self):
        query = self._synthetic_mri()
        different_pixels = np.zeros((96, 96), dtype=np.uint8)
        different_pixels[18:78, 18:78] = 90
        different_pixels[44:52, 18:78] = 220
        different = Image.fromarray(different_pixels, mode="L").convert("RGB")
        references = [
            {"case_id": "different", "label": "User supplied", "image": different},
            {"case_id": "identical", "label": "User supplied", "image": query},
        ]

        results = _rank_similar_images(query, references, top_k=2)

        self.assertEqual(results[0]["case_id"], "identical")
        self.assertAlmostEqual(results[0]["similarity"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

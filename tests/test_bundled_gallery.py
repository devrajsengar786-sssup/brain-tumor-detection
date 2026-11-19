"""The reference gallery shipped in data/gallery_2d (used by Streamlit Cloud, which has no local artifacts/)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from retrieval.gallery2d import Gallery2D

REPO = Path(__file__).resolve().parents[1]
BUNDLED = REPO / "data" / "gallery_2d"
# Must match inference.EMBEDDING_NAME; written out here so this test does not import TensorFlow.
EMBEDDING_NAME = "multiclass_resnet50.h5:resnet50-global-average-pool:brain-crop-128"


class BundledGalleryTests(unittest.TestCase):
    def test_bundled_gallery_is_complete_and_held_out_cases_are_not_searchable(self):
        gallery = Gallery2D.load(BUNDLED, expected_feature=EMBEDDING_NAME)
        self.assertEqual(len(gallery.case_ids), 103)
        self.assertEqual(gallery.embeddings.shape, (103, 2048))
        demo = gallery.demo_cases()
        self.assertEqual(len(demo), 22)
        self.assertTrue(set(demo).isdisjoint(gallery.case_ids))
        cases = pd.read_csv(BUNDLED / "cases.csv")
        self.assertTrue(cases["verified"].all())
        self.assertEqual(cases["participant_id"].str.match(r"UPENN-GBM-\d{5}_11").sum(), len(cases))
        for case_id in gallery.case_ids:
            self.assertTrue((BUNDLED / "reference" / f"{case_id}.png").is_file(), case_id)
        self.assertEqual(sorted(p.stem for p in (BUNDLED / "demo").glob("*.png")), sorted(demo))
        # No raw scans are shipped, only the exported slices.
        self.assertEqual(list(BUNDLED.rglob("*.nii*")), [])

    def test_inference_feature_name_matches(self):
        source = (REPO / "inference.py").read_text(encoding="utf-8")
        self.assertIn('EMBEDDING_NAME = f"{TYPE_MODEL[\'file\']}:resnet50-global-average-pool:brain-crop-128"', source)
        self.assertIn('"file": "multiclass_resnet50.h5"', source)

    def test_app_finds_a_gallery_without_any_configuration(self):
        """Like Streamlit Cloud: no GALLERY_2D_DIR, so the app must fall back to a gallery on its own."""
        st.cache_resource.clear()
        os.environ.pop("GALLERY_2D_DIR", None)
        at = AppTest.from_file(str(REPO / "app.py"), default_timeout=120)
        at.run()
        self.assertFalse(at.exception, at.exception)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Reference library: 103 cases", text)
        self.assertNotIn("Reference library unavailable", text)
        self.assertTrue(any(b.key == "demo-button" for b in at.button))


if __name__ == "__main__":
    unittest.main()

"""Page-level tests of app.py using Streamlit's AppTest.

The two classifiers are replaced by tiny stand-in Keras models (mocks) and the
reference gallery is a synthetic one, so these tests check the page's behaviour,
not model quality. The real models and real gallery are checked manually in the browser.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

import inference
from inference import ModelSet
from retrieval.experiment import build_upenn_gallery
from retrieval.gallery2d import build_index, prepare_slices
from retrieval.upenn import build_inventory
from tests.test_inference import scan, tiny_binary, tiny_type
from tests.upenn_fixtures import assets_for, participants_for, write_subject

APP = str(Path(__file__).resolve().parents[1] / "app.py")
NUMBERS = list(range(2, 18))


class CountingModels(ModelSet):
    calls = 0

    def analyse(self, image):
        type(self).calls += 1
        return super().analyse(image)

    def _type_forward(self, image):
        # The random stand-in backbone gives near-identical features for every image, so use
        # a simple image-only feature (mean-centred 8x8 intensity grid) as the embedding.
        _, probabilities = super()._type_forward(image)
        grid = np.asarray(image.convert("L").resize((8, 8)), dtype=np.float32).ravel()
        return grid - grid.mean(), probabilities


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class AppPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        raw = root / "raw"
        raw.mkdir()
        for n in NUMBERS:
            write_subject(raw, n)
        bundle = root / "bundle"
        build_upenn_gallery(build_inventory(assets_for(raw, NUMBERS, followups=[3, 7])), participants_for(NUMBERS, followups=[3, 7, 20]), raw, bundle, seed=7, progress=None)
        cls.gallery_dir = root / "gallery_2d"
        prepare_slices(bundle, cls.gallery_dir, workers=2, fetch=lambda url: (raw / url.rsplit("/", 1)[-1]).read_bytes(), progress=None)
        cls.tumour_models = CountingModels(tiny_binary(8.0), tiny_type(1))
        build_index(cls.gallery_dir, cls.tumour_models.embed, inference.EMBEDDING_NAME)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        st.cache_resource.clear()
        CountingModels.calls = 0
        os.environ["GALLERY_2D_DIR"] = str(self.gallery_dir)
        self.models = self.tumour_models
        patcher = mock.patch.object(ModelSet, "load", classmethod(lambda cls_, *a, **k: self.models))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        os.environ.pop("GALLERY_2D_DIR", None)

    def app(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=120)
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    @staticmethod
    def text(at: AppTest) -> str:
        return " ".join(m.value for m in at.markdown)

    def button(self, at: AppTest, key: str):
        return next(b for b in at.button if b.key == key)

    # ----------------------------------------------------------------- states
    def test_empty_state_has_one_uploader_and_a_demo_action(self):
        at = self.app()
        self.assertEqual(len(at.file_uploader), 1)
        self.assertTrue(any(b.key == "demo-button" for b in at.button))
        text = self.text(at)
        self.assertIn("Brain MRI Analysis", text)
        self.assertNotIn("Model prediction", text)
        self.assertEqual(CountingModels.calls, 0)
        self.assertEqual(len(at.selectbox) + len(at.tabs) + len(at.sidebar.children), 0)  # no model picker, tabs or sidebar

    def test_demo_scan_shows_everything_in_order_on_one_page(self):
        at = self.app()
        self.button(at, "demo-button").click().run()
        self.assertFalse(at.exception, at.exception)
        text = self.text(at)
        order = [text.index(part) for part in ("Model prediction", "Predicted tumour type", "Similar cases", "Rank 1", "Reference case · documented diagnosis")]
        self.assertEqual(order, sorted(order))
        for expected in ("Tumour detected", "Glioma", "2D visual similarity", "Uploaded scan · model prediction", "Library scope",
                         "Tumour volume (from supplied mask)", "DOI 10.7937/TCIA.709X-DN49", "UPENN-GBM-"):
            self.assertIn(expected, text)
        self.assertEqual(at.segmented_control[0].value, "Grad-CAM overlay")
        self.assertEqual(len([b for b in at.button if str(b.key).startswith("select-")]), 5)
        self.assertEqual(CountingModels.calls, 1)

    def test_selecting_a_case_or_view_does_not_rerun_inference(self):
        at = self.app()
        self.button(at, "demo-button").click().run()
        selectors = [b for b in at.button if str(b.key).startswith("select-")]
        self.assertEqual([b.label for b in selectors], ["Selected", "Compare", "Compare", "Compare", "Compare"])  # top result open
        selectors[2].click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual([b.label for b in at.button if str(b.key).startswith("select-")], ["Compare", "Compare", "Selected", "Compare", "Compare"])
        self.assertIn("rank 3", self.text(at))
        at.segmented_control[0].set_value("Original").run()
        self.assertEqual(CountingModels.calls, 1)

    def test_upload_then_replacement_clears_previous_results(self):
        at = self.app()
        at.file_uploader[0].upload("first.png", png_bytes(scan()), "image/png").run()
        self.assertFalse(at.exception, at.exception)
        self.assertIn("first.png", self.text(at))
        self.assertEqual(CountingModels.calls, 1)
        [b for b in at.button if str(b.key).startswith("select-")][3].click().run()
        first_id = at.session_state["result"]["id"]
        self.assertIsNotNone(at.session_state["selected_case"])

        at.file_uploader[0].upload("second.png", png_bytes(scan(offset=-40)), "image/png").run()
        self.assertFalse(at.exception, at.exception)
        text = self.text(at)
        self.assertIn("second.png", text)
        self.assertNotIn("first.png", text)
        self.assertNotEqual(at.session_state["result"]["id"], first_id)
        self.assertNotIn("selected_case", at.session_state)
        self.assertEqual([b.label for b in at.button if str(b.key).startswith("select-")][0], "Selected")
        self.assertEqual(CountingModels.calls, 2)

    def test_uploading_a_reference_image_excludes_it_from_its_own_results(self):
        reference = sorted((self.gallery_dir / "reference").glob("*.png"))[0]
        at = self.app()
        at.file_uploader[0].upload(reference.name, reference.read_bytes(), "image/png").run()
        text = self.text(at)
        self.assertIn(f"identical to reference case {reference.stem}", text)
        self.assertFalse(any(b.key == f"select-{reference.stem}" for b in at.button))

    def test_invalid_file_gives_a_readable_error(self):
        at = self.app()
        at.file_uploader[0].upload("notes.png", b"this is not an image", "image/png").run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any("could not be read" in e.value for e in at.error))
        self.assertNotIn("Model prediction", self.text(at))
        self.assertEqual(CountingModels.calls, 0)

    def test_no_tumour_prediction_shows_scope_warning(self):
        self.models = CountingModels(tiny_binary(-8.0), tiny_type(1))
        at = self.app()
        at.file_uploader[0].upload("scan.png", png_bytes(scan()), "image/png").run()
        text = self.text(at)
        self.assertIn("No tumour detected", text)
        self.assertIn("Not reported", text)
        self.assertIn("The model predicted no tumour for this scan, but the reference library only contains glioblastoma", text)

    def test_other_tumour_type_shows_scope_warning(self):
        self.models = CountingModels(tiny_binary(8.0), tiny_type(2))
        at = self.app()
        at.file_uploader[0].upload("scan.png", png_bytes(scan()), "image/png").run()
        self.assertIn("The model predicted meningioma for this scan", self.text(at))

    # ----------------------------------------------------------------- failures keep available results
    def test_missing_gallery_is_explained_and_analysis_still_shown(self):
        os.environ["GALLERY_2D_DIR"] = str(Path(self.tmp.name) / "no_gallery")
        at = self.app()
        self.assertIn("Reference library unavailable", self.text(at))
        self.assertIn("python -m retrieval.gallery2d prepare", self.text(at))
        self.assertFalse(any(b.key == "demo-button" for b in at.button))
        at.file_uploader[0].upload("scan.png", png_bytes(scan()), "image/png").run()
        self.assertFalse(at.exception, at.exception)
        text = self.text(at)
        self.assertIn("Tumour detected", text)
        self.assertIn("Reference library unavailable", text)
        self.assertIn("The analysis above is unaffected", text)

    def test_reference_search_failure_keeps_the_prediction(self):
        with mock.patch.object(inference, "EMBEDDING_NAME", "a-different-feature"):
            at = self.app()
            at.file_uploader[0].upload("scan.png", png_bytes(scan()), "image/png").run()
            self.assertFalse(at.exception, at.exception)
            text = self.text(at)
        self.assertIn("Tumour detected", text)
        self.assertIn("Reference search failed", text)

    def test_heat_map_failure_keeps_prediction_and_references(self):
        class NoConv:
            layers = []

            def predict(self, *_a, **_k):
                return np.array([[0.9]])

        self.models = CountingModels(NoConv(), tiny_type(1))
        at = self.app()
        at.file_uploader[0].upload("scan.png", png_bytes(scan()), "image/png").run()
        self.assertFalse(at.exception, at.exception)
        text = self.text(at)
        self.assertIn("Heat map unavailable", text)
        self.assertIn("Tumour detected", text)
        self.assertIn("Rank 1", text)
        self.assertEqual(len(at.segmented_control), 0)


if __name__ == "__main__":
    unittest.main()

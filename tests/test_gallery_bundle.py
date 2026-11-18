"""End-to-end: synthetic mirror files -> gallery bundle -> search, upload and Streamlit page."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from retrieval.bundle import BundleError, GalleryBundle
from retrieval.data import ManifestError
from retrieval.experiment import build_upenn_gallery
from retrieval.query import detect_role, files_from_zip, prepare_query
from retrieval.upenn import build_inventory
from tests.upenn_fixtures import AFFINE, assets_for, participants_for, write_subject

NUMBERS = list(range(2, 16))
FOLLOWUPS = [3, 7]


class GalleryBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.raw = root / "raw"
        cls.raw.mkdir()
        for n in NUMBERS:
            write_subject(cls.raw, n)
        write_subject(cls.raw, 40, roles=("t1", "t1gd", "t2", "flair"))  # incomplete: no mask
        bad = write_subject(cls.raw, 41)
        nib.save(nib.Nifti1Image(np.zeros((32, 32, 20), np.uint8), AFFINE), bad["mask"])  # empty mask
        cls.inventory = build_inventory(assets_for(cls.raw, NUMBERS, followups=FOLLOWUPS))
        cls.participants = participants_for(NUMBERS, followups=FOLLOWUPS + [20])
        cls.out = root / "bundle"
        cls.manifest = build_upenn_gallery(cls.inventory, cls.participants, cls.raw, cls.out, seed=7, progress=None)
        cls.bundle = GalleryBundle.load(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _files(self, number: int) -> dict[str, tuple[str, bytes]]:
        roles = {"t1": "T1w", "t1gd": "ce-gd_T1w", "t2": "T2w", "flair": "FLAIR", "mask": "seg"}
        return {r: (f"sub-{number:03d}_{s}.nii.gz", (self.raw / f"sub-{number:03d}_{s}.nii.gz").read_bytes()) for r, s in roles.items()}

    # ----------------------------------------------------------------- build + bundle
    def test_counts_exclusions_and_split_integrity(self):
        counts = self.manifest["counts"]
        self.assertEqual(counts["passed_validation"], len(NUMBERS))
        self.assertEqual(counts["gallery"] + counts["validation"] + counts["test"], len(NUMBERS))
        exclusions = pd.read_csv(self.out / "reports" / "exclusions.csv")
        reasons = dict(zip(exclusions["subject"], exclusions["reason"]))
        self.assertIn("mask", reasons["sub-040"])
        self.assertIn("empty", reasons["sub-041"])
        self.assertTrue(set(self.bundle.demo_features.index).isdisjoint(self.bundle.index.case_ids))
        self.assertEqual(self.manifest["sanity"]["test_cases_in_gallery"], [])
        self.assertEqual(self.manifest["sanity"]["gallery_self_retrieval_rate"], 1.0)
        self.assertTrue(self.manifest["subject_mapping_check"]["passed"])

    def test_scaler_fitted_on_gallery_only(self):
        gallery = self.bundle.index.raw_features
        np.testing.assert_allclose(self.bundle.index.space.mean, gallery.mean().to_numpy(), atol=1e-9)
        self.assertEqual(self.bundle.index.space.fitted_on, len(gallery))

    def test_bundle_keeps_secondary_labels_out_of_display_and_ranking(self):
        metadata_columns = set(pd.read_csv(self.out / "gallery_metadata.csv").columns)
        feature_columns = set(self.bundle.index.space.columns)
        for forbidden in ("idh1", "mgmt", "survival_days", "kps", "gtr_over_90pct"):
            self.assertNotIn(forbidden, metadata_columns)
            self.assertFalse(any(forbidden in c for c in feature_columns))
        self.assertTrue((self.out / "evaluation" / "secondary_labels.csv").is_file())
        self.assertIsNone(self.manifest["similarity_cutoff"])
        self.assertIsNone(self.bundle.cutoff)

    def test_previews_exist_for_gallery_and_demo(self):
        for case_id in list(self.bundle.index.case_ids) + list(self.bundle.demo_features.index):
            self.assertIsNotNone(self.bundle.preview_path(case_id), case_id)

    def test_no_raw_scans_in_bundle(self):
        self.assertEqual(list(self.out.rglob("*.nii*")), [])

    # ----------------------------------------------------------------- search
    def test_demo_query_search(self):
        case_id = self.bundle.demo_features.index[0]
        results, excluded = self.bundle.search(self.bundle.demo_features.loc[[case_id]], top_k=3, exclude_patient_id=case_id)
        self.assertEqual(len(results), 3)
        self.assertTrue(results["similarity"].is_monotonic_decreasing)
        self.assertEqual(excluded, [])

    def test_query_patient_is_excluded(self):
        gallery_case = self.bundle.index.case_ids[0]
        features = self.bundle.index.raw_features.loc[[gallery_case]]
        results, excluded = self.bundle.search(features, top_k=len(self.bundle.index.case_ids))
        self.assertIn(gallery_case, excluded)  # identical examination
        self.assertNotIn(gallery_case, results["case_id"].tolist())
        results, excluded = self.bundle.search(self.bundle.demo_features.iloc[[0]], top_k=50, exclude_patient_id=gallery_case)
        self.assertNotIn(gallery_case, results["case_id"].tolist())

    def test_validated_cutoff_gives_no_close_matches(self):
        copy = Path(self.tmp.name) / "bundle_cutoff"
        shutil.copytree(self.out, copy)
        header = json.loads((copy / "bundle.json").read_text())
        header["similarity_cutoff"] = {"threshold": 0.9999, "validated_on": "test"}
        (copy / "bundle.json").write_text(json.dumps(header))
        bundle = GalleryBundle.load(copy)
        results, _ = bundle.search(bundle.demo_features.iloc[[0]], top_k=5)
        self.assertTrue(results.empty)

    # ----------------------------------------------------------------- upload path
    def test_upload_query_matches_precomputed_features(self):
        test_case = self.bundle.demo_features.index[0]
        prepared = prepare_query(self._files(int(test_case[4:])))
        self.assertEqual(prepared.patient_hint, test_case)
        np.testing.assert_allclose(
            prepared.features[self.bundle.index.space.columns].to_numpy(),
            self.bundle.demo_features.loc[[test_case]].to_numpy(),
            rtol=1e-6, atol=1e-6,
        )
        self.bundle.check_geometry(prepared.shape, prepared.affine)

    def test_zip_upload_and_role_detection(self):
        files = self._files(NUMBERS[0])
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, payload in files.values():
                archive.writestr(f"exam/{name}", payload)
        found = files_from_zip(buffer.getvalue())
        self.assertEqual(set(found), {"t1", "t1gd", "t2", "flair", "mask"})
        self.assertEqual(detect_role("BraTS_t1ce.nii.gz"), "t1gd")
        self.assertEqual(detect_role("x_T1w.nii.gz"), "t1")
        self.assertEqual(detect_role("x_seg.nii.gz"), "mask")
        self.assertIsNone(detect_role("notes.txt"))

    def test_missing_input_is_rejected(self):
        files = self._files(NUMBERS[0])
        del files["flair"]
        with self.assertRaisesRegex(ManifestError, "FLAIR"):
            prepare_query(files)

    def test_misaligned_input_is_rejected(self):
        files = self._files(NUMBERS[0])
        image = nib.load(str(self.raw / files["t2"][0]))
        moved = AFFINE.copy()
        moved[2, 3] += 3
        buffer = io.BytesIO()
        file_map = nib.Nifti1Image(np.asanyarray(image.dataobj), moved).to_bytes()
        files["t2"] = ("moved_T2w.nii", file_map)
        with self.assertRaisesRegex(ManifestError, "not aligned"):
            prepare_query(files)

    def test_wrong_atlas_grid_is_rejected(self):
        with self.assertRaises(BundleError):
            self.bundle.check_geometry((240, 240, 155), AFFINE)

    # ----------------------------------------------------------------- unavailable gallery
    def test_unavailable_or_broken_bundle(self):
        with self.assertRaisesRegex(BundleError, "not found"):
            GalleryBundle.load(Path(self.tmp.name) / "does-not-exist")
        broken = Path(self.tmp.name) / "broken"
        broken.mkdir(exist_ok=True)
        (broken / "bundle.json").write_text("{}")
        with self.assertRaisesRegex(BundleError, "incomplete"):
            GalleryBundle.load(broken)
        old = Path(self.tmp.name) / "old_schema"
        shutil.copytree(self.out, old)
        header = json.loads((old / "bundle.json").read_text())
        header["schema_version"] = 99
        (old / "bundle.json").write_text(json.dumps(header))
        with self.assertRaisesRegex(BundleError, "schema"):
            GalleryBundle.load(old)

    # ----------------------------------------------------------------- Streamlit page
    def _app(self, bundle_dir: str):
        from streamlit.testing.v1 import AppTest

        def page():
            from retrieval.ui import render_retrieval_tab

            render_retrieval_tab()

        os.environ["GALLERY_BUNDLE_DIR"] = bundle_dir
        os.environ.pop("GALLERY_BUNDLE_URL", None)
        return AppTest.from_function(page, default_timeout=60)

    def test_streamlit_demo_query_renders_results(self):
        app = self._app(str(self.out))
        app.run()
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(any("Gallery loaded" in s.value for s in app.success))
        markdown = " ".join(m.value for m in app.markdown)
        self.assertIn("#1", markdown)
        self.assertIn("Why it matched", markdown)

    def test_streamlit_unavailable_gallery_shows_error(self):
        app = self._app(str(Path(self.tmp.name) / "missing_bundle"))
        app.run()
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(any("unavailable" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()

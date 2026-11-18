"""2D reference gallery: slice export, verification, index, search and exclusions (no TensorFlow)."""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from retrieval.experiment import build_upenn_gallery
from retrieval.gallery2d import (
    Gallery2D,
    Gallery2DError,
    build_index,
    pixel_sha256,
    prepare_slices,
    read_bundle_cases,
    render_slice,
)
from retrieval.upenn import build_inventory
from tests.upenn_fixtures import assets_for, participants_for, write_subject

NUMBERS = list(range(2, 18))
FEATURE = "test-feature"


def toy_embedder(image: Image.Image) -> np.ndarray:
    """Deterministic image-only feature: a coarse 8x8 intensity grid."""
    return np.asarray(image.convert("L").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32).ravel() + 1.0


class Gallery2DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.raw = root / "raw"
        cls.raw.mkdir()
        for n in NUMBERS:
            write_subject(cls.raw, n)
        inventory = build_inventory(assets_for(cls.raw, NUMBERS, followups=[3, 7]))
        cls.bundle = root / "bundle"
        build_upenn_gallery(inventory, participants_for(NUMBERS, followups=[3, 7, 20]), cls.raw, cls.bundle, seed=7, progress=None)
        cls.fetched: list[str] = []

        def fetch(url: str) -> bytes:  # stands in for the mirror download
            cls.fetched.append(url.rsplit("/", 1)[-1])
            return (cls.raw / url.rsplit("/", 1)[-1]).read_bytes()

        cls.fetch = staticmethod(fetch)
        cls.gallery_dir = root / "gallery_2d"
        cls.table = prepare_slices(cls.bundle, cls.gallery_dir, workers=2, fetch=fetch, progress=None)
        build_index(cls.gallery_dir, toy_embedder, FEATURE)
        cls.gallery = Gallery2D.load(cls.gallery_dir, expected_feature=FEATURE)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ----------------------------------------------------------------- preparation
    def test_export_uses_clean_t1gd_and_mask_only(self):
        self.assertTrue(all(name.endswith(("_ce-gd_T1w.nii.gz", "_seg.nii.gz")) for name in self.fetched))
        self.assertEqual(list(self.gallery_dir.rglob("*.nii*")), [])  # no raw scans written

    def test_every_slice_is_verified_against_mask_and_ids(self):
        self.assertTrue(self.table["verified"].all())
        cases = read_bundle_cases(self.bundle)
        self.assertTrue(cases["id_consistent"].all())
        self.assertEqual(set(cases["role"]), {"reference", "demo"})

    def test_unverified_join_is_not_searchable(self):
        copy = Path(self.tmp.name) / "bundle_bad"
        shutil.copytree(self.bundle, copy)
        metadata = pd.read_csv(copy / "gallery_metadata.csv")
        metadata.loc[0, "participant_id"] = "UPENN-GBM-09999_11"  # wrong patient linked to this image
        metadata.loc[1, "tumour_volume_ml"] += 5.0  # metadata that does not match the supplied mask
        metadata.to_csv(copy / "gallery_metadata.csv", index=False)
        out = Path(self.tmp.name) / "gallery_bad"
        table = prepare_slices(copy, out, workers=1, fetch=self.fetch, progress=None)
        bad = set(metadata.loc[[0, 1], "case_id"])
        self.assertEqual(set(table.loc[~table["verified"], "case_id"]), bad)
        build_index(out, toy_embedder, FEATURE)
        self.assertTrue(bad.isdisjoint(Gallery2D.load(out).case_ids))

    def test_resume_skips_finished_cases(self):
        before = len(self.fetched)
        prepare_slices(self.bundle, self.gallery_dir, workers=1, fetch=self.fetch, progress=None)
        self.assertEqual(len(self.fetched), before)

    def test_render_slice_is_8bit_and_keeps_shape(self):
        image = render_slice(np.arange(20 * 30, dtype=np.float32).reshape(20, 30) + 1)
        self.assertEqual(image.mode, "L")
        self.assertEqual(image.size, (20, 30))  # (x, y) array -> width x, height y

    # ----------------------------------------------------------------- index and search
    def test_held_out_cases_are_never_indexed(self):
        demo = set(self.gallery.demo_cases())
        self.assertTrue(demo)
        self.assertTrue(demo.isdisjoint(self.gallery.case_ids))
        splits = pd.read_csv(self.bundle / "reports" / "splits.csv").set_index("case_id")["split"]
        self.assertTrue((splits.loc[self.gallery.case_ids] == "gallery").all())
        self.assertTrue((splits.loc[sorted(demo)] == "test").all())

    def test_search_ranks_by_similarity_with_linked_metadata(self):
        demo_id = self.gallery.demo_cases()[0]
        image = Image.open(self.gallery.demo_path(demo_id)).convert("RGB")
        result = self.gallery.search(toy_embedder(image), image, top_k=5)
        self.assertEqual(len(result.matches), 5)
        self.assertEqual(result.excluded, [])
        scores = [m.similarity for m in result.matches]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([m.rank for m in result.matches], [1, 2, 3, 4, 5])
        bundle_meta = pd.read_csv(self.bundle / "gallery_metadata.csv").set_index("case_id")
        for match in result.matches:
            row = bundle_meta.loc[match.case_id]
            self.assertTrue(match.image_path.is_file())
            self.assertEqual(match.details["Source ID"], row["participant_id"])
            self.assertEqual(match.details["Tumour volume (from supplied mask)"], f"{row['tumour_volume_ml']:.1f} mL")
            self.assertEqual(match.details["Documented diagnosis"], "Glioblastoma")

    def test_exact_copy_of_a_reference_is_not_its_own_match(self):
        case_id = self.gallery.case_ids[0]
        image = Image.open(self.gallery_dir / "reference" / f"{case_id}.png")
        result = self.gallery.search(toy_embedder(image), image.convert("RGB"), top_k=5)
        self.assertIn(case_id, result.excluded)
        self.assertNotIn(case_id, [m.case_id for m in result.matches])
        # Same pixels re-saved in another PNG encoding, with unrelated features: caught by the pixel hash.
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG", compress_level=1)
        copy = Image.open(io.BytesIO(buffer.getvalue()))
        self.assertEqual(pixel_sha256(copy), pixel_sha256(image))
        other = self.gallery.embeddings[5]
        self.assertIn(case_id, self.gallery.search(other, copy, top_k=3).excluded)

    def test_missing_fields_are_omitted_not_filled(self):
        cases = self.gallery.cases.set_index("case_id")
        missing_mgmt = [c for c in self.gallery.case_ids if pd.isna(cases.loc[c, "mgmt"])]
        present_mgmt = [c for c in self.gallery.case_ids if not pd.isna(cases.loc[c, "mgmt"])]
        self.assertTrue(missing_mgmt and present_mgmt)
        self.assertNotIn("MGMT promoter", self.gallery.details(missing_mgmt[0]))
        self.assertIn("MGMT promoter", self.gallery.details(present_mgmt[0]))
        for value in self.gallery.details(missing_mgmt[0]).values():
            self.assertNotIn("nan", value.lower())

    # ----------------------------------------------------------------- failure modes
    def test_missing_gallery_and_index_messages(self):
        with self.assertRaisesRegex(Gallery2DError, "retrieval.gallery2d prepare"):
            Gallery2D.load(Path(self.tmp.name) / "nowhere")
        partial = Path(self.tmp.name) / "partial"
        shutil.copytree(self.gallery_dir, partial)
        (partial / "index.npz").unlink()
        with self.assertRaisesRegex(Gallery2DError, "retrieval.gallery2d index"):
            Gallery2D.load(partial)
        with self.assertRaisesRegex(Gallery2DError, "built with"):
            Gallery2D.load(self.gallery_dir, expected_feature="another-feature")

    def test_index_containing_a_held_out_case_is_rejected(self):
        leaky = Path(self.tmp.name) / "leaky"
        shutil.copytree(self.gallery_dir, leaky)
        demo_id = self.gallery.demo_cases()[0]
        np.savez_compressed(
            leaky / "index.npz",
            case_ids=np.asarray(self.gallery.case_ids + [demo_id]),
            embeddings=np.vstack([self.gallery.embeddings, self.gallery.embeddings[:1]]),
            feature_name=np.asarray(FEATURE),
        )
        with self.assertRaisesRegex(Gallery2DError, "must not be searchable"):
            Gallery2D.load(leaky)

    def test_wrong_embedding_size_is_reported(self):
        with self.assertRaisesRegex(Gallery2DError, "rebuild"):
            self.gallery.search(np.ones(7))


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the audit findings: unscaled features, cancelling weights, metric choice."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize

from retrieval.index import FeatureSpace, FittedFuser, RetrievalIndex, legacy_unscaled_similarity


def _gallery():
    # Two features per family (single-feature families would make each family cosine +-1).
    # "same_volume" shares the query's size only; "same_rest" has a smaller tumour but the
    # same intensity and location. Filler cases provide gallery variance.
    columns = ["shape_volume_mm3", "shape_bbox_mm", "intensity_mean", "intensity_p90", "location_x_mm", "location_y_mm"]
    rows = {
        "same_volume": [30000.0, 45.0, -2.0, -1.0, -40.0, 30.0],
        "same_rest": [24000.0, 40.0, 1.0, 2.0, 5.0, -10.0],
        "filler_1": [10000.0, 25.0, 0.0, 0.5, 0.0, 0.0],
        "filler_2": [50000.0, 60.0, 0.5, 1.5, 20.0, 15.0],
        "filler_3": [15000.0, 30.0, -1.0, 0.0, -10.0, -20.0],
        "filler_4": [20000.0, 35.0, 1.5, 2.5, -25.0, 5.0],
    }
    gallery = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    query = pd.DataFrame([[30000.0, 45.0, 1.0, 2.0, 5.0, -10.0]], index=["query"], columns=columns)
    return gallery, query


class UnscaledFeatureTests(unittest.TestCase):
    def test_legacy_cosine_cannot_discriminate(self):
        """Raw volume (~10^4) swamps the other features, so every case scores ~1.0."""
        gallery, query = _gallery()
        legacy = legacy_unscaled_similarity(gallery, query)[:, 0]
        self.assertGreater(legacy.min(), 0.999)
        self.assertLess(legacy.max() - legacy.min(), 1e-3)
        fixed = RetrievalIndex.fit(gallery).query(query, top_k=len(gallery))["similarity"]
        self.assertGreater(fixed.max() - fixed.min(), 0.5)

    def test_fixed_ranking_uses_all_families(self):
        gallery, query = _gallery()
        index = RetrievalIndex.fit(gallery)
        result = index.query(query, top_k=2)
        self.assertEqual(result.iloc[0]["case_id"], "same_rest")

    def test_scaler_is_fitted_on_gallery_only(self):
        gallery, query = _gallery()
        space = FeatureSpace.fit(gallery)
        np.testing.assert_allclose(space.mean, gallery.mean().to_numpy())
        np.testing.assert_allclose(space.scale, gallery.std(ddof=0).to_numpy())
        before = space.transform(gallery)
        space.transform(pd.DataFrame([{c: 1e9 for c in gallery.columns}], index=["outlier"]))
        np.testing.assert_allclose(space.transform(gallery), before)

    def test_constant_feature_does_not_crash_or_dominate(self):
        gallery, query = _gallery()
        gallery["shape_constant"] = 7.0
        query["shape_constant"] = 7.0
        result = RetrievalIndex.fit(gallery).query(query, top_k=1)
        self.assertTrue(np.isfinite(result.iloc[0]["similarity"]))


class FusionWeightTests(unittest.TestCase):
    def test_old_formula_cancels_weights(self):
        """Document the bug: weight * block, then L2 per block, divides the weight out."""
        rng = np.random.default_rng(0)
        block = StandardScaler().fit_transform(rng.normal(size=(6, 3)))
        np.testing.assert_allclose(normalize(block * 1.0), normalize(block * 10.0))

    def test_weights_now_change_distance_as_weighted_mean(self):
        gallery, query = _gallery()
        space = FeatureSpace.fit(gallery)
        z_g, z_q = space.standardize(gallery), space.standardize(query).iloc[0]
        d2 = {f: ((z_g[c] - z_q[c]) ** 2).mean(axis=1).to_numpy() for f, c in space.families.items()}
        for weights in ({"shape": 1, "intensity": 1, "location": 1}, {"shape": 5, "intensity": 1, "location": 0.5}):
            weighted = space.with_weights(weights)
            sims = weighted.similarity(weighted.transform(gallery), weighted.transform(query)[0])
            expected = np.exp(-sum(weights[f] * d2[f] for f in d2) / sum(weights.values()) / 2)
            np.testing.assert_allclose(sims, expected, atol=1e-10)
        heavy_shape = RetrievalIndex.fit(gallery, weights={"shape": 1000, "intensity": 1, "location": 1}).query(query, top_k=1)
        self.assertEqual(heavy_shape.iloc[0]["case_id"], "same_volume")

    def test_cosine_option_is_weighted_mean_of_family_cosines(self):
        gallery, query = _gallery()
        space = FeatureSpace.fit(gallery, metric="cosine")
        z_g, z_q = space.standardize(gallery), space.standardize(query).iloc[0]
        cos = {}
        for family, columns in space.families.items():
            g, q = z_g[columns].to_numpy(), z_q[columns].to_numpy()
            cos[family] = (g @ q) / (np.linalg.norm(g, axis=1) * np.linalg.norm(q))
        weights = {"shape": 5, "intensity": 1, "location": 0.5}
        weighted = space.with_weights(weights)
        sims = weighted.similarity(weighted.transform(gallery), weighted.transform(query)[0])
        np.testing.assert_allclose(sims, sum(weights[f] * cos[f] for f in cos) / sum(weights.values()), atol=1e-10)

    def test_cosine_on_z_scores_ignores_magnitude_but_euclidean_does_not(self):
        """Why euclidean is the default: 'twice as abnormal' is not 'identical'."""
        gallery, query = _gallery()
        space = FeatureSpace.fit(gallery, metric="cosine")
        z = space.standardize(query).to_numpy()[0]
        doubled = pd.DataFrame([space.mean + 2 * z * space.scale], columns=gallery.columns, index=["doubled"])
        cosine_index = RetrievalIndex.fit(pd.concat([gallery, doubled]), space=space)
        self.assertAlmostEqual(cosine_index.query(query, top_k=1).iloc[0]["similarity"], 1.0, places=10)
        euclid = RetrievalIndex.fit(pd.concat([gallery, doubled]), space=FeatureSpace.fit(gallery))
        self.assertLess(euclid.query(query, top_k=len(gallery) + 1).set_index("case_id").loc["doubled", "similarity"], 0.9)

    def test_fitted_fuser_weights_matter(self):
        gallery, _ = _gallery()
        families = {"a": gallery[["shape_volume_mm3"]], "b": gallery[["intensity_mean", "location_x_mm"]]}
        equal = FittedFuser.fit(families).transform(families).to_numpy()
        skewed = FittedFuser.fit(families, {"a": 9.0, "b": 1.0}).transform(families).to_numpy()
        self.assertFalse(np.allclose(equal @ equal.T, skewed @ skewed.T))


class IndexBehaviourTests(unittest.TestCase):
    def test_family_terms_sum_to_distance_and_explain(self):
        gallery, query = _gallery()
        index = RetrievalIndex.fit(gallery)
        result = index.query(query, top_k=6)
        columns = [c for c in result.columns if c.startswith("distance_")]
        self.assertEqual(len(columns), 3)
        np.testing.assert_allclose(result[columns].sum(axis=1), -2 * np.log(result["similarity"]), atol=1e-10)
        explanation = index.explain(query, "same_rest", top_n=10)
        d2 = -2 * np.log(result.set_index("case_id").loc["same_rest", "similarity"])
        self.assertAlmostEqual(explanation["term"].sum(), d2, places=10)
        self.assertEqual(explanation.iloc[0]["feature"].split("_")[0], "shape")  # where they differ most
        cosine = RetrievalIndex.fit(gallery, metric="cosine").query(query, top_k=6)
        cols = [c for c in cosine.columns if c.startswith("contribution_")]
        np.testing.assert_allclose(cosine[cols].sum(axis=1), cosine["similarity"], atol=1e-10)

    def test_patient_exclusion_removes_every_exam_of_that_patient(self):
        gallery, query = _gallery()
        metadata = pd.DataFrame({"patient_id": ["p1", "p1", "p2", "p3", "p4", "p5"]}, index=gallery.index)
        index = RetrievalIndex.fit(gallery, metadata)
        result = index.query(query, top_k=5, exclude_patient_id="p1")
        self.assertNotIn("same_rest", result["case_id"].tolist())
        self.assertNotIn("same_volume", result["case_id"].tolist())
        self.assertEqual(len(result), 4)

    def test_min_similarity_cutoff(self):
        gallery, query = _gallery()
        result = RetrievalIndex.fit(gallery).query(query, top_k=5, min_similarity=2.0)
        self.assertTrue(result.empty)

    def test_save_load_round_trip(self):
        gallery, query = _gallery()
        index = RetrievalIndex.fit(gallery, pd.DataFrame({"patient_id": list("abcdef")}, index=gallery.index), weights={"shape": 2})
        with tempfile.TemporaryDirectory() as tmp:
            index.save(tmp)
            loaded = RetrievalIndex.load(tmp)
        pd.testing.assert_frame_equal(index.query(query, top_k=5), loaded.query(query, top_k=5))


if __name__ == "__main__":
    unittest.main()

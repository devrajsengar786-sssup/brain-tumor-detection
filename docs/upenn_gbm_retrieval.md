# UPENN-GBM similar-case retrieval: milestone 1 design notes

This note covers what the real-data baseline does, why each decision was made, and what is still open.
The code is in `retrieval/`, the cloud build is `notebooks/upenn_gbm_gallery_colab.ipynb`, and the
tests are in `tests/`.

## 1. Data source and what was verified (2026-11-15)

| Item | Finding | How it was checked |
|---|---|---|
| Mirror listing | 4,424 files, 630 subjects (`sub-001` … `sub-630`), plus `participants.tsv` and `qc.tsv` | GitHub release API |
| Complete baseline cases | **147** subjects have T1w, ce-gd_T1w, T2w, FLAIR **and** seg; **1.60 GB** compressed | `retrieval.upenn.build_inventory` |
| Incomplete | 483 subjects have no `seg` file (one of them also lacks T2w), so they are excluded and reported | inventory report |
| Follow-ups | 41 subjects have `sub-NNN_2_*` follow-up files; these are not used (one baseline exam per patient) | inventory |
| Geometry | 240 × 240 × 155 voxels, 1 mm isotropic, identical affine for all five files of a sampled case | NIfTI headers read from the first 64 KB of each file over HTTP (no scan saved to disk) |
| Stored type | Image voxels are stored as `uint8` in this mirror (probably rescaled from the original release). Intensities are therefore z-scored per sequence, and absolute values are not used. | header |
| Mask labels | **{0, 1, 2, 3}**, not the BraTS {0, 1, 2, 4}. Sample `sub-002`: 1 = 15,366, 2 = 168,592, 3 = 27,793 voxels | downloaded mask (42 KB) |
| Label meaning | Assumed 1 = necrotic core, 2 = oedema, 3 = enhancing. The notebook **tests** this on all cases: enhancing must be brightest on T1-Gd and necrosis darkest, or the run stops. | `label_semantics_check` |
| `participants.tsv` | 671 rows, 630 patients; 611 `_11` (baseline) rows and 60 `_21` (follow-up) rows; patients 612–630 have only `_21` rows | pandas |

### The `sub-002` → `UPENN-GBM-00002_11` join
The numeric match is only a candidate. It is corroborated by an independent test. The mirror marks which
subjects have follow-up scans, and `participants.tsv` marks which patients have `_21` rows. **All 41 mirror
follow-up subjects map to one of the 60 `_21` rows.** If the two numberings were unrelated, that would happen
with hypergeometric probability C(60,41)/C(630,41) ≈ 4 × 10⁻⁵⁰ (this check was run on the real listing). All 147 segmented subjects have
an `_11` row. A join is marked `join_verified` only if this test passes **and** exactly one `_11` row exists.
Unverified cases show no metadata. Fields reading "Not Available" or "NA" are recorded as missing per case
(`reports/metadata_join_report.csv`).

*Viva phrasing:* "the mapping is corroborated by an independent consistency test, not assumed".
Stronger proof would compare our mask volumes with the TCIA radiomics CSV (keyed by `UPENN-GBM-NNNNN_11`).
That is a listed next step.

## 2. Pipeline

1. **Inventory**: one API request, which classifies each file by role. Only complete baseline cases go forward.
2. **Download**: resumable (`.part` file plus HTTP Range), and a file is accepted only when its size matches the listing. Files go to the Colab VM disk, not Drive or your laptop. Failures are reported, not fatal.
3. **Validate**: every file is opened, all five must share shape and affine, and the mask must have labels within {0..4}, be non-empty, and not be binary.
4. **Split by patient**: 70 / 15 / 15 into *gallery* (reference), *validation queries* and *test queries*, with seed 42. Each mirror subject is one patient with one baseline exam, so no patient appears in two splits.
5. **Features** (36, in three families, all documented formulas in `retrieval/features.py`):
   * *shape* (9): log volume, necrotic/oedema/enhancing fractions, sphericity, bounding-box fill, elongation, flatness, number of foci
   * *location* (4): tumour centroid offset from the brain centre (x, y, z in mm) and distance from the midline
   * *intensity* (23): mean, spread, p10, median and p90 inside the tumour for each of T1/T1-Gd/T2/FLAIR (z-scored per scan), plus enhancing-rim T1-Gd, necrotic-core T1-Gd and oedema FLAIR means
6. **Fit on the gallery only**: z-score mean and std are computed from gallery cases and frozen. Validation and test queries, and anything uploaded in the app, use that same transform.
7. **Search**: weighted standardised Euclidean distance (section 3), excluding the query's own patient.
8. **Export**: the bundle (features, scaler, display metadata, previews, demo queries, manifest, reports). Secondary labels go in `evaluation/`, which the app never reads.

## 3. The audit findings and the fixes

**(a) Unscaled features.** The old index took the cosine of raw vectors. Volume in mm³ (~10⁴–10⁵) swamped everything else.
On the synthetic test cohort the largest feature took about 98% of each vector's squared norm (`dominant_feature_share`),
so every case scored about 1.0 and the ranking could not discriminate.
**Fix:** z-score every feature with gallery statistics. Test: `test_legacy_cosine_cannot_discriminate`. The notebook
reports the same diagnostic on the real gallery.

**(b) Fusion weights cancelling.** The old fuser computed `normalize(scaler(x) * w)`. L2-normalising after multiplying
divides `w` out, so every weight setting gave identical rankings. **Fix:** normalise each family first, then apply
the weight. Tests: `test_old_formula_cancels_weights`, `test_weights_now_change_distance_as_weighted_mean`.

**(c) Found while fixing (b): cosine measures direction, not closeness.** On z-scored features, cosine compares the
*direction* of deviation from the gallery average. A tumour twice as far from average in the same direction
gets cosine 1. For "find a similar tumour" that is wrong. **Default metric:**

```
d_f^2 = mean over features in family f of (z_query - z_ref)^2
D^2   = sum_f w_f d_f^2 / sum_f w_f       (weights 1:1:1, fixed before any results)
similarity = exp(-D^2 / 2)                 (1 = identical; ~0.37 = a typical unrelated pair)
```

Averaging *within* a family means a family's influence does not depend on its number of features (otherwise
intensity, with 23 features, would outvote location, with 4). Averaging *across* families with weights makes the weights
mean what they say. Because D² is a sum, every match has an exact explanation: family distances add up
to D², and per-feature terms show which features made the two tumours alike or different.
Cosine stays available as `metric="cosine"` for the representation comparison.
Test: `test_cosine_on_z_scores_ignores_magnitude_but_euclidean_does_not`.

**(d) Other fixes.** The query patient is now excluded (previously only the exact case ID). Also, the nnU-Net
`dataset.json` declared labels {0, 1} while the masks contain {1, 2, 3}, which nnU-Net rejects; it now
declares all three subregions. The index is saved as JSON/CSV instead of a pickle, so a bundle downloaded from
shared storage cannot execute code. Binary masks are rejected, because composition features would be meaningless.

## 4. Evaluation status

* **Independent relevance judgments do not exist yet.** Precision@K, Recall@K and mAP@K are therefore **not reported**.
  `retrieval.evaluation` computes them once a predefined `relevance.json` exists (for example blinded expert ratings,
  or a frozen imaging-criteria protocol written *before* looking at rankings).
* Building relevance from our own features would be circular, so we did not do it.
* **What is reported instead:** example rankings for validation and test queries, sanity checks (every gallery case
  retrieves itself first when not excluded; no test case is in the gallery), and a **secondary** analysis. That analysis asks
  how often the Top-5 shares MGMT or IDH1 status with the query, compared with chance. These labels are never used in ranking.
  Expect the IDH1 result to be uninformative: about 81% of `participants.tsv` rows are wildtype and only about 3% are mutated.
* **No similarity cutoff is validated**, so the app always shows the Top-K and never claims "no close match".
  `calibrate_cutoff` fits one on *validation* queries once relevance labels exist. If a bundle carries a cutoff, the
  app shows "No close matches" when nothing passes it.

## 5. What remains (final milestone)

1. Train nnU-Net v2 on **gallery patients only** (same `reports/splits.csv`), and report Dice and HD95 per subregion on test patients.
2. Generate masks for test queries, then compare retrieval with supplied masks against model masks. Overlap of the Top-K lists
   and rank correlation need no relevance labels; P@K and mAP@K need them.
3. Representation comparisons E1–E5 (radiomics / whole-image deep / tumour-region deep / fusions), all on the same gallery,
   query sets and relevance criteria. `FeatureSpace` families and `run_ablation` support this.
4. Collect the independent relevance judgments, then calibrate a cutoff on validation queries.
5. Optional: confirm the ID mapping against TCIA radiomic volumes, and check preview orientation on real scans.

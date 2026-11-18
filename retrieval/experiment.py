"""Build the real UPENN-GBM gallery bundle from downloaded mirror files (run in Colab)."""

from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .bundle import write_bundle
from .data import CaseRecord, create_patient_splits
from .evaluation import secondary_label_agreement
from .features import FEATURE_SET_VERSION, SUBREGION_LABELS, display_measurements, extract_engineered_features
from .index import RetrievalIndex, dominant_feature_share, legacy_unscaled_similarity
from .previews import preview_png, tumour_slice_index
from .upenn import (
    MRI_ROLES,
    join_metadata,
    local_paths,
    validate_case_files,
    verify_subject_mapping,
)
from .volumes import load_case


# Pre-specified before looking at any results: equal weight per feature family.
DEFAULT_WEIGHTS = {"shape": 1.0, "location": 1.0, "intensity": 1.0}
SPLIT_FRACTIONS = {"gallery": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_NAMES = {"train": "gallery", "validation": "validation", "test": "test"}


def _log(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


def build_upenn_gallery(
    inventory: pd.DataFrame,
    participants: pd.DataFrame,
    raw_dir: str | Path,
    out_dir: str | Path,
    download_report: pd.DataFrame | None = None,
    seed: int = 42,
    weights: Mapping[str, float] | None = None,
    top_k: int = 5,
    progress: Callable[[str], None] | None = print,
) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    weights = dict(weights or DEFAULT_WEIGHTS)
    exclusions: list[dict] = []

    # 1) Inventory: subjects without all five baseline files are excluded up front.
    for _, row in inventory[~inventory["complete"]].iterrows():
        exclusions.append({"subject": row["subject"], "stage": "inventory", "reason": f"missing files: {row['missing_roles']}"})
    candidates = inventory[inventory["complete"]].copy()
    failed_downloads = set()
    if download_report is not None and not download_report.empty:
        failed_downloads = set(download_report.loc[download_report["status"] == "failed", "subject"])
    for subject in sorted(failed_downloads):
        exclusions.append({"subject": subject, "stage": "download", "reason": "one or more files failed to download"})
    candidates = candidates[~candidates["subject"].isin(failed_downloads)]
    _log(progress, f"{len(candidates)} complete, downloaded candidate cases")

    # 2) Validation: open every file, check geometry and mask labels.
    validation_rows = []
    cohort_shape = None
    for _, row in candidates.iterrows():
        result = validate_case_files(row["subject"], local_paths(row, raw_dir), expected_shape=cohort_shape)
        if result.ok and cohort_shape is None:
            cohort_shape = tuple(int(v) for v in result.shape.split("x"))
        validation_rows.append(result.__dict__)
        if not result.ok:
            exclusions.append({"subject": row["subject"], "stage": "validation", "reason": result.reason})
    validation = pd.DataFrame(validation_rows)
    valid_subjects = validation.loc[validation["ok"], "subject"].tolist() if not validation.empty else []
    _log(progress, f"{len(valid_subjects)} cases passed validation")
    if len(valid_subjects) < 3:
        raise RuntimeError("Fewer than three valid cases; cannot create patient-level splits.")

    # 3) Metadata join (verified joins only) — kept separate from the ranking features.
    mapping_check = verify_subject_mapping(inventory, participants)
    joined = join_metadata(valid_subjects, participants, mapping_check).set_index("subject")

    # 4) Patient-level split. Each mirror subject is one patient with one baseline exam.
    by_subject = candidates.set_index("subject")
    records = []
    for subject in valid_subjects:
        paths = local_paths(by_subject.loc[subject], raw_dir)
        records.append(CaseRecord(subject, subject, tuple(paths[r] for r in MRI_ROLES), paths["mask"]))
    records = create_patient_splits(records, SPLIT_FRACTIONS["gallery"], SPLIT_FRACTIONS["validation"], seed)
    split_of = {record.case_id: SPLIT_NAMES[record.split] for record in records}

    # 5) Features, measurements, previews and a label-semantics check, one load per case.
    feature_rows, measurement_rows, previews, semantics = [], [], {}, []
    geometry = None
    for number, record in enumerate(records, start=1):
        try:
            case = load_case(record)
            features = extract_engineered_features(case)
        except Exception as error:
            exclusions.append({"subject": record.case_id, "stage": "features", "reason": str(error)})
            continue
        if geometry is None:
            geometry = {"shape": list(case.mask.shape), "affine": np.round(case.affine, 4).tolist(), "spacing_mm": list(case.spacing_mm)}
        feature_rows.append({"case_id": record.case_id, **features})
        measurement_rows.append({"case_id": record.case_id, "preview_slice": tumour_slice_index(case), **display_measurements(case)})
        previews[record.case_id] = preview_png(case)
        t1gd = case.images[..., 1]
        row = {"case_id": record.case_id}
        for value in (1, 2, 3, 4):
            region = case.labels == value
            row[f"label{value}_t1gd_mean_z"] = float(t1gd[region].mean()) if region.any() else np.nan
            row[f"label{value}_flair_mean_z"] = float(case.images[..., 3][region].mean()) if region.any() else np.nan
        semantics.append(row)
        if number % 20 == 0:
            _log(progress, f"features: {number}/{len(records)}")
    features = pd.DataFrame(feature_rows).set_index("case_id").sort_index()
    measurements = pd.DataFrame(measurement_rows).set_index("case_id")
    semantics = pd.DataFrame(semantics).set_index("case_id")

    # 6) Display metadata (verified joins only) and secondary labels (evaluation only).
    meta = pd.DataFrame(index=features.index)
    meta["patient_id"] = meta.index
    meta["split"] = [split_of[c] for c in meta.index]
    meta["participant_id"] = joined["participant_id"].reindex(meta.index)
    meta["join_verified"] = joined["join_verified"].reindex(meta.index)
    meta["sex"] = joined["sex"].reindex(meta.index).where(meta["join_verified"])
    meta["age_years"] = joined["age_years"].reindex(meta.index).where(meta["join_verified"])
    meta = meta.join(measurements)
    secondary = joined[["idh1", "mgmt", "survival_days", "kps", "gtr_over_90pct", "join_verified"]].reindex(features.index)
    metadata_report = joined.reset_index()[["subject", "participant_id", "join_verified", "join_note", "missing_fields"]]

    gallery_ids = meta.index[meta["split"] == "gallery"]
    validation_ids = meta.index[meta["split"] == "validation"]
    test_ids = meta.index[meta["split"] == "test"]

    # 7) Fit the feature space on the gallery ONLY; queries reuse the frozen transform.
    index = RetrievalIndex.fit(features.loc[gallery_ids], meta.loc[gallery_ids], weights=weights)

    def rankings(ids) -> pd.DataFrame:
        rows = []
        for case_id in ids:
            result = index.query(features.loc[[case_id]], top_k=top_k, exclude_patient_id=case_id)
            for rank, match in enumerate(result.itertuples(), start=1):
                rows.append({"query": case_id, "rank": rank, "match": match.case_id, "similarity": round(match.similarity, 4),
                             **{c: round(getattr(match, c), 4) for c in result.columns if c.startswith(("contribution_", "distance_"))}})
        return pd.DataFrame(rows)

    validation_rankings, test_rankings = rankings(validation_ids), rankings(test_ids)

    # Sanity checks that need no relevance labels.
    self_hits = sum(index.query(features.loc[[c]], top_k=1).iloc[0]["case_id"] == c for c in gallery_ids)
    leaks = [c for c in test_ids if c in index.case_ids]
    secondary_results = {}
    for label in ("mgmt", "idh1"):
        secondary_results[label] = secondary_label_agreement(
            index, features.loc[test_ids], secondary[label].reindex(test_ids), secondary[label].reindex(gallery_ids), k=top_k
        )

    # Audit diagnostic: what the earlier unscaled ranking would have done.
    legacy = features.loc[gallery_ids].copy()
    legacy["legacy_volume_mm3"] = measurements.loc[gallery_ids, "tumour_volume_ml"] * 1000.0
    legacy_test = features.loc[test_ids].copy()
    legacy_test["legacy_volume_mm3"] = measurements.loc[test_ids, "tumour_volume_ml"] * 1000.0
    legacy_sims = legacy_unscaled_similarity(legacy, legacy_test)
    overlap = []
    for j, case_id in enumerate(test_ids):
        legacy_top = set(np.asarray(gallery_ids)[np.argsort(-legacy_sims[:, j])[:top_k]])
        fixed_top = set(test_rankings.loc[test_rankings["query"] == case_id, "match"])
        overlap.append(len(legacy_top & fixed_top) / top_k)
    legacy_volume_corr = []
    for j, case_id in enumerate(test_ids):
        volume_gap = -np.abs(np.log(legacy["legacy_volume_mm3"].to_numpy()) - np.log(legacy_test["legacy_volume_mm3"].iloc[j]))
        legacy_volume_corr.append(pd.Series(legacy_sims[:, j]).corr(pd.Series(volume_gap), method="spearman"))
    audit = {
        "legacy_mean_dominant_feature_share": float(dominant_feature_share(legacy).mean()),
        "fixed_mean_dominant_feature_share": float(dominant_feature_share(index.space.standardize(features.loc[gallery_ids])).mean()),
        "legacy_vs_fixed_topk_overlap": float(np.mean(overlap)) if overlap else None,
        "legacy_similarity_vs_volume_gap_spearman": float(np.nanmean(legacy_volume_corr)) if legacy_volume_corr else None,
    }

    label_semantics = {
        f"label{v}": {
            "cases": int(semantics[f"label{v}_t1gd_mean_z"].notna().sum()),
            "mean_t1gd_z": float(semantics[f"label{v}_t1gd_mean_z"].mean()) if semantics[f"label{v}_t1gd_mean_z"].notna().any() else None,
            "mean_flair_z": float(semantics[f"label{v}_flair_mean_z"].mean()) if semantics[f"label{v}_flair_mean_z"].notna().any() else None,
        }
        for v in (1, 2, 3, 4)
    }

    exclusion_frame = pd.DataFrame(exclusions, columns=["subject", "stage", "reason"])
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": "UPENN-GBM (TCIA DOI 10.7937/TCIA.709X-DN49), GitHub mirror data-nih/tcia release upenn-gbm",
        "feature_set_version": FEATURE_SET_VERSION,
        "label_convention": {k: list(v) for k, v in SUBREGION_LABELS.items()},
        "label_semantics_check": label_semantics,
        "seed": seed,
        "split_fractions": SPLIT_FRACTIONS,
        "split_unit": "patient (one baseline examination per mirror subject)",
        "counts": {
            "mirror_subjects": int(len(inventory)),
            "complete_candidates": int(inventory["complete"].sum()),
            "failed_downloads": len(failed_downloads),
            "passed_validation": len(valid_subjects),
            "with_features": int(len(features)),
            "gallery": int(len(gallery_ids)),
            "validation": int(len(validation_ids)),
            "test": int(len(test_ids)),
            "excluded": int(len(exclusion_frame)),
            "verified_metadata_joins": int(meta["join_verified"].fillna(False).astype(bool).sum()),
        },
        "subject_mapping_check": mapping_check,
        "fusion": {
            "weights": weights,
            "metric": index.space.metric,
            "rule": "z-score (gallery-fitted) -> mean squared difference per family -> weighted mean over families = D^2; similarity = exp(-D^2/2)",
            "family_sizes": {f: len(c) for f, c in index.space.families.items()},
        },
        "ranking_inputs": "image-derived features only; molecular markers and outcomes excluded",
        "sanity": {"gallery_self_retrieval_rate": self_hits / max(len(gallery_ids), 1), "test_cases_in_gallery": leaks},
        "secondary_label_agreement_test": secondary_results,
        "audit_unscaled_vs_fixed": audit,
        "similarity_cutoff": None,
        "similarity_cutoff_note": "No independent relevance judgments exist yet, so no cutoff is validated; the app shows all Top-K.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }

    reports = {
        "download_report": download_report if download_report is not None else pd.DataFrame(),
        "validation_report": validation,
        "exclusions": exclusion_frame,
        "metadata_join_report": metadata_report,
        "splits": meta[["split"]].reset_index(),
        "validation_rankings": validation_rankings,
        "test_rankings": test_rankings,
        "label_semantics_per_case": semantics.reset_index(),
    }
    write_bundle(
        out_dir,
        index,
        meta.loc[gallery_ids],
        {c: previews[c] for c in list(gallery_ids) + list(test_ids)},
        manifest,
        geometry,
        demo_features=features.loc[test_ids],
        demo_metadata=meta.loc[test_ids],
        reports=reports,
        secondary_labels=secondary,
    )
    features.to_csv(out_dir / "evaluation" / "all_features.csv", index_label="case_id")
    _log(progress, f"bundle written to {out_dir}")
    return manifest

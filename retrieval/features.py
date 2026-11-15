"""Mask-derived shape, location and intensity features for the interpretable baseline.

Every ranking feature is a documented formula on the tumour mask and the four
z-scored sequences, so each value shown in the app can be traced back to code.

Feature families (used for block-weighted fusion in ``retrieval.index``):

* ``shape``     – size, subregion composition and geometry of the tumour mask
* ``location``  – where the tumour sits relative to the centre of the brain
* ``intensity`` – how the tumour looks on T1, T1-Gd, T2 and FLAIR
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage

from .data import ManifestError
from .volumes import LoadedCase


FEATURE_MODALITIES = ("t1", "t1gd", "t2", "flair")
FEATURE_SET_VERSION = "engineered-v2"

# UPENN-GBM subregion labels. The GitHub mirror stores enhancing tumour as 3;
# the original BraTS/UPENN convention stores it as 4. Both are accepted.
SUBREGION_LABELS: dict[str, tuple[int, ...]] = {
    "ncr": (1,),  # necrotic / non-enhancing tumour core
    "ed": (2,),  # peritumoural oedema / infiltrated tissue
    "et": (3, 4),  # enhancing tumour
}
ALLOWED_LABELS = frozenset({0, 1, 2, 3, 4})
SUBREGION_NAMES = {"ncr": "necrotic core", "ed": "oedema", "et": "enhancing tumour"}

# Components smaller than this (in mL) are ignored when counting tumour foci.
MIN_FOCUS_ML = 0.1


def validate_label_map(labels: np.ndarray, case_id: str = "case") -> dict[str, int]:
    """Check that a mask is a subregion label map compatible with the gallery."""
    present = {int(value) for value in np.unique(labels)}
    unexpected = present - ALLOWED_LABELS
    if unexpected:
        raise ManifestError(
            f"{case_id}: mask contains unsupported labels {sorted(unexpected)}; expected 0-3 "
            "(0 background, 1 necrotic core, 2 oedema, 3 or 4 enhancing tumour)."
        )
    if {3, 4} <= present:
        raise ManifestError(f"{case_id}: mask mixes labels 3 and 4; use one enhancing-tumour convention.")
    if present <= {0, 1}:
        raise ManifestError(
            f"{case_id}: mask is binary (labels 0/1). The gallery uses subregion masks "
            "(1 necrotic core, 2 oedema, 3 enhancing tumour), so a binary mask would produce "
            "incomparable composition features."
        )
    return {f"label_{value}_voxels": int((labels == value).sum()) for value in sorted(present)}


def _surface_area_mm2(mask: np.ndarray, spacing: np.ndarray) -> float:
    """Area of exposed voxel faces (a simple, deterministic surface estimate)."""
    padded = np.pad(mask, 1).astype(np.int8)
    face_areas = (spacing[1] * spacing[2], spacing[0] * spacing[2], spacing[0] * spacing[1])
    return float(sum(np.abs(np.diff(padded, axis=axis)).sum() * area for axis, area in enumerate(face_areas)))


def _principal_axes(coordinates_mm: np.ndarray) -> np.ndarray:
    if len(coordinates_mm) < 4:
        return np.array([1.0, 1.0, 1.0])
    eigenvalues = np.linalg.eigvalsh(np.cov(coordinates_mm.T))
    return np.sort(np.clip(eigenvalues, 1e-12, None))[::-1]


def _count_foci(mask: np.ndarray, voxel_ml: float) -> int:
    labelled, count = ndimage.label(mask)
    if count == 0:
        return 0
    sizes = np.bincount(labelled.ravel())[1:] * voxel_ml
    return int((sizes >= MIN_FOCUS_ML).sum())


def _subregion_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.isin(labels, values) for name, values in SUBREGION_LABELS.items()}


def extract_engineered_features(case: LoadedCase) -> dict[str, float]:
    """Return the ranking features for one aligned case with a subregion mask."""
    labels = case.labels if case.labels is not None else case.mask.astype(np.int16)
    validate_label_map(labels, case.case_id)
    mask = labels > 0
    spacing = np.asarray(case.spacing_mm, dtype=np.float64)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    voxels = int(mask.sum())
    volume_ml = voxels * voxel_ml

    coordinates = np.argwhere(mask)
    coordinates_mm = coordinates * spacing
    minimum, maximum = coordinates.min(axis=0), coordinates.max(axis=0)
    bbox_voxels = float(np.prod(maximum - minimum + 1))
    surface = _surface_area_mm2(mask, spacing)
    volume_mm3 = volume_ml * 1000.0
    sphericity = (np.pi ** (1 / 3)) * ((6 * volume_mm3) ** (2 / 3)) / surface if surface else 0.0
    axes = _principal_axes(coordinates_mm)
    subregions = _subregion_masks(labels)

    features: dict[str, float] = {
        "shape_log_volume_ml": float(np.log1p(volume_ml)),
        "shape_ncr_fraction": float(subregions["ncr"].sum() / voxels),
        "shape_ed_fraction": float(subregions["ed"].sum() / voxels),
        "shape_et_fraction": float(subregions["et"].sum() / voxels),
        "shape_sphericity": float(sphericity),
        "shape_bbox_fill": float(voxels / bbox_voxels),
        "shape_elongation": float(np.sqrt(axes[1] / axes[0])),
        "shape_flatness": float(np.sqrt(axes[2] / axes[0])),
        "shape_log_foci": float(np.log1p(_count_foci(mask, voxel_ml))),
    }

    # Location relative to the brain centre, in world millimetres, so the features
    # are comparable across cases registered to the same atlas.
    brain = case.brain_mask if case.brain_mask is not None else np.any(case.images != 0, axis=-1)
    brain_centre_world = (case.affine @ np.array([*np.argwhere(brain).mean(axis=0), 1.0]))[:3]
    tumour_centre_world = (case.affine @ np.array([*coordinates.mean(axis=0), 1.0]))[:3]
    offset = tumour_centre_world - brain_centre_world
    features.update(
        {
            "location_x_mm": float(offset[0]),
            "location_y_mm": float(offset[1]),
            "location_z_mm": float(offset[2]),
            "location_abs_x_mm": float(abs(offset[0])),
        }
    )

    for channel_index, modality_name in enumerate(FEATURE_MODALITIES):
        values = case.images[..., channel_index][mask].astype(np.float64)
        p10, median, p90 = np.percentile(values, (10, 50, 90))
        features.update(
            {
                f"intensity_{modality_name}_mean": float(values.mean()),
                f"intensity_{modality_name}_std": float(values.std()),
                f"intensity_{modality_name}_p10": float(p10),
                f"intensity_{modality_name}_median": float(median),
                f"intensity_{modality_name}_p90": float(p90),
            }
        )
    # Subregion contrasts that radiologists look at: enhancement and necrosis on T1-Gd,
    # oedema signal on FLAIR. Absent subregions get the whole-tumour mean (neutral value).
    for name, channel, modality_name in (("et", 1, "t1gd"), ("ncr", 1, "t1gd"), ("ed", 3, "flair")):
        region = subregions[name]
        values = case.images[..., channel][region] if region.any() else case.images[..., channel][mask]
        features[f"intensity_{name}_{modality_name}_mean"] = float(np.mean(values))

    return {name: (0.0 if not np.isfinite(value) else float(value)) for name, value in features.items()}


def display_measurements(case: LoadedCase) -> dict[str, float]:
    """Human-readable measurements shown in the app (not all are ranking features)."""
    labels = case.labels if case.labels is not None else case.mask.astype(np.int16)
    voxel_ml = float(np.prod(case.spacing_mm)) / 1000.0
    subregions = _subregion_masks(labels)
    return {
        "tumour_volume_ml": float((labels > 0).sum() * voxel_ml),
        **{f"{name}_volume_ml": float(region.sum() * voxel_ml) for name, region in subregions.items()},
    }


def feature_family(column: str) -> str:
    """Map a feature column to its fusion family from its prefix."""
    prefix = column.split("_", 1)[0]
    if prefix not in {"shape", "location", "intensity"}:
        raise ValueError(f"Feature '{column}' does not start with a known family prefix.")
    return prefix


def families_for(columns: Iterable[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for column in columns:
        families.setdefault(feature_family(column), []).append(column)
    return families


FEATURE_DESCRIPTIONS = {
    "shape_log_volume_ml": "Tumour volume (log mL)",
    "shape_ncr_fraction": "Necrotic-core share of tumour",
    "shape_ed_fraction": "Oedema share of tumour",
    "shape_et_fraction": "Enhancing-tumour share",
    "shape_sphericity": "Sphericity (1 = sphere)",
    "shape_bbox_fill": "Bounding-box fill ratio",
    "shape_elongation": "Elongation (minor/major axis)",
    "shape_flatness": "Flatness",
    "shape_log_foci": "Number of tumour foci (log)",
    "location_x_mm": "Left-right offset from brain centre (mm)",
    "location_y_mm": "Front-back offset from brain centre (mm)",
    "location_z_mm": "Up-down offset from brain centre (mm)",
    "location_abs_x_mm": "Distance from midline (mm)",
    "intensity_et_t1gd_mean": "Enhancement brightness on T1-Gd",
    "intensity_ncr_t1gd_mean": "Necrotic-core signal on T1-Gd",
    "intensity_ed_flair_mean": "Oedema signal on FLAIR",
}


def describe_feature(column: str) -> str:
    if column in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[column]
    parts = column.split("_")
    if len(parts) == 3 and parts[0] == "intensity":
        modality = {"t1": "T1", "t1gd": "T1-Gd", "t2": "T2", "flair": "FLAIR"}.get(parts[1], parts[1])
        statistic = {"mean": "mean", "std": "spread", "p10": "10th percentile", "median": "median", "p90": "90th percentile"}.get(parts[2], parts[2])
        return f"Tumour {modality} {statistic}"
    return column


def feature_frame(cases: Iterable[LoadedCase]) -> pd.DataFrame:
    """Return one numeric row per case, indexed by the stable case ID."""
    rows = []
    for case in cases:
        row = {"case_id": case.case_id, "patient_id": case.patient_id, "split": case.split or ""}
        row.update(extract_engineered_features(case))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Cannot create features from zero cases.")
    return frame.set_index("case_id", drop=True).sort_index()


def extract_pyradiomics_features(image_path: str, mask_path: str, prefix: str) -> dict[str, float]:
    """
    Optional IBSI-style PyRadiomics expansion for a single modality and mask.

    PyRadiomics is intentionally optional so that the core retrieval pipeline remains
    testable on a lightweight environment. Install requirements-retrieval.txt to use
    this extractor in the formal radiomics baseline (experiment E1).
    """
    try:
        from radiomics import featureextractor
    except ImportError as error:
        raise RuntimeError(
            "PyRadiomics is not installed. Install requirements-retrieval.txt "
            "before running the formal radiomics baseline."
        ) from error

    extractor = featureextractor.RadiomicsFeatureExtractor()
    results = extractor.execute(image_path, mask_path)
    features = {}
    for name, value in results.items():
        if name.startswith("diagnostics_"):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            features[f"{prefix}_{name}"] = numeric_value
    return features

"""Mask-derived morphology and reproducible intensity-feature extraction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage, stats

from .volumes import LoadedCase


FEATURE_MODALITIES = ("t1", "t1gd", "t2", "flair")


def _entropy(values: np.ndarray) -> float:
    histogram, _ = np.histogram(values, bins=64, density=False)
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def extract_engineered_features(case: LoadedCase) -> dict[str, float]:
    """
    Create auditable tumour features from a mask and four normalized modalities.

    These descriptors are a lightweight, deterministic baseline. They intentionally
    remain separate from a full PyRadiomics experiment so that every value shown in
    the app can be traced back to a documented formula.
    """
    mask = case.mask
    coordinates = np.argwhere(mask)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    dimensions_voxels = maximum - minimum + 1
    spacing = np.asarray(case.spacing_mm, dtype=np.float64)
    voxel_volume = float(np.prod(spacing))
    tumor_voxels = int(mask.sum())

    centroid_voxel = coordinates.mean(axis=0)
    centroid_world = case.affine @ np.array([*centroid_voxel, 1.0])
    boundary = mask & ~ndimage.binary_erosion(mask, border_value=0)
    extent_volume = float(np.prod(dimensions_voxels))

    features: dict[str, float] = {
        "morph_volume_mm3": tumor_voxels * voxel_volume,
        "morph_volume_ml": tumor_voxels * voxel_volume / 1000.0,
        "morph_voxel_count": float(tumor_voxels),
        "morph_bbox_extent": tumor_voxels / extent_volume,
        "morph_boundary_voxel_fraction": float(boundary.sum() / tumor_voxels),
        "morph_bbox_x_mm": float(dimensions_voxels[0] * spacing[0]),
        "morph_bbox_y_mm": float(dimensions_voxels[1] * spacing[1]),
        "morph_bbox_z_mm": float(dimensions_voxels[2] * spacing[2]),
        "spatial_centroid_x_mm": float(centroid_world[0]),
        "spatial_centroid_y_mm": float(centroid_world[1]),
        "spatial_centroid_z_mm": float(centroid_world[2]),
    }

    for channel_index, modality_name in enumerate(FEATURE_MODALITIES):
        values = case.images[..., channel_index][mask].astype(np.float64)
        features.update(
            {
                f"{modality_name}_mean": float(values.mean()),
                f"{modality_name}_std": float(values.std()),
                f"{modality_name}_p10": float(np.percentile(values, 10)),
                f"{modality_name}_median": float(np.median(values)),
                f"{modality_name}_p90": float(np.percentile(values, 90)),
                f"{modality_name}_skewness": float(stats.skew(values, bias=False)) if values.size > 2 else 0.0,
                f"{modality_name}_kurtosis": float(stats.kurtosis(values, bias=False)) if values.size > 3 else 0.0,
                f"{modality_name}_entropy": _entropy(values),
            }
        )
    return {name: (0.0 if not np.isfinite(value) else float(value)) for name, value in features.items()}


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
    this extractor in the formal radiomics baseline.
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

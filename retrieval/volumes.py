"""NIfTI loading, geometry checks, and MRI normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .data import CaseRecord, ManifestError


# Tolerance for comparing voxel-to-world affines between sequences of one exam.
AFFINE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class LoadedCase:
    """Aligned, normalized image channels and a tumour label map."""

    case_id: str
    patient_id: str
    images: np.ndarray  # (X, Y, Z, 4), z-scored channel order: T1, T1GD, T2, FLAIR
    mask: np.ndarray  # (X, Y, Z), bool: any tumour label
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]
    split: str | None
    labels: np.ndarray | None = None  # (X, Y, Z), int16 subregion labels as stored in the file
    brain_mask: np.ndarray | None = None  # (X, Y, Z), bool: voxels non-zero in any sequence


def _z_score_nonzero(volume: np.ndarray) -> np.ndarray:
    """Z-score using brain (non-zero) voxels so background does not dominate the statistics."""
    foreground = volume[np.isfinite(volume) & (volume != 0)]
    if foreground.size == 0:
        raise ManifestError("A modality has no finite, non-zero voxels for normalization.")
    mean = float(foreground.mean())
    std = float(foreground.std())
    if std == 0:
        raise ManifestError("A modality has zero intensity variance after foreground selection.")
    normalized = (volume.astype(np.float32) - mean) / std
    normalized[~np.isfinite(normalized)] = 0
    normalized[volume == 0] = 0
    return normalized


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    try:
        image = nib.load(str(path))
        data = image.get_fdata(dtype=np.float32)
    except Exception as error:  # nibabel's exceptions differ by backend
        raise ManifestError(f"Could not load NIfTI file '{Path(path).name}': {error}") from error
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ManifestError(f"'{Path(path).name}' must be a 3D NIfTI volume; received shape {data.shape}.")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return data, image.affine, spacing


def load_case(record: CaseRecord) -> LoadedCase:
    """Load a case only when all modalities and its mask share exact geometry."""
    image_volumes = []
    raw_volumes = []
    reference_shape: tuple[int, ...] | None = None
    reference_affine: np.ndarray | None = None
    reference_spacing: tuple[float, float, float] | None = None
    modality_names = ("T1", "T1-Gd", "T2", "FLAIR")

    for modality_name, modality_path in zip(modality_names, record.modalities):
        volume, affine, spacing = _load_nifti(modality_path)
        if reference_shape is None:
            reference_shape, reference_affine, reference_spacing = volume.shape, affine, spacing
        elif volume.shape != reference_shape or not np.allclose(affine, reference_affine, atol=AFFINE_TOLERANCE):
            raise ManifestError(
                f"Case '{record.case_id}': {modality_name} is not aligned with T1 "
                f"(shape {volume.shape} vs {reference_shape}). "
                "All four sequences must share one voxel grid; register/resample them first."
            )
        raw_volumes.append(volume)
        image_volumes.append(_z_score_nonzero(volume))

    if record.mask_path is None:
        raise ManifestError(
            f"Case '{record.case_id}' has no tumour mask. Run segmentation before retrieval."
        )
    raw_mask, mask_affine, _ = _load_nifti(record.mask_path)
    if raw_mask.shape != reference_shape or not np.allclose(mask_affine, reference_affine, atol=AFFINE_TOLERANCE):
        raise ManifestError(
            f"Case '{record.case_id}': the tumour mask (shape {raw_mask.shape}) is not aligned "
            f"with the MRI sequences (shape {reference_shape})."
        )
    if not np.allclose(raw_mask, np.round(raw_mask)):
        raise ManifestError(f"Case '{record.case_id}': the mask contains non-integer values; it must be a label map.")
    labels = np.round(raw_mask).astype(np.int16)
    if labels.min() < 0:
        raise ManifestError(f"Case '{record.case_id}': the mask contains negative labels.")
    binary_mask = labels > 0
    if not np.any(binary_mask):
        raise ManifestError(f"Case '{record.case_id}' has an empty tumour mask.")

    brain_mask = np.any(np.stack(raw_volumes, axis=-1) != 0, axis=-1)
    return LoadedCase(
        case_id=record.case_id,
        patient_id=record.patient_id,
        images=np.stack(image_volumes, axis=-1),
        mask=binary_mask,
        affine=reference_affine,
        spacing_mm=reference_spacing,
        split=record.split,
        labels=labels,
        brain_mask=brain_mask,
    )


def load_query_case(
    case_id: str,
    modality_paths: tuple[Path, Path, Path, Path],
    mask_path: Path,
) -> LoadedCase:
    """Load a query with the same contract as reference cases."""
    return load_case(CaseRecord(case_id, case_id, modality_paths, mask_path, "query"))

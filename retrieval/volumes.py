"""NIfTI loading, geometry checks, and MRI normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .data import CaseRecord, ManifestError


@dataclass(frozen=True)
class LoadedCase:
    """Aligned, normalized image channels and a binary tumour mask."""

    case_id: str
    patient_id: str
    images: np.ndarray  # (D, H, W, 4), normalized channel order: T1, T1GD, T2, FLAIR
    mask: np.ndarray  # (D, H, W), bool
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]
    split: str | None


def _z_score_nonzero(volume: np.ndarray) -> np.ndarray:
    foreground = volume[np.isfinite(volume) & (volume != 0)]
    if foreground.size == 0:
        raise ManifestError("A modality has no finite, non-zero voxels for normalization.")
    mean = float(foreground.mean())
    std = float(foreground.std())
    if std == 0:
        raise ManifestError("A modality has zero intensity variance after foreground selection.")
    normalized = (volume.astype(np.float32) - mean) / std
    normalized[~np.isfinite(normalized)] = 0
    return normalized


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    try:
        image = nib.load(str(path))
    except Exception as error:  # nibabel's exceptions differ by backend
        raise ManifestError(f"Could not load NIfTI file '{path}': {error}") from error
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ManifestError(f"'{path}' must be a 3D NIfTI volume; received shape {data.shape}.")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return data, image.affine, spacing


def load_case(record: CaseRecord) -> LoadedCase:
    """Load a case only when all modalities and its mask share exact geometry."""
    image_volumes = []
    reference_shape: tuple[int, ...] | None = None
    reference_affine: np.ndarray | None = None
    reference_spacing: tuple[float, float, float] | None = None

    for modality_path in record.modalities:
        volume, affine, spacing = _load_nifti(modality_path)
        if reference_shape is None:
            reference_shape, reference_affine, reference_spacing = volume.shape, affine, spacing
        elif volume.shape != reference_shape or not np.allclose(affine, reference_affine, atol=1e-4):
            raise ManifestError(
                f"Case '{record.case_id}' has unaligned modalities. "
                "Resample/register all modalities before feature extraction."
            )
        image_volumes.append(_z_score_nonzero(volume))

    if record.mask_path is None:
        raise ManifestError(
            f"Case '{record.case_id}' has no tumour mask. Run segmentation before retrieval."
        )
    raw_mask, mask_affine, _ = _load_nifti(record.mask_path)
    if raw_mask.shape != reference_shape or not np.allclose(mask_affine, reference_affine, atol=1e-4):
        raise ManifestError(
            f"Case '{record.case_id}' has a mask that is not aligned with its modalities."
        )
    binary_mask = raw_mask > 0
    if not np.any(binary_mask):
        raise ManifestError(f"Case '{record.case_id}' has an empty tumour mask.")

    return LoadedCase(
        case_id=record.case_id,
        patient_id=record.patient_id,
        images=np.stack(image_volumes, axis=-1),
        mask=binary_mask,
        affine=reference_affine,
        spacing_mm=reference_spacing,
        split=record.split,
    )


def load_query_case(
    case_id: str,
    modality_paths: tuple[Path, Path, Path, Path],
    mask_path: Path,
) -> LoadedCase:
    """Load a query with the same contract as reference cases."""
    return load_case(CaseRecord(case_id, case_id, modality_paths, mask_path, "query"))

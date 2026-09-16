"""Safe preparation and command generation for a 3D nnU-Net segmentation baseline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .data import MODALITY_COLUMNS, ManifestError, read_manifest


def _safe_case_id(case_id: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in "-_" else "_" for character in case_id)
    if not sanitized:
        raise ManifestError("A case ID cannot be converted to an nnU-Net-safe filename.")
    return sanitized


def prepare_nnunet_dataset(
    manifest_path: str | Path,
    raw_root: str | Path,
    dataset_id: int = 501,
    dataset_name: str = "UPENNGBM",
) -> Path:
    """
    Convert training-manifest rows into nnU-Net v2's lossless NIfTI layout.

    Only rows explicitly assigned to the train split are copied. This protects the
    held-out validation/test cohort from influencing segmentation training.
    """
    if not 11 <= dataset_id <= 999:
        raise ValueError("Choose an nnU-Net dataset ID between 011 and 999.")
    records = [record for record in read_manifest(manifest_path) if record.split == "train"]
    if not records:
        raise ManifestError("No train rows found. Generate patient-level splits first.")

    dataset_directory = Path(raw_root) / f"Dataset{dataset_id:03d}_{dataset_name}"
    if dataset_directory.exists() and any(dataset_directory.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty nnU-Net dataset directory: {dataset_directory}"
        )
    images_directory = dataset_directory / "imagesTr"
    labels_directory = dataset_directory / "labelsTr"
    images_directory.mkdir(parents=True, exist_ok=False)
    labels_directory.mkdir(parents=True, exist_ok=False)

    used_case_ids: set[str] = set()
    for record in records:
        case_id = _safe_case_id(record.case_id)
        if case_id in used_case_ids:
            raise ManifestError(f"Two case IDs normalize to the same nnU-Net filename: {case_id}")
        used_case_ids.add(case_id)
        for channel_index, modality_path in enumerate(record.modalities):
            destination = images_directory / f"{case_id}_{channel_index:04d}.nii.gz"
            shutil.copy2(modality_path, destination)
        shutil.copy2(record.mask_path, labels_directory / f"{case_id}.nii.gz")

    dataset_json = {
        "channel_names": {str(index): modality.upper() for index, modality in enumerate(MODALITY_COLUMNS)},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(records),
        "file_ending": ".nii.gz",
    }
    with (dataset_directory / "dataset.json").open("w", encoding="utf-8") as stream:
        json.dump(dataset_json, stream, indent=2)
    return dataset_directory


def nnunet_environment(raw_root: str | Path, preprocessed_root: str | Path, results_root: str | Path) -> dict[str, str]:
    """Return the three environment variables required by nnU-Net v2 commands."""
    return {
        "nnUNet_raw": str(Path(raw_root).resolve()),
        "nnUNet_preprocessed": str(Path(preprocessed_root).resolve()),
        "nnUNet_results": str(Path(results_root).resolve()),
    }


def run_nnunet_command(command: list[str], environment: dict[str, str]) -> None:
    """Run an explicitly selected nnU-Net command with no shell interpolation."""
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeError(
            f"'{command[0]}' is unavailable. Install the segmentation environment from "
            "requirements-segmentation.txt and ensure its scripts are on PATH."
        )
    subprocess.run([executable, *command[1:]], env={**os.environ, **environment}, check=True)

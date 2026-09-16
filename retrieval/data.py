"""Manifest validation and patient-level splitting for MRI retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


MODALITY_COLUMNS = ("t1", "t1gd", "t2", "flair")
REQUIRED_COLUMNS = ("case_id", "patient_id", *MODALITY_COLUMNS, "mask_path")


class ManifestError(ValueError):
    """Raised when a retrieval manifest cannot support a reproducible experiment."""


@dataclass(frozen=True)
class CaseRecord:
    """One aligned multi-parametric MRI examination and its tumour mask."""

    case_id: str
    patient_id: str
    modalities: tuple[Path, Path, Path, Path]
    mask_path: Path | None
    split: str | None = None


def _resolve_path(value: object, base_dir: Path) -> Path:
    if pd.isna(value) or not str(value).strip():
        raise ManifestError("A required image or mask path is empty.")
    path = Path(str(value))
    return path if path.is_absolute() else (base_dir / path).resolve()


def read_manifest(manifest_path: str | Path, require_masks: bool = True) -> list[CaseRecord]:
    """Read and validate a CSV manifest without guessing a dataset-specific layout."""
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"Manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    required = list(REQUIRED_COLUMNS if require_masks else REQUIRED_COLUMNS[:-1])
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ManifestError(
            "Manifest is missing required columns: " + ", ".join(missing_columns)
        )
    if frame["case_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["case_id"].duplicated(), "case_id"].tolist()
        raise ManifestError(f"case_id values must be unique. Duplicates: {duplicate_ids[:5]}")

    base_dir = manifest_path.parent
    records: list[CaseRecord] = []
    for _, row in frame.iterrows():
        case_id = str(row["case_id"]).strip()
        patient_id = str(row["patient_id"]).strip()
        if not case_id or not patient_id:
            raise ManifestError("case_id and patient_id must both be non-empty.")

        modality_paths = tuple(_resolve_path(row[column], base_dir) for column in MODALITY_COLUMNS)
        mask_path = _resolve_path(row["mask_path"], base_dir) if require_masks else None
        required_paths = [*modality_paths, *( [mask_path] if mask_path else [])]
        missing_paths = [str(path) for path in required_paths if not path.is_file()]
        if missing_paths:
            raise ManifestError(
                f"Case '{case_id}' contains missing files: " + ", ".join(missing_paths)
            )

        split = str(row["split"]).strip().lower() if "split" in frame.columns and not pd.isna(row["split"]) else None
        if split and split not in {"train", "validation", "test", "reference", "query"}:
            raise ManifestError(
                f"Case '{case_id}' has unsupported split '{split}'. "
                "Use train, validation, test, reference, or query."
            )
        records.append(CaseRecord(case_id, patient_id, modality_paths, mask_path, split))
    return records


def records_to_frame(records: Iterable[CaseRecord]) -> pd.DataFrame:
    """Serialize validated records into a portable manifest frame."""
    rows = []
    for record in records:
        row = {
            "case_id": record.case_id,
            "patient_id": record.patient_id,
            "split": record.split or "",
            "mask_path": str(record.mask_path) if record.mask_path else "",
        }
        row.update({column: str(path) for column, path in zip(MODALITY_COLUMNS, record.modalities)})
        rows.append(row)
    return pd.DataFrame(rows)


def create_patient_splits(
    records: Iterable[CaseRecord],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> list[CaseRecord]:
    """Assign every examination from a patient to exactly one split."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ManifestError("train_fraction and validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ManifestError("train_fraction + validation_fraction must be less than 1.")

    records = list(records)
    patient_ids = sorted({record.patient_id for record in records})
    if len(patient_ids) < 3:
        raise ManifestError("At least three patients are required for train/validation/test splits.")

    # Integer counts guarantee each split remains non-empty for small pilot cohorts.
    train_count = min(max(1, round(len(patient_ids) * train_fraction)), len(patient_ids) - 2)
    train_ids, holdout_ids = train_test_split(
        patient_ids, train_size=train_count, random_state=seed, shuffle=True
    )
    validation_count = min(
        max(1, round(len(patient_ids) * validation_fraction)), len(holdout_ids) - 1
    )
    validation_ids, test_ids = train_test_split(
        holdout_ids,
        train_size=validation_count,
        random_state=seed,
        shuffle=True,
    )
    split_by_patient = {
        **{patient_id: "train" for patient_id in train_ids},
        **{patient_id: "validation" for patient_id in validation_ids},
        **{patient_id: "test" for patient_id in test_ids},
    }
    return [
        CaseRecord(
            record.case_id,
            record.patient_id,
            record.modalities,
            record.mask_path,
            split_by_patient[record.patient_id],
        )
        for record in records
    ]

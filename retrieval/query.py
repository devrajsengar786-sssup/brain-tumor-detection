"""Turn one uploaded examination (four sequences + mask) into query features."""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data import CaseRecord, ManifestError
from .features import display_measurements, extract_engineered_features
from .previews import preview_array
from .volumes import load_case


ROLE_LABELS = {"t1": "T1", "t1gd": "T1-Gd (post-contrast)", "t2": "T2", "flair": "FLAIR", "mask": "Tumour mask"}
# Filename patterns checked in order; T1-Gd must be tested before plain T1.
ROLE_PATTERNS = (
    ("mask", re.compile(r"(seg|mask|label)", re.I)),
    ("t1gd", re.compile(r"(ce-gd|t1gd|t1ce|t1-gd|t1c\b|t1c[_.-]|post|gad)", re.I)),
    ("flair", re.compile(r"flair", re.I)),
    ("t2", re.compile(r"t2", re.I)),
    ("t1", re.compile(r"t1", re.I)),
)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
SUBJECT_PATTERN = re.compile(r"(sub-\d{3})")


def detect_role(filename: str) -> str | None:
    name = Path(filename).name
    if not name.lower().endswith((".nii", ".nii.gz")):
        return None
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(name):
            return role
    return None


def files_from_zip(payload: bytes) -> dict[str, tuple[str, bytes]]:
    """Map role -> (filename, bytes) from a zip containing one examination."""
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ManifestError("The zip is larger than 200 MB; upload a single examination.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ManifestError("The uploaded file is not a valid zip archive.") from error
    found: dict[str, tuple[str, bytes]] = {}
    for info in archive.infolist():
        if info.is_dir() or "__MACOSX" in info.filename:
            continue
        role = detect_role(info.filename)
        if role is None:
            continue
        if role in found:
            raise ManifestError(f"The zip contains more than one {ROLE_LABELS[role]} file ({found[role][0]}, {info.filename}).")
        if info.file_size > MAX_UPLOAD_BYTES:
            raise ManifestError(f"{info.filename} is too large.")
        found[role] = (Path(info.filename).name, archive.read(info))
    return found


def missing_roles(files: Mapping[str, object]) -> list[str]:
    return [ROLE_LABELS[role] for role in ROLE_LABELS if role not in files]


@dataclass
class PreparedQuery:
    case_id: str
    patient_hint: str | None
    features: pd.DataFrame
    measurements: dict[str, float]
    preview: np.ndarray
    shape: tuple[int, ...]
    affine: np.ndarray


def prepare_query(files: Mapping[str, tuple[str, bytes]], case_id: str = "uploaded-exam") -> PreparedQuery:
    """Validate geometry/labels and extract the same features used for the gallery."""
    missing = missing_roles(files)
    if missing:
        raise ManifestError("Missing input(s): " + ", ".join(missing) + ".")
    hint = None
    for name, _ in files.values():
        match = SUBJECT_PATTERN.search(name)
        if match:
            hint = match.group(1)
            break
    with tempfile.TemporaryDirectory(prefix="mri_query_") as tmp:
        tmp = Path(tmp)
        paths = {}
        for role, (name, payload) in files.items():
            suffix = ".nii.gz" if name.lower().endswith(".gz") else ".nii"
            path = tmp / f"{role}{suffix}"
            path.write_bytes(payload)
            paths[role] = path
        record = CaseRecord(case_id, hint or case_id, tuple(paths[r] for r in ("t1", "t1gd", "t2", "flair")), paths["mask"], "query")
        case = load_case(record)
        features = extract_engineered_features(case)
    return PreparedQuery(
        case_id,
        hint,
        pd.DataFrame([features], index=pd.Index([case_id], name="case_id")),
        display_measurements(case),
        preview_array(case),
        tuple(case.mask.shape),
        case.affine,
    )

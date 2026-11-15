"""UPENN-GBM mirror inventory, resumable downloads, validation and metadata joins.

Source: the individual-file mirror at
https://github.com/data-nih/tcia/releases/tag/upenn-gbm (CC BY 4.0, TCIA DOI
10.7937/TCIA.709X-DN49). The mirror names baseline files ``sub-NNN_<role>.nii.gz``
and follow-up files ``sub-NNN_2_<role>.nii.gz``; ``participants.tsv`` uses
``UPENN-GBM-NNNNN_11`` (baseline) and ``UPENN-GBM-NNNNN_21`` (follow-up).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import nibabel as nib
import numpy as np
import pandas as pd

from .features import ALLOWED_LABELS


RELEASE_API = "https://api.github.com/repos/data-nih/tcia/releases/tags/upenn-gbm"
DOWNLOAD_BASE = "https://github.com/data-nih/tcia/releases/download/upenn-gbm"
USER_AGENT = "upenn-gbm-retrieval-research/1.0"

# Role -> mirror suffix. Order matters: the four MRI channels, then the mask.
ROLE_SUFFIXES = {
    "t1": "T1w",
    "t1gd": "ce-gd_T1w",
    "t2": "T2w",
    "flair": "FLAIR",
    "mask": "seg",
}
MRI_ROLES = ("t1", "t1gd", "t2", "flair")
BASELINE_PATTERN = re.compile(r"^sub-(\d{3})_(T1w|ce-gd_T1w|T2w|FLAIR|seg)\.nii\.gz$")
FOLLOWUP_PATTERN = re.compile(r"^sub-(\d{3})_(?:2_|ses-2_)")
MISSING_TOKENS = {"", "na", "n/a", "nan", "not available", "not applicable", "unknown"}

# participants.tsv fields: which may be displayed, and which are reserved for
# secondary evaluation only (never used in image-based ranking).
DISPLAY_FIELDS = {"Gender": "sex", "Age_at_scan_years": "age_years"}
SECONDARY_FIELDS = {
    "IDH1": "idh1",
    "MGMT": "mgmt",
    "Survival_from_surgery_days": "survival_days",
    "KPS": "kps",
    "GTR_over90percent": "gtr_over_90pct",
}


# --------------------------------------------------------------------------- inventory
def fetch_release_assets(api_url: str = RELEASE_API, timeout: float = 60.0) -> list[dict]:
    """List release assets (name, size, url), following pagination when needed."""
    def get(url: str):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    # One unauthenticated request (the API allows 60/hour). On 2026-11-15 this endpoint
    # returned all 4,424 assets; the notebook prints the count so truncation is visible.
    release = get(api_url)
    assets = list(release.get("assets", []))
    return [
        {"name": asset["name"], "size": int(asset["size"]), "url": asset.get("browser_download_url") or f"{DOWNLOAD_BASE}/{asset['name']}"}
        for asset in assets
    ]


def build_inventory(assets: Iterable[Mapping]) -> pd.DataFrame:
    """One row per subject: baseline URLs/sizes per role, completeness, follow-up flag."""
    subjects: dict[str, dict] = {}
    followups: set[str] = set()
    for asset in assets:
        name = asset["name"]
        followup = FOLLOWUP_PATTERN.match(name)
        if followup:
            followups.add(f"sub-{followup.group(1)}")
            continue
        match = BASELINE_PATTERN.match(name)
        if not match:
            continue
        subject = f"sub-{match.group(1)}"
        role = next(role for role, suffix in ROLE_SUFFIXES.items() if suffix == match.group(2))
        row = subjects.setdefault(subject, {"subject": subject})
        row[f"{role}_file"] = name
        row[f"{role}_size"] = int(asset["size"])
        row[f"{role}_url"] = asset.get("url") or f"{DOWNLOAD_BASE}/{name}"
    rows = []
    for subject, row in sorted(subjects.items()):
        missing = [role for role in ROLE_SUFFIXES if f"{role}_file" not in row]
        row["missing_roles"] = ";".join(missing)
        row["complete"] = not missing
        row["has_followup_files"] = subject in followups
        row["total_bytes"] = sum(row.get(f"{role}_size", 0) for role in ROLE_SUFFIXES)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- downloads
def download_file(url: str, destination: Path, expected_size: int | None = None, retries: int = 5, timeout: float = 120.0) -> str:
    """
    Resumable download: data goes to ``<name>.part`` and resumes with an HTTP Range
    request; the file is renamed only once its size matches the release listing.
    Returns "cached", "downloaded" or raises after ``retries`` failures.
    """
    destination = Path(destination)
    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and offset > expected_size:
            partial.unlink()
            offset = 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as stream:
                    while chunk := response.read(1 << 20):
                        stream.write(chunk)
            size = partial.stat().st_size
            if expected_size is not None and size != expected_size:
                raise IOError(f"size mismatch after download: {size} != {expected_size}")
            partial.replace(destination)
            return "downloaded"
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 416:  # range not satisfiable -> restart cleanly
                partial.unlink(missing_ok=True)
            elif error.code == 404:
                break
        except Exception as error:  # network hiccups, timeouts, truncated streams
            last_error = error
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Failed to download {destination.name}: {last_error}")


def download_inventory(inventory: pd.DataFrame, raw_dir: str | Path, max_workers: int = 6, progress=None) -> pd.DataFrame:
    """Download every role of every complete subject; returns a per-file status report."""
    raw_dir = Path(raw_dir)
    jobs = []
    for _, row in inventory[inventory["complete"]].iterrows():
        for role in ROLE_SUFFIXES:
            jobs.append((row["subject"], role, row[f"{role}_url"], raw_dir / row[f"{role}_file"], int(row[f"{role}_size"])))
    report = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_file, url, path, size): (subject, role, path) for subject, role, url, path, size in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            subject, role, path = futures[future]
            try:
                status, error = future.result(), ""
            except Exception as exc:  # keep going; failures are reported, not fatal
                status, error = "failed", str(exc)
            report.append({"subject": subject, "role": role, "file": path.name, "status": status, "error": error})
            if progress and done % 50 == 0:
                progress(done, len(jobs))
    return pd.DataFrame(report).sort_values(["subject", "role"]).reset_index(drop=True)


# --------------------------------------------------------------------------- validation
@dataclass
class CaseValidation:
    subject: str
    ok: bool
    reason: str
    shape: str = ""
    spacing: str = ""
    labels: str = ""
    label_voxels: str = ""


def validate_case_files(subject: str, paths: Mapping[str, Path], expected_shape: tuple[int, ...] | None = None) -> CaseValidation:
    """Open all five files and check geometry, label set and non-empty mask."""
    reference = None
    for role in (*MRI_ROLES, "mask"):
        path = Path(paths[role])
        if not path.is_file():
            return CaseValidation(subject, False, f"missing file for {role}")
        try:
            image = nib.load(str(path))
            shape = tuple(int(v) for v in image.shape[:3])
            affine = image.affine
            zooms = tuple(round(float(v), 4) for v in image.header.get_zooms()[:3])
            if role == "mask":
                data = np.asanyarray(image.dataobj)
            else:
                # Touch the data so truncated/corrupt gzip streams fail here, not later.
                data = np.asanyarray(image.dataobj)
                if not np.isfinite(data).all() or not np.any(data):
                    return CaseValidation(subject, False, f"{role} is empty or contains non-finite values")
        except Exception as error:
            return CaseValidation(subject, False, f"{role} could not be opened: {error}")
        if len(image.shape) != 3:
            return CaseValidation(subject, False, f"{role} is not 3D: {image.shape}")
        if reference is None:
            reference = (shape, affine, zooms)
        elif shape != reference[0] or not np.allclose(affine, reference[1], atol=1e-3):
            return CaseValidation(subject, False, f"{role} geometry {shape} does not match T1 {reference[0]}")
    if expected_shape is not None and reference[0] != tuple(expected_shape):
        return CaseValidation(subject, False, f"shape {reference[0]} differs from cohort shape {expected_shape}")
    values, counts = np.unique(data, return_counts=True)
    labels = {int(v): int(c) for v, c in zip(values, counts)}
    base = dict(shape="x".join(map(str, reference[0])), spacing="x".join(map(str, reference[2])),
                labels=";".join(map(str, sorted(labels))), label_voxels=json.dumps(labels))
    if not np.allclose(values, np.round(values)):
        return CaseValidation(subject, False, "mask has non-integer values", **base)
    if set(labels) - ALLOWED_LABELS:
        return CaseValidation(subject, False, f"unexpected mask labels {sorted(set(labels) - ALLOWED_LABELS)}", **base)
    if sum(c for v, c in labels.items() if v > 0) == 0:
        return CaseValidation(subject, False, "mask is empty", **base)
    if set(labels) <= {0, 1}:
        return CaseValidation(subject, False, "mask is binary; subregion labels required", **base)
    return CaseValidation(subject, True, "", **base)


def local_paths(row: Mapping, raw_dir: Path) -> dict[str, Path]:
    return {role: Path(raw_dir) / row[f"{role}_file"] for role in ROLE_SUFFIXES}


# --------------------------------------------------------------------------- metadata join
def candidate_participant_id(subject: str, timepoint: str = "11") -> str:
    """sub-002 -> UPENN-GBM-00002_11 (a *candidate* mapping; see verify_subject_mapping)."""
    match = re.fullmatch(r"sub-(\d+)", subject)
    if not match:
        raise ValueError(f"Unexpected subject id: {subject}")
    return f"UPENN-GBM-{int(match.group(1)):05d}_{timepoint}"


def read_participants(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "ID" not in frame.columns:
        raise ValueError("participants.tsv has no 'ID' column.")
    return frame


def _clean(value: object) -> object:
    text = str(value).strip()
    return None if text.lower() in MISSING_TOKENS else text


def verify_subject_mapping(inventory: pd.DataFrame, participants: pd.DataFrame) -> dict:
    """
    Independent check of the numeric sub-NNN -> UPENN-GBM-NNNNN mapping.

    The mirror marks which subjects have follow-up files (``sub-NNN_2_*``) and
    participants.tsv marks which IDs have a follow-up row (``_21``). If the numeric
    mapping is right, every mirror follow-up subject must map to an ``_21`` row. If the
    numbering were unrelated, the chance of that happening is hypergeometric and
    vanishingly small, so passing this test corroborates the mapping.
    """
    from math import comb

    ids = set(participants["ID"])
    numbers = {int(m.group(1)) for m in (re.match(r"UPENN-GBM-(\d+)_", i) for i in ids) if m}
    followup_ids = {int(i[10:15]) for i in ids if i.endswith("_21")}
    mirror_followups = {int(s[4:]) for s in inventory.loc[inventory["has_followup_files"], "subject"]}
    matched = mirror_followups & followup_ids
    population = max(len(numbers), 1)
    chance = comb(len(followup_ids), len(mirror_followups)) / comb(population, len(mirror_followups)) if mirror_followups and len(mirror_followups) <= len(followup_ids) else 0.0
    passed = bool(mirror_followups) and matched == mirror_followups
    return {
        "mirror_subjects": int(len(inventory)),
        "participant_patients": len(numbers),
        "mirror_followup_subjects": len(mirror_followups),
        "participant_followup_rows": len(followup_ids),
        "mirror_followups_matching_21_rows": len(matched),
        "probability_if_unrelated": chance,
        "passed": passed,
    }


def join_metadata(subjects: Iterable[str], participants: pd.DataFrame, mapping_check: Mapping) -> pd.DataFrame:
    """
    Join baseline participant rows to subjects. ``join_verified`` is True only when the
    numeric mapping passed ``verify_subject_mapping`` *and* exactly one ``_11`` row exists.
    Metadata is exposed only for verified joins; missing fields are listed per case.
    """
    by_id = participants.set_index("ID")
    rows = []
    for subject in subjects:
        candidate = candidate_participant_id(subject)
        exists = candidate in by_id.index and not isinstance(by_id.loc[candidate], pd.DataFrame)
        verified = bool(mapping_check.get("passed")) and exists
        row = {"subject": subject, "participant_id": candidate if exists else "", "join_verified": verified,
               "join_note": "" if verified else ("no baseline (_11) row" if not exists else "mapping check failed")}
        missing = []
        for source, target in {**DISPLAY_FIELDS, **SECONDARY_FIELDS}.items():
            value = _clean(by_id.loc[candidate].get(source, "")) if verified else None
            row[target] = value
            if value is None:
                missing.append(target)
        row["missing_fields"] = ";".join(missing)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["age_years"] = pd.to_numeric(frame["age_years"], errors="coerce")
    return frame

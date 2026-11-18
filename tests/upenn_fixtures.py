"""Synthetic, mirror-named UPENN-GBM-like cases for tests (no patient data)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SHAPE = (32, 32, 20)
AFFINE = np.array([[-1.0, 0, 0, 0], [0, -1.0, 0, 31.0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
SUFFIX = {"t1": "T1w", "t1gd": "ce-gd_T1w", "t2": "T2w", "flair": "FLAIR", "mask": "seg"}


def make_volumes(number: int, shape=SHAPE):
    rng = np.random.default_rng(number)
    grid = np.indices(shape, dtype=np.float32)
    centre = np.array([10 + (number * 3) % 12, 10 + (number * 5) % 12, 6 + number % 8], dtype=np.float32)
    radii = np.array([3 + number % 4, 3 + (number // 2) % 4, 2 + number % 3], dtype=np.float32)
    distance = np.sqrt((((grid - centre[:, None, None, None]) / radii[:, None, None, None]) ** 2).sum(axis=0))
    labels = np.zeros(shape, dtype=np.uint8)
    labels[distance <= 1.0] = 2
    labels[distance <= 0.65] = 3
    labels[distance <= 0.3 + 0.05 * (number % 3)] = 1
    brain = np.sqrt((((grid - np.array(shape)[:, None, None, None] / 2) / (np.array(shape)[:, None, None, None] / 2.1)) ** 2).sum(axis=0)) <= 1
    images = {}
    for k, role in enumerate(("t1", "t1gd", "t2", "flair"), start=1):
        volume = (80 + 10 * k + rng.normal(0, 4, shape)).astype(np.float32)
        volume[labels == 3] += 40 if role == "t1gd" else 10 * k
        volume[labels == 1] -= 20
        volume[labels == 2] += 25 if role == "flair" else 5
        volume[~brain] = 0
        images[role] = np.clip(volume, 0, 255).astype(np.uint8)
    labels[~brain] = 0
    return images, labels


def write_subject(root: Path, number: int, roles=("t1", "t1gd", "t2", "flair", "mask"), affine=AFFINE, shape=SHAPE) -> dict[str, Path]:
    images, labels = make_volumes(number, shape)
    paths = {}
    for role in roles:
        data = labels if role == "mask" else images[role]
        path = root / f"sub-{number:03d}_{SUFFIX[role]}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), path)
        paths[role] = path
    return paths


def assets_for(root: Path, numbers, followups=()) -> list[dict]:
    assets = []
    for path in sorted(root.glob("sub-*.nii.gz")):
        assets.append({"name": path.name, "size": path.stat().st_size, "url": path.as_uri()})
    for number in followups:
        assets.append({"name": f"sub-{number:03d}_2_FLAIR.nii.gz", "size": 10, "url": "file:///nonexistent"})
    return assets


def participants_for(numbers, followups=(), total=40) -> pd.DataFrame:
    rows = []
    for n in range(1, total + 1):
        rows.append({"ID": f"UPENN-GBM-{n:05d}_11", "Gender": "F" if n % 2 else "M", "Age_at_scan_years": f"{40 + n % 30}.5",
                     "Survival_from_surgery_days": "Not Available" if n % 3 == 0 else str(300 + n), "IDH1": "Wildtype",
                     "MGMT": ["Methylated", "Unmethylated", "Not Available"][n % 3], "KPS": "Not Available",
                     "GTR_over90percent": "Y", "Time_since_baseline_preop": "0", "PsP_TP_score": "NA"})
    for n in followups:
        rows.append({"ID": f"UPENN-GBM-{n:05d}_21", "Gender": "F", "Age_at_scan_years": "50", "Survival_from_surgery_days": "NA",
                     "IDH1": "Wildtype", "MGMT": "Not Available", "KPS": "NA", "GTR_over90percent": "Not Applicable",
                     "Time_since_baseline_preop": "200", "PsP_TP_score": "3"})
    return pd.DataFrame(rows)

"""The gallery bundle: everything Streamlit needs to search, with no raw reference scans.

Layout (all portable text/PNG files; no pickle, so it is safe to load from shared storage)::

    bundle.json              schema version, feature space (gallery-fitted scaler + weights),
                             gallery geometry, similarity cutoff (null unless validated)
    gallery_features.csv     one row of ranking features per reference case
    gallery_metadata.csv     display-ready metadata (verified joins only) + measurements
    demo_queries.csv         features of held-out test queries (never in the gallery)
    demo_metadata.csv        their display metadata
    previews/<case_id>.png   small T1-Gd | FLAIR preview with subregion outlines
    experiment_manifest.json how the bundle was produced (split, versions, counts, reports)
    reports/*.csv            download, validation, exclusion and metadata reports

Secondary labels (IDH1, MGMT, survival, ...) are written to
``evaluation/secondary_labels.csv`` and are never read by the app or the ranking.
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .index import FeatureSpace, RetrievalIndex


SCHEMA_VERSION = 1
REQUIRED_FILES = ("bundle.json", "gallery_features.csv", "gallery_metadata.csv")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class BundleError(RuntimeError):
    """The gallery bundle is missing, incomplete, or incompatible with this code."""


def _check_id(case_id: str) -> str:
    if not SAFE_ID.match(case_id):
        raise BundleError(f"Unsafe case id in bundle: {case_id!r}")
    return case_id


def write_bundle(
    directory: str | Path,
    index: RetrievalIndex,
    gallery_metadata: pd.DataFrame,
    previews: Mapping[str, bytes],
    manifest: Mapping,
    geometry: Mapping,
    demo_features: pd.DataFrame | None = None,
    demo_metadata: pd.DataFrame | None = None,
    similarity_cutoff: Mapping | None = None,
    reports: Mapping[str, pd.DataFrame] | None = None,
    secondary_labels: pd.DataFrame | None = None,
) -> Path:
    directory = Path(directory)
    (directory / "previews").mkdir(parents=True, exist_ok=True)
    index.raw_features.to_csv(directory / "gallery_features.csv", index_label="case_id")
    gallery_metadata.reindex(index.raw_features.index).to_csv(directory / "gallery_metadata.csv", index_label="case_id")
    if demo_features is not None:
        demo_features[index.space.columns].to_csv(directory / "demo_queries.csv", index_label="case_id")
        (demo_metadata if demo_metadata is not None else pd.DataFrame(index=demo_features.index)).reindex(
            demo_features.index
        ).to_csv(directory / "demo_metadata.csv", index_label="case_id")
    for case_id, png in previews.items():
        (directory / "previews" / f"{_check_id(case_id)}.png").write_bytes(png)
    for name, frame in (reports or {}).items():
        (directory / "reports").mkdir(exist_ok=True)
        frame.to_csv(directory / "reports" / f"{name}.csv", index=False)
    if secondary_labels is not None:
        (directory / "evaluation").mkdir(exist_ok=True)
        secondary_labels.to_csv(directory / "evaluation" / "secondary_labels.csv", index_label="case_id")
    (directory / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    header = {
        "schema_version": SCHEMA_VERSION,
        "feature_set_version": manifest.get("feature_set_version"),
        "feature_space": index.space.to_dict(),
        "geometry": dict(geometry),
        "similarity_cutoff": dict(similarity_cutoff) if similarity_cutoff else None,
        "gallery_cases": len(index.case_ids),
        "demo_queries": 0 if demo_features is None else len(demo_features),
    }
    (directory / "bundle.json").write_text(json.dumps(header, indent=2), encoding="utf-8")
    return directory


@dataclass
class GalleryBundle:
    directory: Path
    header: dict
    index: RetrievalIndex
    demo_features: pd.DataFrame
    demo_metadata: pd.DataFrame
    manifest: dict

    @property
    def cutoff(self) -> float | None:
        cutoff = self.header.get("similarity_cutoff")
        return float(cutoff["threshold"]) if cutoff and cutoff.get("threshold") is not None else None

    @classmethod
    def load(cls, directory: str | Path) -> "GalleryBundle":
        directory = Path(directory)
        if not directory.is_dir():
            raise BundleError(f"Gallery bundle folder not found: {directory}")
        missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
        if missing:
            raise BundleError(f"Gallery bundle at {directory} is incomplete; missing {', '.join(missing)}.")
        try:
            header = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise BundleError(f"bundle.json is not valid JSON: {error}") from error
        if header.get("schema_version") != SCHEMA_VERSION:
            raise BundleError(
                f"Bundle schema {header.get('schema_version')} is not supported (expected {SCHEMA_VERSION}). "
                "Rebuild it with the Colab notebook from this branch."
            )
        space = FeatureSpace.from_dict(header["feature_space"])
        features = pd.read_csv(directory / "gallery_features.csv", index_col="case_id")
        features.index = features.index.astype(str)
        missing_columns = [c for c in space.columns if c not in features.columns]
        if missing_columns:
            raise BundleError(f"gallery_features.csv lacks columns: {missing_columns[:5]}")
        metadata = pd.read_csv(directory / "gallery_metadata.csv", index_col="case_id", dtype={"patient_id": str})
        metadata.index = metadata.index.astype(str)
        if features.empty:
            raise BundleError("The gallery contains no cases.")
        index = RetrievalIndex.fit(features[space.columns], metadata, space=space)
        demo_path, demo_meta_path = directory / "demo_queries.csv", directory / "demo_metadata.csv"
        demo = pd.read_csv(demo_path, index_col="case_id") if demo_path.is_file() else pd.DataFrame(columns=space.columns)
        demo.index = demo.index.astype(str)
        demo_meta = pd.read_csv(demo_meta_path, index_col="case_id", dtype={"patient_id": str}) if demo_meta_path.is_file() else pd.DataFrame(index=demo.index)
        demo_meta.index = demo_meta.index.astype(str)
        manifest_path = directory / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        return cls(directory, header, index, demo, demo_meta, manifest)

    def preview_path(self, case_id: str) -> Path | None:
        path = self.directory / "previews" / f"{_check_id(str(case_id))}.png"
        return path if path.is_file() else None

    def check_geometry(self, shape: tuple[int, ...], affine: np.ndarray) -> None:
        """Location features are only comparable in the gallery's atlas space."""
        geometry = self.header.get("geometry") or {}
        expected_shape = tuple(geometry.get("shape", ()))
        if expected_shape and tuple(shape) != expected_shape:
            raise BundleError(
                f"The examination grid {tuple(shape)} differs from the gallery atlas grid {expected_shape}. "
                "Register the scans to the same atlas (UPENN-GBM uses SRI24, 240x240x155 at 1 mm) first."
            )
        expected_affine = geometry.get("affine")
        if expected_affine is not None and not np.allclose(affine, np.asarray(expected_affine), atol=1e-2):
            raise BundleError("The examination's voxel-to-world orientation differs from the gallery atlas.")

    def search(
        self,
        query_features: pd.DataFrame,
        top_k: int = 5,
        exclude_patient_id: str | None = None,
        apply_cutoff: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Rank gallery cases for one query. Returns (results, excluded_case_ids).

        A gallery case is excluded when it belongs to ``exclude_patient_id`` or when its
        features are identical to the query (the same examination uploaded again).
        """
        query = query_features[self.index.space.columns]
        excluded = []
        if exclude_patient_id:
            excluded += [c for c, p in zip(self.index.case_ids, self.index.patient_ids) if p == str(exclude_patient_id)]
        identical = np.isclose(self.index.raw_features.to_numpy(), query.to_numpy(dtype=np.float64), rtol=1e-6, atol=1e-6).all(axis=1)
        excluded += [c for c, same in zip(self.index.case_ids, identical) if same and c not in excluded]
        excluded_patients = {p for c, p in zip(self.index.case_ids, self.index.patient_ids) if c in excluded}
        results = self.index.query(query, top_k=len(self.index.case_ids))
        if not results.empty:
            results = results[~results["case_id"].isin(excluded) & ~results["patient_id"].isin(excluded_patients)]
        if apply_cutoff and self.cutoff is not None and not results.empty:
            results = results[results["similarity"] >= self.cutoff]
        return results.head(top_k).reset_index(drop=True), excluded


def fetch_bundle_zip(url: str, cache_dir: str | Path, timeout: float = 120.0) -> Path:
    """Download a zipped bundle once and extract it safely into ``cache_dir``."""
    cache_dir = Path(cache_dir)
    marker = cache_dir / ".source_url"
    if (cache_dir / "bundle.json").is_file() and marker.is_file() and marker.read_text() == url:
        return cache_dir
    request = urllib.request.Request(url, headers={"User-Agent": "upenn-gbm-retrieval-app/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except Exception as error:
        raise BundleError(f"Could not download the gallery bundle: {error}") from error
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise BundleError("The gallery bundle URL did not return a zip file.") from error
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_dir.resolve()
    names = archive.namelist()
    prefix = ""
    if "bundle.json" not in names:
        tops = {name.split("/", 1)[0] for name in names if "/" in name}
        if len(tops) == 1 and f"{next(iter(tops))}/bundle.json" in names:
            prefix = next(iter(tops)) + "/"
    for name in names:
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        target = (root / name[len(prefix):]).resolve()
        if root not in target.parents:
            raise BundleError(f"Refusing unsafe path in bundle zip: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
    marker.write_text(url)
    return cache_dir

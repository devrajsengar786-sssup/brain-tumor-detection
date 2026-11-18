"""2D reference gallery for JPG/PNG queries: clean slices, image-feature index, metadata.

Why a separate gallery: the 3D bundle's index needs volumes and tumour masks, and its
preview PNGs have tumour outlines burned in. A 2D upload can only be compared with
*clean* slices, using image features alone.

Layout of ``artifacts/gallery_2d/`` (git-ignored)::

    reference/<case_id>.png   clean axial post-contrast T1 slice, gallery split only (searchable)
    demo/<case_id>.png        the same for held-out test cases (never indexed)
    cases.csv                 one row per exported slice with verified, display-ready metadata
    index.npz                 L2-normalised image embeddings of the reference slices
    manifest.json             source, counts, verification summary, feature description

Preparation (``python -m retrieval.gallery2d prepare``) reads the source scans from the
``data-nih/tcia`` UPENN-GBM mirror into memory, exports one slice per case, and writes
no raw scan to disk. Uploads only ever read the generated local gallery.
"""

from __future__ import annotations

if __name__ == "__main__":  # keep the gallery builder's memory use low too
    import runtime_limits  # noqa: F401

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_BUNDLE_DIR = Path("artifacts/gallery_bundle")
DEFAULT_GALLERY_DIR = Path("artifacts/gallery_2d")
MIRROR_BASE = "https://github.com/data-nih/tcia/releases/download/upenn-gbm"
SOURCE_CITATION = (
    "UPENN-GBM (Bakas et al., Scientific Data 2022), The Cancer Imaging Archive, "
    "DOI 10.7937/TCIA.709X-DN49, CC BY 4.0"
)
SEQUENCE_LABEL = "Post-contrast T1 (T1-Gd), axial"
REFERENCE_DIAGNOSIS = "Glioblastoma"
GALLERY_SCOPE = (
    "Every reference case is a de novo glioblastoma from the UPENN-GBM collection. "
    "The library contains no meningioma, pituitary-tumour or tumour-free scans."
)
SCHEMA_VERSION = 1
# A reference whose embedding is this close to the query is treated as the same image.
NEAR_DUPLICATE_SIMILARITY = 0.9995

Embedder = Callable[[Image.Image], np.ndarray]


class Gallery2DError(RuntimeError):
    """The 2D reference gallery is missing, incomplete or inconsistent."""


# --------------------------------------------------------------------------- image helpers
def pixel_sha256(image: Image.Image) -> str:
    """Hash of the decoded 8-bit greyscale pixels (independent of file encoding)."""
    grey = np.asarray(image.convert("L"))
    return hashlib.sha256(f"{grey.shape}".encode() + grey.tobytes()).hexdigest()


def render_slice(volume_slice: np.ndarray, flip_vertical: bool = False) -> Image.Image:
    """Window an axial slice to 8-bit greyscale with anterior at the top."""
    slice_ = np.asarray(volume_slice, dtype=np.float32)
    brain = slice_ > 0
    if not brain.any():
        raise Gallery2DError("The selected slice contains no brain voxels.")
    low, high = np.percentile(slice_[brain], (0.5, 99.5))
    scaled = np.clip((slice_ - low) / max(high - low, 1e-6), 0, 1)
    scaled[~brain] = 0
    pixels = (scaled * 255).round().astype(np.uint8).T  # array axes are (x, y) -> rows = y
    return Image.fromarray(np.flipud(pixels) if flip_vertical else pixels, mode="L")


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


# --------------------------------------------------------------------------- preparation
def _fetch(url: str, retries: int = 4, timeout: float = 120.0) -> bytes:
    last = None
    for _ in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "upenn-gbm-retrieval-research/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # transient network errors are retried
            last = error
    raise Gallery2DError(f"Could not download {url}: {last}")


def _load_nifti_bytes(payload: bytes):
    import nibabel as nib

    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return nib.Nifti1Image.from_bytes(raw)


def read_bundle_cases(bundle_dir: Path) -> pd.DataFrame:
    """Case list from the 3D bundle: gallery split -> reference, test split -> demo."""
    bundle_dir = Path(bundle_dir)
    needed = ["gallery_metadata.csv", "demo_metadata.csv", "evaluation/secondary_labels.csv"]
    missing = [name for name in needed if not (bundle_dir / name).is_file()]
    if missing:
        raise Gallery2DError(f"Gallery bundle at {bundle_dir} is missing: {', '.join(missing)}")
    gallery = pd.read_csv(bundle_dir / "gallery_metadata.csv").assign(role="reference")
    demo = pd.read_csv(bundle_dir / "demo_metadata.csv").assign(role="demo")
    cases = pd.concat([gallery, demo], ignore_index=True)
    if cases["case_id"].duplicated().any():
        raise Gallery2DError("A case appears in both the gallery and the held-out demo set.")
    if not ((gallery["split"] == "gallery").all() and (demo["split"] == "test").all()):
        raise Gallery2DError("Bundle split labels do not match the expected gallery/test roles.")
    secondary = pd.read_csv(bundle_dir / "evaluation" / "secondary_labels.csv")[["case_id", "idh1", "mgmt"]]
    cases = cases.merge(secondary, on="case_id", how="left", validate="one_to_one")
    # The participant ID must be the numeric counterpart of the mirror subject ID.
    expected = cases["case_id"].str.extract(r"sub-(\d+)")[0].astype(int).map(lambda n: f"UPENN-GBM-{n:05d}_11")
    cases["id_consistent"] = cases["participant_id"].eq(expected) & cases["join_verified"].astype(bool)
    return cases


def export_case(case: pd.Series, out_dir: Path, fetch: Callable[[str], bytes] = _fetch) -> dict:
    """Export one clean T1-Gd slice and verify it against the supplied mask."""
    case_id, z = case["case_id"], int(case["preview_slice"])
    target = out_dir / case["role"] / f"{case_id}.png"
    image_url = f"{MIRROR_BASE}/{case_id}_ce-gd_T1w.nii.gz"
    mask = np.asanyarray(_load_nifti_bytes(fetch(f"{MIRROR_BASE}/{case_id}_seg.nii.gz")).dataobj)
    volume = _load_nifti_bytes(fetch(image_url))
    if volume.shape != mask.shape:
        raise Gallery2DError(f"{case_id}: scan {volume.shape} and mask {mask.shape} differ in shape.")
    voxel_ml = float(np.prod(volume.header.get_zooms()[:3])) / 1000.0
    areas = (mask > 0).sum(axis=(0, 1))
    volume_ml = float((mask > 0).sum() * voxel_ml)
    image = render_slice(np.asanyarray(volume.dataobj[:, :, z]), flip_vertical=volume.affine[1, 1] > 0)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG", optimize=True)
    return {
        "case_id": case_id,
        "slice_is_largest_tumour_area": bool(int(np.argmax(areas)) == z),
        "tumour_area_px_in_slice": int(areas[z]),
        "volume_ml_from_mask": round(volume_ml, 3),
        "volume_matches_bundle": bool(abs(volume_ml - float(case["tumour_volume_ml"])) < 0.01),
        "pixel_sha256": pixel_sha256(image),
        "source_file": f"{case_id}_ce-gd_T1w.nii.gz",
    }


def prepare_slices(
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    out_dir: Path = DEFAULT_GALLERY_DIR,
    workers: int = 6,
    fetch: Callable[[str], bytes] = _fetch,
    progress: Callable[[str], None] | None = print,
) -> pd.DataFrame:
    """Export clean slices for every bundle case; resumable (existing verified rows are kept)."""
    bundle_dir, out_dir = Path(bundle_dir), Path(out_dir)
    cases = read_bundle_cases(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "export_progress.json"
    done: dict[str, dict] = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    done = {k: v for k, v in done.items() if (out_dir / cases.set_index("case_id").loc[k, "role"] / f"{k}.png").is_file()} if done else {}
    todo = cases[~cases["case_id"].isin(done)]
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(export_case, row, out_dir, fetch): row["case_id"] for _, row in todo.iterrows()}
        for count, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            try:
                done[case_id] = future.result()
            except Exception as error:
                failures[case_id] = str(error)
            if count % 10 == 0 or count == len(futures):
                progress_path.write_text(json.dumps(done))
                if progress:
                    progress(f"exported {len(done)}/{len(cases)} (failed {len(failures)})")
    progress_path.write_text(json.dumps(done))

    verified = pd.DataFrame(done.values())
    table = cases.merge(verified, on="case_id", how="inner", validate="one_to_one")
    table["verified"] = table["id_consistent"] & table["slice_is_largest_tumour_area"] & table["volume_matches_bundle"]
    table["slice_index"] = table["preview_slice"].astype(int)
    table["sequence"] = SEQUENCE_LABEL
    table["diagnosis"] = REFERENCE_DIAGNOSIS
    columns = ["case_id", "role", "participant_id", "verified", "diagnosis", "sequence", "slice_index", "age_years", "sex",
               "tumour_volume_ml", "idh1", "mgmt", "pixel_sha256", "source_file", "tumour_area_px_in_slice"]
    table[columns].to_csv(out_dir / "cases.csv", index=False)
    duplicates = table.loc[table["pixel_sha256"].duplicated(keep=False), "case_id"].tolist()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE_CITATION,
        "mirror": MIRROR_BASE,
        "bundle_dir": str(bundle_dir),
        "counts": {
            "bundle_cases": int(len(cases)),
            "exported": int(len(table)),
            "reference": int((table["role"] == "reference").sum()),
            "demo_held_out": int((table["role"] == "demo").sum()),
            "verified": int(table["verified"].sum()),
            "failed": len(failures),
        },
        "failures": failures,
        "unverified_cases": table.loc[~table["verified"], "case_id"].tolist(),
        "identical_slices": duplicates,
        "verification": "participant ID is the numeric counterpart of the mirror subject; recorded slice is the largest "
                        "tumour-area slice of the supplied mask; mask volume equals the bundle's tumour_volume_ml",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return table


# --------------------------------------------------------------------------- index
def build_index(gallery_dir: Path, embedder: Embedder, feature_name: str, batch_log: Callable[[str], None] | None = None) -> int:
    """Embed every verified reference slice. Held-out demo slices are never indexed."""
    gallery_dir = Path(gallery_dir)
    cases = _read_cases(gallery_dir)
    reference = cases[(cases["role"] == "reference") & cases["verified"]]
    if reference.empty:
        raise Gallery2DError("No verified reference slices to index.")
    vectors = []
    for number, case_id in enumerate(reference["case_id"], start=1):
        with Image.open(gallery_dir / "reference" / f"{case_id}.png") as image:
            vectors.append(np.asarray(embedder(image.convert("RGB")), dtype=np.float32).ravel())
        if batch_log and number % 25 == 0:
            batch_log(f"embedded {number}/{len(reference)}")
    np.savez_compressed(
        gallery_dir / "index.npz",
        case_ids=np.asarray(reference["case_id"].tolist()),
        embeddings=_normalise(np.stack(vectors)),
        feature_name=np.asarray(feature_name),
    )
    manifest_path = gallery_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["index"] = {"feature": feature_name, "cases": int(len(reference)), "dimensions": int(len(vectors[0])),
                         "metric": "cosine similarity of L2-normalised image embeddings",
                         "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(reference)


def _read_cases(gallery_dir: Path) -> pd.DataFrame:
    path = Path(gallery_dir) / "cases.csv"
    if not path.is_file():
        raise Gallery2DError(
            f"2D reference gallery not found at {Path(gallery_dir)}. Build it with: "
            "python -m retrieval.gallery2d prepare"
        )
    cases = pd.read_csv(path, dtype={"case_id": str, "participant_id": str})
    cases["verified"] = cases["verified"].astype(bool)
    return cases


@dataclass
class Match:
    rank: int
    case_id: str
    similarity: float
    image_path: Path
    details: dict[str, str]


@dataclass
class SearchResult:
    matches: list[Match]
    excluded: list[str] = field(default_factory=list)  # reference cases identical to the upload


@dataclass
class Gallery2D:
    directory: Path
    cases: pd.DataFrame
    case_ids: list[str]
    embeddings: np.ndarray
    feature_name: str
    manifest: dict

    @classmethod
    def load(cls, directory: str | Path = DEFAULT_GALLERY_DIR, expected_feature: str | None = None) -> "Gallery2D":
        directory = Path(directory)
        cases = _read_cases(directory)
        index_path = directory / "index.npz"
        if not index_path.is_file():
            raise Gallery2DError(
                f"The reference slices exist but the search index ({index_path}) is missing. "
                "Build it with: python -m retrieval.gallery2d index"
            )
        with np.load(index_path, allow_pickle=False) as data:
            case_ids = [str(c) for c in data["case_ids"]]
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            feature_name = str(data["feature_name"])
        if expected_feature and feature_name != expected_feature:
            raise Gallery2DError(
                f"The search index was built with '{feature_name}' but the app uses '{expected_feature}'. "
                "Rebuild it with: python -m retrieval.gallery2d index"
            )
        by_id = cases.set_index("case_id")
        unknown = [c for c in case_ids if c not in by_id.index]
        leaked = [c for c in case_ids if c in by_id.index and (by_id.loc[c, "role"] != "reference" or not by_id.loc[c, "verified"])]
        if unknown or leaked:
            raise Gallery2DError(f"The search index contains cases that must not be searchable: {(unknown + leaked)[:5]}")
        missing_images = [c for c in case_ids if not (directory / "reference" / f"{c}.png").is_file()]
        if missing_images:
            raise Gallery2DError(f"Reference slices are missing for: {missing_images[:5]}")
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        return cls(directory, cases, case_ids, embeddings, feature_name, manifest)

    # ----------------------------------------------------------------- demo scans
    def demo_cases(self) -> list[str]:
        demo = self.cases[(self.cases["role"] == "demo") & self.cases["verified"]]["case_id"]
        return [c for c in demo if (self.directory / "demo" / f"{c}.png").is_file()]

    def demo_path(self, case_id: str) -> Path:
        if case_id not in self.demo_cases():
            raise Gallery2DError(f"{case_id} is not a held-out demo case.")
        return self.directory / "demo" / f"{case_id}.png"

    # ----------------------------------------------------------------- search
    def details(self, case_id: str) -> dict[str, str]:
        """Documented reference fields, in display order; missing values are omitted."""
        row = self.cases.set_index("case_id").loc[case_id]

        def present(value) -> bool:
            return value is not None and not pd.isna(value) and str(value).strip() != ""

        out: dict[str, str] = {"Documented diagnosis": str(row["diagnosis"])}
        if present(row.get("age_years")):
            out["Age at scan"] = f"{float(row['age_years']):.0f} years"
        if present(row.get("sex")):
            out["Sex"] = {"F": "Female", "M": "Male"}.get(str(row["sex"]), str(row["sex"]))
        out["MRI sequence"] = f"{row['sequence']} slice {int(row['slice_index'])}"
        if present(row.get("tumour_volume_ml")):
            out["Tumour volume (from supplied mask)"] = f"{float(row['tumour_volume_ml']):.1f} mL"
        if present(row.get("idh1")):
            out["IDH1"] = str(row["idh1"])
        if present(row.get("mgmt")):
            out["MGMT promoter"] = str(row["mgmt"])
        out["Source ID"] = str(row["participant_id"])
        return out

    def search(self, query_embedding: np.ndarray, query_image: Image.Image | None = None, top_k: int = 5) -> SearchResult:
        """Rank reference slices by cosine similarity; drop references identical to the query."""
        query = _normalise(np.asarray(query_embedding, dtype=np.float32).ravel())
        if query.shape[0] != self.embeddings.shape[1]:
            raise Gallery2DError("The query embedding does not match the index dimensions; rebuild the index.")
        similarities = self.embeddings @ query
        hashes = self.cases.set_index("case_id")["pixel_sha256"]
        query_hash = pixel_sha256(query_image) if query_image is not None else None
        matches, excluded = [], []
        for position in np.argsort(-similarities, kind="stable"):
            case_id, similarity = self.case_ids[position], float(similarities[position])
            if (query_hash is not None and hashes[case_id] == query_hash) or similarity >= NEAR_DUPLICATE_SIMILARITY:
                excluded.append(case_id)
                continue
            matches.append(Match(len(matches) + 1, case_id, similarity, self.directory / "reference" / f"{case_id}.png", self.details(case_id)))
            if len(matches) == top_k:
                break
        return SearchResult(matches, excluded)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the 2D reference gallery for JPG/PNG queries")
    parser.add_argument("command", choices=["prepare", "slices", "index"],
                        help="slices: export clean slices; index: embed them; prepare: both")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_GALLERY_DIR)
    args = parser.parse_args()
    if args.command in ("prepare", "slices"):
        table = prepare_slices(args.bundle, args.out)
        print(json.dumps(json.loads((args.out / "manifest.json").read_text())["counts"], indent=2))
        if not table["verified"].all():
            print("Unverified cases are excluded from search:", table.loc[~table["verified"], "case_id"].tolist())
    if args.command in ("prepare", "index"):
        from inference import EMBEDDING_NAME, ModelSet

        models = ModelSet.load()
        count = build_index(args.out, models.embed, EMBEDDING_NAME, print)
        print(f"indexed {count} reference slices with '{EMBEDDING_NAME}'")


if __name__ == "__main__":
    main()

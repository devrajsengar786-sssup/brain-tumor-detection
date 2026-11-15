"""Streamlit page: search the real UPENN-GBM gallery bundle with one examination."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st

from .bundle import BundleError, GalleryBundle, fetch_bundle_zip
from .data import CaseRecord, ManifestError, records_to_frame
from .features import describe_feature
from .pipeline import build_engineered_reference_database, query_engineered_reference_database
from .query import ROLE_LABELS, detect_role, files_from_zip, missing_roles, prepare_query


DEFAULT_BUNDLE_DIRECTORY = Path("artifacts/gallery_bundle")
BUNDLE_CACHE_DIRECTORY = Path("artifacts/gallery_bundle_cache")
FAMILY_LABELS = {"shape": "Shape & size", "location": "Location", "intensity": "Signal intensity"}


# --------------------------------------------------------------------------- bundle loading
def _setting(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:
        return None


def resolve_bundle_directory() -> Path:
    url = _setting("GALLERY_BUNDLE_URL")
    if url:
        return fetch_bundle_zip(url, BUNDLE_CACHE_DIRECTORY)
    return Path(_setting("GALLERY_BUNDLE_DIR") or DEFAULT_BUNDLE_DIRECTORY)


@st.cache_resource(show_spinner="Loading the reference gallery...")
def _load_bundle(directory: str, stamp: float) -> GalleryBundle:
    return GalleryBundle.load(directory)


def load_gallery() -> GalleryBundle:
    directory = resolve_bundle_directory()
    header = directory / "bundle.json"
    return _load_bundle(str(directory), header.stat().st_mtime if header.is_file() else 0.0)


# --------------------------------------------------------------------------- query preparation
@st.cache_data(show_spinner=False, max_entries=4)
def _prepare_cached(files: tuple[tuple[str, str, bytes], ...]):
    prepared = prepare_query({role: (name, payload) for role, name, payload in files})
    return prepared


def collect_uploads(uploads) -> tuple[dict[str, tuple[str, bytes]], list[str]]:
    """Return role -> (name, bytes) and a list of problems (duplicates, unknown files)."""
    files: dict[str, tuple[str, bytes]] = {}
    problems: list[str] = []
    for upload in uploads or []:
        name, payload = upload.name, upload.getvalue()
        if name.lower().endswith(".zip"):
            try:
                for role, item in files_from_zip(payload).items():
                    if role in files:
                        problems.append(f"Two {ROLE_LABELS[role]} files were supplied.")
                    files[role] = item
            except ManifestError as error:
                problems.append(str(error))
            continue
        role = detect_role(name)
        if role is None:
            problems.append(f"Could not tell which sequence '{name}' is. Include T1, T1-Gd/ce-gd, T2, FLAIR or seg in the filename.")
        elif role in files:
            problems.append(f"Two {ROLE_LABELS[role]} files were supplied ({files[role][0]}, {name}).")
        else:
            files[role] = (name, payload)
    return files, problems


# --------------------------------------------------------------------------- rendering
def _measurement_row(container, measurements: dict) -> None:
    columns = container.columns(4)
    columns[0].metric("Tumour volume", f"{measurements.get('tumour_volume_ml', float('nan')):.1f} mL")
    columns[1].metric("Enhancing", f"{measurements.get('et_volume_ml', float('nan')):.1f} mL")
    columns[2].metric("Necrotic core", f"{measurements.get('ncr_volume_ml', float('nan')):.1f} mL")
    columns[3].metric("Oedema", f"{measurements.get('ed_volume_ml', float('nan')):.1f} mL")


def _metadata_caption(row: pd.Series) -> str:
    if not bool(row.get("join_verified", False)):
        return "Clinical metadata: not shown (no verified join)."
    parts = []
    if isinstance(row.get("sex"), str):
        parts.append(f"sex {row['sex']}")
    age = row.get("age_years")
    if age is not None and not pd.isna(age):
        parts.append(f"age {float(age):.0f}")
    return "Demographics: " + (", ".join(parts) if parts else "fields missing in participants.tsv")


def render_results(bundle: GalleryBundle, query_features: pd.DataFrame, results: pd.DataFrame, excluded: list[str]) -> None:
    if excluded:
        st.caption(
            f"Excluded from results because they belong to the query patient or are the same examination: {', '.join(excluded)}"
        )
    if results.empty:
        if bundle.cutoff is not None:
            st.warning(f"No close matches: no reference case reached the validated similarity cutoff ({bundle.cutoff:.2f}).")
        else:
            st.info("No reference cases were returned.")
        return
    st.markdown(f"##### Top {len(results)} similar reference cases")
    for rank, match in enumerate(results.itertuples(index=False), start=1):
        row = pd.Series(match._asdict())
        with st.container(border=True):
            left, right = st.columns([1, 1])
            with left:
                st.markdown(f"**#{rank} · {row['case_id']}** · similarity **{row['similarity']:.3f}**")
                preview = bundle.preview_path(row["case_id"])
                if preview:
                    st.image(str(preview), caption="T1-Gd | FLAIR · yellow necrotic core, blue oedema, red enhancing", use_container_width=True)
                st.caption(
                    f"Volume {row.get('tumour_volume_ml', float('nan')):.1f} mL · {_metadata_caption(row)}"
                )
            with right:
                _render_explanation(bundle, query_features, row)
    if bundle.index.space.metric == "cosine":
        note = "Contributions sum to the similarity; negative values pull it down."
    else:
        note = (
            "Distance is measured in gallery standard deviations: 0 means identical, about 2 is a typical unrelated "
            "pair. Family distances add up to the total, and similarity = exp(-distance / 2)."
        )
    st.caption(note + " Similar imaging features do not imply the same diagnosis, prognosis or treatment.")


def _render_explanation(bundle: GalleryBundle, query_features: pd.DataFrame, row: pd.Series) -> None:
    index = bundle.index
    families = list(index.space.families)
    terms = [float(row.get(f"{index.term_prefix}{f}", 0.0)) for f in families]
    label = "Contribution" if index.space.metric == "cosine" else "Distance"
    st.markdown("**Why it matched**")
    st.dataframe(
        pd.DataFrame({"Feature family": [FAMILY_LABELS.get(f, f) for f in families], label: terms}).style.format({label: "{:.3f}"}),
        hide_index=True,
        use_container_width=True,
    )
    def table(frame: pd.DataFrame) -> None:
        frame = frame.assign(Feature=frame["feature"].map(describe_feature))
        st.dataframe(
            frame[["Feature", "query_value", "reference_value"]]
            .rename(columns={"query_value": "Query", "reference_value": "Match"})
            .style.format({"Query": "{:.2f}", "Match": "{:.2f}"}),
            hide_index=True,
            use_container_width=True,
        )
    if index.space.metric == "cosine":
        table(index.explain(query_features, row["case_id"], top_n=5))
        return
    st.markdown("Most alike")
    table(index.explain(query_features, row["case_id"], top_n=3, closest=True))
    st.markdown("Most different")
    table(index.explain(query_features, row["case_id"], top_n=3))


def _run_and_render(bundle: GalleryBundle, query_features: pd.DataFrame, top_k: int, exclude_patient: str | None) -> None:
    results, excluded = bundle.search(query_features, top_k=top_k, exclude_patient_id=exclude_patient)
    render_results(bundle, query_features, results, excluded)


def render_demo_query(bundle: GalleryBundle, top_k: int) -> None:
    if bundle.demo_features.empty:
        st.info("This bundle has no built-in held-out demo queries.")
        return
    case_id = st.selectbox("Held-out test query (never part of the gallery)", bundle.demo_features.index.tolist(), key="demo_query_id")
    meta = bundle.demo_metadata.reindex([case_id]).iloc[0]
    st.markdown(f"##### Query · {case_id}")
    left, right = st.columns([3, 2])
    preview = bundle.preview_path(case_id)
    if preview:
        left.image(str(preview), caption="Query T1-Gd | FLAIR with supplied mask", use_container_width=True)
    right.caption(_metadata_caption(meta))
    _measurement_row(st, meta.to_dict())
    patient = meta.get("patient_id") if isinstance(meta.get("patient_id"), str) else case_id
    _run_and_render(bundle, bundle.demo_features.loc[[case_id]], top_k, patient)


def render_upload_query(bundle: GalleryBundle, top_k: int) -> None:
    st.caption(
        "Upload one examination: either a .zip with the five files, or the five NIfTI files together. "
        "Filenames must say which is which (e.g. `_T1w`, `_ce-gd_T1w`/`t1gd`, `_T2w`, `_FLAIR`, `_seg`). "
        "Scans must be skull-stripped and registered to the gallery atlas (UPENN-GBM: SRI24, 240×240×155, 1 mm). "
        "Do not upload identifiable patient data."
    )
    uploads = st.file_uploader(
        "Examination files", type=["nii", "gz", "zip"], accept_multiple_files=True, key="exam_upload"
    )
    if not uploads:
        return
    files, problems = collect_uploads(uploads)
    for problem in problems:
        st.error(problem)
    st.dataframe(
        pd.DataFrame([{"Input": ROLE_LABELS[r], "File": files[r][0] if r in files else "— missing —"} for r in ROLE_LABELS]),
        hide_index=True,
        use_container_width=True,
    )
    missing = missing_roles(files)
    if missing:
        st.error("Missing input(s): " + ", ".join(missing) + ". Retrieval runs automatically once all five are present.")
        return
    if problems:
        return
    try:
        with st.spinner("Checking alignment and extracting tumour features..."):
            prepared = _prepare_cached(tuple((role, name, payload) for role, (name, payload) in sorted(files.items())))
        bundle.check_geometry(prepared.shape, prepared.affine)
    except (ManifestError, BundleError) as error:
        st.error(f"This examination cannot be searched: {error}")
        return
    except Exception as error:  # unexpected file problems should not crash the page
        st.error(f"This examination could not be processed: {error}")
        return
    st.markdown("##### Uploaded query")
    st.image(prepared.preview, caption="Query T1-Gd | FLAIR with supplied mask", use_container_width=True)
    _measurement_row(st, prepared.measurements)
    if prepared.patient_hint:
        st.caption(f"Filename suggests subject {prepared.patient_hint}; that patient is excluded from the results.")
    _run_and_render(bundle, prepared.features, top_k, prepared.patient_hint)


def render_retrieval_tab() -> None:
    """Research-only retrieval interface backed by the exported gallery bundle."""
    st.markdown("#### 📚 Similar Glioblastoma Research Cases (UPENN-GBM)")
    st.caption(
        "Research prototype. Cases are ranked by interpretable tumour shape, location and intensity features "
        "computed from a supplied subregion mask. Similarity is not clinical equivalence, diagnosis, prognosis "
        "or a treatment recommendation."
    )
    try:
        bundle = load_gallery()
    except BundleError as error:
        st.error(f"The reference gallery is unavailable: {error}")
        st.info(
            "Build the bundle with `notebooks/upenn_gbm_gallery_colab.ipynb`, then either unzip it to "
            "`artifacts/gallery_bundle/`, or set `GALLERY_BUNDLE_DIR` (a folder) or `GALLERY_BUNDLE_URL` "
            "(a zip link) as an environment variable or Streamlit secret."
        )
        if st.toggle("Show the synthetic pipeline demo instead (generated volumes, not patients)", key="synthetic_toggle"):
            _render_missing_index_demo()
        return

    counts = bundle.manifest.get("counts", {})
    st.success(
        f"Gallery loaded: {len(bundle.index.case_ids)} reference cases · {len(bundle.demo_features)} held-out demo queries · "
        f"feature set {bundle.header.get('feature_set_version')}"
    )
    with st.expander("How this gallery was built"):
        st.json(
            {
                "counts": counts,
                "fusion": bundle.manifest.get("fusion"),
                "ranking_inputs": bundle.manifest.get("ranking_inputs"),
                "similarity_cutoff": bundle.header.get("similarity_cutoff") or "not validated: all Top-K shown",
            }
        )
    top_k = st.slider("Number of similar cases", min_value=1, max_value=min(10, len(bundle.index.case_ids)), value=min(5, len(bundle.index.case_ids)), key="gallery_top_k")
    mode = st.radio("Query", ["Built-in held-out demo query", "Upload an examination"], horizontal=True, key="gallery_mode")
    if mode.startswith("Built-in"):
        render_demo_query(bundle, top_k)
    else:
        render_upload_query(bundle, top_k)


# --------------------------------------------------------------------------- synthetic fallback
def _tumour_slice(image_path: str | Path, mask_path: str | Path) -> np.ndarray:
    image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32) > 0
    if image.shape != mask.shape or not mask.any():
        raise ValueError("The selected reference image and mask cannot produce a tumour slice.")
    slice_index = int(np.argmax(mask.sum(axis=(0, 1))))
    raw_slice = image[:, :, slice_index]
    lower, upper = np.percentile(raw_slice, (1, 99))
    normalized = np.clip((raw_slice - lower) / max(upper - lower, 1e-6), 0, 1)
    rgb = np.repeat((normalized * 255).astype(np.uint8)[..., np.newaxis], 3, axis=-1)
    rgb[mask[:, :, slice_index], :] = np.array([255, 86, 86], dtype=np.uint8)
    return rgb


def _write_synthetic_case(
    directory: Path,
    case_id: str,
    centre: tuple[float, float, float],
    radii: tuple[float, float, float],
    tumour_intensity: float,
    seed: int,
) -> CaseRecord:
    """Create one clearly labelled, non-patient test case with three subregions."""
    shape = (48, 48, 32)
    grid_x, grid_y, grid_z = np.indices(shape, dtype=np.float32)
    distance = np.sqrt(
        ((grid_x - centre[0]) / radii[0]) ** 2 + ((grid_y - centre[1]) / radii[1]) ** 2 + ((grid_z - centre[2]) / radii[2]) ** 2
    )
    labels = np.zeros(shape, dtype=np.uint8)
    labels[distance <= 1.0] = 2  # oedema
    labels[distance <= 0.6] = 3  # enhancing rim
    labels[distance <= 0.35] = 1  # necrotic core
    mask = labels > 0
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    generator = np.random.default_rng(seed)
    modality_paths = []
    for modality_index, modality_name in enumerate(("t1", "t1gd", "t2", "flair"), start=1):
        volume = (45.0 + 0.24 * grid_x + 0.10 * grid_y + 0.18 * grid_z + generator.normal(0.0, 2.0, shape)).astype(np.float32)
        volume[mask] += tumour_intensity * (0.65 + modality_index * 0.18)
        path = directory / f"{case_id}_{modality_name}.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine), path)
        modality_paths.append(path)
    mask_path = directory / f"{case_id}_mask.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), mask_path)
    return CaseRecord(case_id, case_id, tuple(modality_paths), mask_path, "reference")


def _run_synthetic_retrieval_demo() -> tuple[pd.DataFrame, dict[str, float], np.ndarray, dict[str, np.ndarray]]:
    """Exercise the full retrieval path with local synthetic volumes, never patient data."""
    with tempfile.TemporaryDirectory(prefix="brain_mri_synthetic_demo_") as temporary_directory:
        root = Path(temporary_directory)
        references = [
            _write_synthetic_case(root, "synthetic-reference-a", (22, 23, 16), (8, 7, 5), 34, 11),
            _write_synthetic_case(root, "synthetic-reference-b", (30, 18, 12), (5, 9, 4), 60, 22),
            _write_synthetic_case(root, "synthetic-reference-c", (14, 31, 20), (11, 5, 7), 20, 33),
        ]
        manifest_path = root / "reference_manifest.csv"
        records_to_frame(references).to_csv(manifest_path, index=False)
        index_directory = root / "reference_index"
        build_engineered_reference_database(manifest_path, index_directory, reference_split="reference")

        query = _write_synthetic_case(root, "synthetic-query", (22, 24, 16), (8, 7, 5), 35, 44)
        query = CaseRecord(query.case_id, query.patient_id, query.modalities, query.mask_path, "query")
        results, query_features = query_engineered_reference_database(index_directory, query, top_k=len(references))
        query_preview = _tumour_slice(query.modalities[3], query.mask_path)
        reference_previews = {record.case_id: _tumour_slice(record.modalities[3], record.mask_path) for record in references}
    return results, query_features, query_preview, reference_previews


def _render_missing_index_demo() -> None:
    try:
        with st.spinner("Creating synthetic MRI volumes and retrieving similar test cases..."):
            results, query_features, query_preview, reference_previews = _run_synthetic_retrieval_demo()
    except Exception as error:
        st.error(f"The synthetic retrieval demonstration could not be completed: {error}")
        return
    st.warning("Synthetic demo: the cases below are generated volumes, not patients.")
    st.image(query_preview, caption="Synthetic query FLAIR slice with mask", use_container_width=True)
    st.dataframe(results[["case_id", "similarity"]], hide_index=True, use_container_width=True)
    for case_id, preview in reference_previews.items():
        st.image(preview, caption=case_id, width=220)

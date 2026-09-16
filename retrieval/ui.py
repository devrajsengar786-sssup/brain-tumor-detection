"""Streamlit components for inspecting a prepared similar-case reference database."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st

from .data import CaseRecord, records_to_frame
from .pipeline import build_engineered_reference_database, query_engineered_reference_database


DEFAULT_INDEX_DIRECTORY = Path("artifacts/retrieval/engineered_baseline")


def _save_upload(upload, directory: Path, filename: str) -> Path:
    destination = directory / filename
    destination.write_bytes(upload.getvalue())
    return destination


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
    """Create one clearly labelled, non-patient test case for the in-app demo."""
    shape = (48, 48, 32)
    grid_x, grid_y, grid_z = np.indices(shape, dtype=np.float32)
    mask = (
        ((grid_x - centre[0]) / radii[0]) ** 2
        + ((grid_y - centre[1]) / radii[1]) ** 2
        + ((grid_z - centre[2]) / radii[2]) ** 2
        <= 1
    )
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    generator = np.random.default_rng(seed)
    modality_paths = []
    for modality_index, modality_name in enumerate(("t1", "t1gd", "t2", "flair"), start=1):
        volume = (
            45.0
            + 0.24 * grid_x
            + 0.10 * grid_y
            + 0.18 * grid_z
            + generator.normal(0.0, 2.0, shape)
        ).astype(np.float32)
        volume[mask] += tumour_intensity * (0.65 + modality_index * 0.18)
        path = directory / f"{case_id}_{modality_name}.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine), path)
        modality_paths.append(path)
    mask_path = directory / f"{case_id}_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), mask_path)
    return CaseRecord(case_id, case_id, tuple(modality_paths), mask_path, "reference")


def _run_synthetic_retrieval_demo() -> tuple[pd.DataFrame, dict[str, float], np.ndarray, dict[str, np.ndarray]]:
    """Exercise the full retrieval path with local synthetic volumes, never patient data."""
    with tempfile.TemporaryDirectory(prefix="brain_mri_synthetic_demo_") as temporary_directory:
        root = Path(temporary_directory)
        references = [
            _write_synthetic_case(root, "synthetic-reference-a", (22, 23, 16), (8, 7, 5), 34, 11),
            _write_synthetic_case(root, "synthetic-reference-b", (27, 20, 15), (6, 9, 4), 48, 22),
            _write_synthetic_case(root, "synthetic-reference-c", (17, 29, 18), (10, 5, 6), 28, 33),
        ]
        manifest_path = root / "reference_manifest.csv"
        records_to_frame(references).to_csv(manifest_path, index=False)
        index_directory = root / "reference_index"
        build_engineered_reference_database(manifest_path, index_directory, reference_split="reference")

        query = _write_synthetic_case(root, "synthetic-query", (22, 24, 16), (8, 7, 5), 35, 44)
        query = CaseRecord(
            query.case_id, query.patient_id, query.modalities, query.mask_path, "query"
        )
        results, query_features = query_engineered_reference_database(
            index_directory, query, top_k=len(references)
        )
        query_preview = _tumour_slice(query.modalities[3], query.mask_path)
        reference_previews = {
            record.case_id: _tumour_slice(record.modalities[3], record.mask_path)
            for record in references
        }
    return results, query_features, query_preview, reference_previews


def _render_retrieval_results(
    results: pd.DataFrame,
    query_features: dict[str, float],
    query_preview: np.ndarray,
    reference_previews: dict[str, np.ndarray] | None = None,
) -> None:
    """Present retrieved cases and auditable query measurements."""
    st.markdown("##### Query tumour measurements")
    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Tumour volume", f"{query_features['morph_volume_ml']:.2f} mL")
    metric_middle.metric("Bounding-box extent", f"{query_features['morph_bbox_extent']:.2f}")
    metric_right.metric("Boundary fraction", f"{query_features['morph_boundary_voxel_fraction']:.2f}")
    st.image(
        query_preview,
        caption="Query FLAIR slice with the supplied tumour segmentation mask",
        use_container_width=True,
    )

    st.markdown("##### Top similar reference cases")
    if results.empty:
        st.info("No reference cases were returned.")
        return
    st.dataframe(results[["case_id", "patient_id", "similarity"]], use_container_width=True, hide_index=True)
    for _, result in results.iterrows():
        case_id = str(result["case_id"])
        with st.expander(f"{case_id} - similarity {result['similarity']:.3f}"):
            try:
                preview = (reference_previews or {}).get(case_id)
                if preview is None:
                    preview = _tumour_slice(result["flair_path"], result["mask_path"])
                st.image(
                    preview,
                    caption="Reference FLAIR slice with tumour mask overlay",
                    use_container_width=True,
                )
            except Exception as error:
                st.warning(f"Reference image preview unavailable: {error}")


def _render_missing_index_demo() -> None:
    """Keep the research tab demonstrably functional before a real index is supplied."""
    st.info(
        "A persistent UPENN-GBM reference database is not present in this project yet. "
        "You can still run the complete retrieval workflow below using built-in synthetic "
        "test volumes; no patient data are used."
    )
    if not st.button("Run built-in synthetic retrieval demo", type="primary", key="synthetic_retrieval_demo"):
        st.caption(
            "For real research cases, build the patient-level UPENN index with "
            "`python -m retrieval.cli build-engineered-index`, then restart the app."
        )
        return
    try:
        with st.spinner("Creating synthetic MRI volumes and retrieving similar test cases..."):
            results, query_features, query_preview, reference_previews = _run_synthetic_retrieval_demo()
    except Exception as error:
        st.error(f"The synthetic retrieval demonstration could not be completed: {error}")
        return

    st.success("Synthetic retrieval demo completed. The cases below are generated test data, not patients.")
    _render_retrieval_results(results, query_features, query_preview, reference_previews)


def render_retrieval_tab(index_directory: str | Path = DEFAULT_INDEX_DIRECTORY) -> None:
    """Render a research-only retrieval interface without making clinical claims."""
    index_directory = Path(index_directory)
    st.markdown("#### 📚 Similar Glioblastoma Research Cases")
    st.caption(
        "This research prototype retrieves reference MRIs by tumour morphology and "
        "intensity features. Similarity is not clinical equivalence, diagnosis, prognosis, "
        "or a treatment recommendation."
    )

    if not (index_directory / "retrieval_index.joblib").is_file():
        _render_missing_index_demo()
        return

    st.warning(
        "The query requires an aligned tumour mask produced by the project's segmentation "
        "workflow. Do not upload patient-identifiable clinical data to this local research demo."
    )
    modality_columns = st.columns(2)
    uploads = {}
    for column, (key, label) in zip(
        modality_columns * 2,
        (("t1", "T1"), ("t1gd", "T1GD"), ("t2", "T2"), ("flair", "FLAIR")),
    ):
        with column:
            uploads[key] = st.file_uploader(f"{label} NIfTI", type=["nii", "gz"], key=f"retrieval_{key}")
    mask_upload = st.file_uploader(
        "Tumour segmentation mask (NIfTI)", type=["nii", "gz"], key="retrieval_mask"
    )
    query_case_id = st.text_input("Research query case ID", value="query-case")
    top_k = st.slider("Number of similar cases", min_value=1, max_value=10, value=5)

    if not st.button("Retrieve similar cases", type="primary", key="retrieve_cases"):
        return
    if not all(uploads.values()) or mask_upload is None:
        st.error("Upload all four MRI modalities and the aligned segmentation mask.")
        return

    with tempfile.TemporaryDirectory(prefix="brain_mri_retrieval_") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        modality_paths = tuple(
            _save_upload(uploads[name], temporary_directory, f"query_{name}.nii.gz")
            for name in ("t1", "t1gd", "t2", "flair")
        )
        mask_path = _save_upload(mask_upload, temporary_directory, "query_mask.nii.gz")
        query_record = CaseRecord(query_case_id, query_case_id, modality_paths, mask_path, "query")
        try:
            with st.spinner("Extracting tumour-aware features and searching reference cases..."):
                results, query_features = query_engineered_reference_database(
                    index_directory, query_record, top_k=top_k
                )
            query_preview = _tumour_slice(modality_paths[3], mask_path)
        except Exception as error:
            st.error(f"Retrieval could not be completed: {error}")
            return

    _render_retrieval_results(results, query_features, query_preview)

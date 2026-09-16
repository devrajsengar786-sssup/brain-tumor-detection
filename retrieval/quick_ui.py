"""A lightweight, usable 2D MRI similar-image demo for the Streamlit app."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image


PUBLIC_DATASET_ID = "akar49/MRI_Classification"
PUBLIC_REFERENCE_COUNT_PER_CLASS = 30


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return np.asarray(image, dtype=np.uint8)


def _crop_nonblack_region(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coordinates = cv2.findNonZero((gray > 10).astype(np.uint8))
    if coordinates is None:
        return image
    x, y, width, height = cv2.boundingRect(coordinates)
    cropped = image[y : y + height, x : x + width]
    return cropped if cropped.size else image


def _edge_orientation_feature(grayscale: np.ndarray) -> np.ndarray:
    """A small HOG-style feature implemented with standard OpenCV operations."""
    gradient_x = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)
    orientation_bins = np.minimum((orientation * 9 / np.pi).astype(np.int32), 8)

    cell_histograms: list[np.ndarray] = []
    for y_start in range(0, 64, 8):
        for x_start in range(0, 64, 8):
            cell_bins = orientation_bins[y_start : y_start + 8, x_start : x_start + 8]
            cell_magnitudes = magnitude[y_start : y_start + 8, x_start : x_start + 8]
            cell_histograms.append(
                np.bincount(
                    cell_bins.ravel(), weights=cell_magnitudes.ravel(), minlength=9
                ).astype(np.float32)
            )
    return np.concatenate(cell_histograms)


def quick_similarity_feature(image: Image.Image | np.ndarray) -> np.ndarray:
    """Create a deterministic visual baseline from MRI shape, edges, and intensity."""
    rgb_image = _crop_nonblack_region(_as_rgb_array(image))
    grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    grayscale = cv2.resize(grayscale, (64, 64), interpolation=cv2.INTER_AREA)
    edge_features = _edge_orientation_feature(grayscale)
    histogram = cv2.calcHist([grayscale], [0], None, [32], [0, 256]).reshape(-1)
    feature = np.concatenate([edge_features, histogram.astype(np.float32)])
    norm = np.linalg.norm(feature)
    return feature / norm if norm else feature


@st.cache_data(show_spinner=False)
def load_public_reference_library(per_class: int = PUBLIC_REFERENCE_COUNT_PER_CLASS) -> list[dict]:
    """Download a small, deterministic reference set already used by this project."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The public demo library needs the 'datasets' package. "
            "Run `pip install -r requirements.txt` after pulling the latest changes."
        ) from error

    dataset = load_dataset(PUBLIC_DATASET_ID)
    references: list[dict] = []
    selected_per_label = {0: 0, 1: 0}
    label_names = {0: "No Tumor", 1: "Tumor"}
    for split_name in ("train", "test"):
        if split_name not in dataset:
            continue
        for row_index, example in enumerate(dataset[split_name]):
            label = int(example["label"])
            if label not in selected_per_label or selected_per_label[label] >= per_class:
                continue
            image = _as_rgb_array(example["image"])
            references.append(
                {
                    "case_id": f"{split_name}-{row_index}",
                    "label": label_names[label],
                    "image": image,
                }
            )
            selected_per_label[label] += 1
            if all(count >= per_class for count in selected_per_label.values()):
                return references
    if len(references) < 2:
        raise RuntimeError("The public dataset did not provide enough images for retrieval.")
    return references


def _uploaded_reference_library(uploaded_files) -> list[dict]:
    references = []
    for position, uploaded_file in enumerate(uploaded_files):
        image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
        references.append(
            {
                "case_id": uploaded_file.name or f"reference-{position + 1}",
                "label": "User supplied",
                "image": _as_rgb_array(image),
            }
        )
    return references


def _rank_similar_images(query_image: Image.Image, references: list[dict], top_k: int) -> list[dict]:
    query_feature = quick_similarity_feature(query_image)
    ranked = []
    for reference in references:
        similarity = float(query_feature @ quick_similarity_feature(reference["image"]))
        ranked.append({**reference, "similarity": similarity})
    return sorted(ranked, key=lambda item: item["similarity"], reverse=True)[:top_k]


def render_quick_similarity_demo() -> None:
    """Render a self-service image-retrieval demo that works without UPENN-GBM data."""
    st.markdown("#### 🖼️ 2D MRI Similarity Demo")
    st.caption(
        "Find visually similar MRI reference images using shape, edge, and intensity "
        "features. This is a demonstration of image similarity only - not diagnosis, "
        "tumour subtype, patient matching, prognosis, or treatment advice."
    )

    reference_source = st.radio(
        "Reference library",
        ["Public MRI sample library", "My uploaded reference images"],
        horizontal=True,
        key="quick_retrieval_source",
    )
    references: list[dict] = []
    if reference_source == "Public MRI sample library":
        if st.button("Load public MRI reference library", key="load_public_reference_library"):
            try:
                with st.spinner("Downloading a small public MRI reference library..."):
                    st.session_state["quick_public_references"] = load_public_reference_library()
            except Exception as error:
                st.error(f"Could not load the public reference library: {error}")
        references = st.session_state.get("quick_public_references", [])
        if references:
            st.success(f"Loaded {len(references)} reference MRIs from the project dataset.")
        else:
            st.info("Load the public reference library, or switch to your own reference images.")
    else:
        uploaded_references = st.file_uploader(
            "Upload at least two reference MRI images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="quick_reference_uploads",
        )
        if uploaded_references:
            try:
                references = _uploaded_reference_library(uploaded_references)
            except Exception as error:
                st.error(f"One or more reference images could not be read: {error}")

    query_upload = st.file_uploader(
        "Upload an MRI image to search",
        type=["jpg", "jpeg", "png"],
        key="quick_query_upload",
    )
    top_k = st.slider("Similar images to show", min_value=1, max_value=10, value=5, key="quick_top_k")

    if query_upload is None or len(references) < 2:
        return
    if not st.button("Find similar MRI images", type="primary", key="quick_similarity_search"):
        return

    try:
        query_image = Image.open(BytesIO(query_upload.getvalue())).convert("RGB")
        results = _rank_similar_images(query_image, references, min(top_k, len(references)))
    except Exception as error:
        st.error(f"Similarity search could not be completed: {error}")
        return

    query_column, result_column = st.columns([1, 2])
    with query_column:
        st.image(query_image, caption="Query MRI", use_container_width=True)
    with result_column:
        st.dataframe(
            [
                {
                    "Rank": rank,
                    "Reference case": result["case_id"],
                    "Source label": result["label"],
                    "Visual similarity": f"{result['similarity'] * 100:.1f}%",
                }
                for rank, result in enumerate(results, start=1)
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("##### Similar reference images")
    image_columns = st.columns(min(3, len(results)))
    for column, result in zip(image_columns * ((len(results) + len(image_columns) - 1) // len(image_columns)), results):
        with column:
            st.image(
                result["image"],
                caption=f"{result['case_id']} - {result['similarity'] * 100:.1f}%",
                use_container_width=True,
            )


def render_similarity_page() -> None:
    """Combine a usable 2D demo with the full 3D UPENN-GBM research pathway."""
    demo_tab, research_tab = st.tabs(["🖼️ 2D MRI Demo", "🧪 UPENN-GBM Research"])
    with demo_tab:
        render_quick_similarity_demo()
    with research_tab:
        try:
            from .ui import render_retrieval_tab
        except ImportError:
            st.info(
                "Install `requirements-retrieval.txt` to use the advanced "
                "UPENN-GBM NIfTI retrieval workflow. The 2D demo remains available."
            )
        else:
            render_retrieval_tab()

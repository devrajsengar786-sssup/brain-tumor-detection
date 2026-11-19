"""Brain MRI Analysis - one upload, one page.

Upload a JPG/PNG slice and the page shows, in order: the detector's result and
confidence, the predicted tumour type, the Grad-CAM overlay, and the closest
reference cases with a side-by-side comparison and documented details.

Layout: an application header, then a two-panel workspace (Scan | Results), then
the Similar cases and Comparison panels. Styling lives in ``assets/app.css``;
Streamlit's own widget colours are in ``.streamlit/config.toml``.
"""

from __future__ import annotations

import runtime_limits  # noqa: F401  (must run before numpy/TensorFlow are imported)

import base64
import hashlib
import hmac
import html
import io
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit.errors import StreamlitSecretNotFoundError

APP_DIR = Path(__file__).parent
MAX_IMAGE_PIXELS = 4096 * 4096
MIN_IMAGE_SIDE = 32
VIEWER_MAX_SIDE = 1024  # display copies are downscaled; analysis always uses the full image
TOP_K = 5

st.set_page_config(page_title="Brain MRI Analysis", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(APP_DIR / 'assets' / 'app.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- access gate
def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return ""


APP_ACCESS_PASSWORD = _secret("APP_ACCESS_PASSWORD")


def require_app_access() -> bool:
    """Optional password gate; unset keeps the public demo mode."""
    if not APP_ACCESS_PASSWORD or st.session_state.get("app_access_granted"):
        return True
    st.markdown('<div class="app-header"><div class="app-title">Brain MRI Analysis</div></div>', unsafe_allow_html=True)
    with st.form("app_access_form"):
        supplied = st.text_input("Access password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Open app", type="primary")
    if submitted:
        if hmac.compare_digest(supplied, APP_ACCESS_PASSWORD):
            st.session_state["app_access_granted"] = True
            st.rerun()
        st.error("Incorrect password. Please try again.")
    return False


if not require_app_access():
    st.stop()


# --------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False)
def get_models():
    from inference import ModelSet

    return ModelSet.load()


LOCAL_GALLERY = APP_DIR / "artifacts" / "gallery_2d"  # written by `python -m retrieval.gallery2d prepare` (git-ignored)
BUNDLED_GALLERY = APP_DIR / "data" / "gallery_2d"  # prebuilt copy shipped with the repository (used on Streamlit Cloud)


def gallery_directory() -> Path:
    """GALLERY_2D_DIR (env or secret) wins; otherwise a locally built gallery; otherwise the bundled one."""
    configured = os.environ.get("GALLERY_2D_DIR") or _secret("GALLERY_2D_DIR")
    if configured:
        return Path(configured)
    if (LOCAL_GALLERY / "cases.csv").is_file():
        return LOCAL_GALLERY
    return BUNDLED_GALLERY


@st.cache_resource(show_spinner=False)
def _load_gallery(directory: str, stamp: float):
    from retrieval.gallery2d import Gallery2D

    return Gallery2D.load(directory)  # TensorFlow-free, so the empty state appears at once


def get_gallery():
    """Return (gallery, None) or (None, reason). The reason is safe to show to the user."""
    from retrieval.gallery2d import Gallery2DError

    directory = gallery_directory()
    index = directory / "index.npz"
    try:
        return _load_gallery(str(directory), index.stat().st_mtime if index.is_file() else 0.0), None
    except Gallery2DError as error:
        return None, str(error)
    except Exception as error:  # a damaged gallery must not take the analysis down with it
        return None, f"The reference library could not be read: {error}"


# --------------------------------------------------------------------------- query handling
def read_image(payload: bytes) -> Image.Image:
    """Decode and sanity-check an upload; raises ValueError with a readable reason."""
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError("This file could not be read as a JPG or PNG image.") from error
    if image.format not in {"JPEG", "PNG"}:
        raise ValueError(f"Only JPG and PNG images are supported (this file is {image.format}).")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("This image is too large. Please upload one under 4096 x 4096 pixels.")
    if min(image.size) < MIN_IMAGE_SIDE:
        raise ValueError(f"This image is too small to analyse (minimum {MIN_IMAGE_SIDE} pixels per side).")
    return image.convert("RGB")


def _on_upload() -> None:
    st.session_state.pop("demo_case", None)


def _start_demo(case_id: str) -> None:
    st.session_state["demo_case"] = case_id
    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1  # clears the uploader


def _clear() -> None:
    st.session_state.pop("demo_case", None)
    st.session_state.pop("result", None)
    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1


def _select_case(case_id: str) -> None:
    st.session_state["selected_case"] = case_id


# --------------------------------------------------------------------------- HTML helpers
def esc(value) -> str:
    return html.escape(str(value))


@st.cache_data(show_spinner=False, max_entries=64)
def _png_uri(payload: bytes, max_side: int) -> str:
    with Image.open(io.BytesIO(payload)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def data_uri(source, max_side: int = VIEWER_MAX_SIDE) -> str:
    """PNG data URI for a path, PIL image or RGB array (downscaled for display only)."""
    if isinstance(source, (str, Path)):
        return _png_uri(Path(source).read_bytes(), max_side)
    image = source if isinstance(source, Image.Image) else Image.fromarray(np.asarray(source, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _png_uri(buffer.getvalue(), max_side)


LEGEND = ('<span class="viewer-legend" role="img" aria-label="Heat-map colour scale from low to high attention">'
          '<span>Low</span><i aria-hidden="true"></i><span>High</span></span>')


def ui_viewer(uri: str, tag: str | None = None, alt: str = "MRI slice", size: str = "main", meta: str | None = None, legend: bool = False) -> str:
    """Black image viewer with a label top-left, an optional note bottom-left and colour bar bottom-right."""
    chip = f'<span class="viewer-tag">{esc(tag)}</span>' if tag else ""
    note = f'<span class="viewer-meta">{esc(meta)}</span>' if meta else ""
    return f'<figure class="viewer viewer-{size}">{chip}{note}{LEGEND if legend else ""}<img src="{uri}" alt="{esc(alt)}"></figure>'


def panel_head(title: str, aside: str = "") -> str:
    return f'<div class="panel-head"><h2>{esc(title)}</h2>{f"<span class=panel-aside>{aside}</span>" if aside else ""}</div>'


def ui_kv(rows: list[tuple[str, str]]) -> str:
    return '<dl class="kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def ui_notice(title: str, body: str, tone: str = "info") -> str:
    return f'<div class="notice {tone}" role="{"alert" if tone == "warn" else "note"}"><strong>{esc(title)}</strong> {body}</div>'


def ui_bar(value: float) -> str:
    return f'<span class="bar" aria-hidden="true"><span style="width:{max(0.0, min(value, 1.0)) * 100:.1f}%"></span></span>'


DISCLAIMER = "Research and education only. Not a medical device and not for diagnosis."


def render_footer() -> None:
    from retrieval.gallery2d import SOURCE_CITATION

    st.markdown(
        f'<footer class="app-foot"><p>{esc(DISCLAIMER)} Do not upload identifiable patient data.</p>'
        "<p>Models: ResNet50 tumour detector (99.1% validation accuracy) and ResNet50 four-class type model (96.2%), "
        f"trained on public 2D MRI datasets. Reference data: {esc(SOURCE_CITATION)}.</p></footer>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- application header
gallery, gallery_problem = get_gallery()
demo_cases = gallery.demo_cases() if gallery else []
upload_key = f"upload-{st.session_state.get('uploader_nonce', 0)}"
has_query = st.session_state.get("demo_case") is not None or st.session_state.get(upload_key) is not None

library_status = (
    f'<span class="status ok"><i aria-hidden="true"></i>Reference library: {len(gallery.case_ids)} cases</span>'
    if gallery else '<span class="status warn"><i aria-hidden="true"></i>Reference library unavailable</span>'
)
st.markdown(
    '<header class="app-header">'
    '<div class="app-brand"><span class="app-logo" aria-hidden="true"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v17M7.5 8.5c1.5.9 3 .9 4.5 0'
    'M12 8.5c1.5.9 3 .9 4.5 0M7.5 15.5c1.5-.9 3-.9 4.5 0M12 15.5c1.5-.9 3-.9 4.5 0"/></svg></span>'
    '<div><div class="app-title">Brain MRI Analysis</div><div class="app-sub">Tumour detection, explanation and similar-case search · research use only</div></div></div>'
    f'<div class="app-status"><span class="status ok"><i aria-hidden="true"></i>Models: ResNet50 ×2</span>{library_status}</div>'
    "</header>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- workspace: Scan | Results
scan_column, result_column = st.columns([1.45, 1], gap="medium")
scan_panel = scan_column.container(key="scan-panel")
results_panel = result_column.container(key="results-panel")

with scan_panel:
    st.markdown(panel_head("Scan"), unsafe_allow_html=True)
    if has_query:
        st.markdown('<span class="is-compact" aria-hidden="true"></span>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload an MRI slice (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        key=upload_key,
        on_change=_on_upload,
        label_visibility="collapsed",
    )
    if has_query:
        hint_column, action_column = st.columns([3, 1], vertical_alignment="center")
        if st.session_state.get("demo_case"):
            hint_column.markdown(
                f'<p class="hint">Sample scan <b>{esc(st.session_state["demo_case"])}</b> · held out from the reference library</p>',
                unsafe_allow_html=True,
            )
        else:
            shown = st.session_state.get(upload_key)
            hint_column.markdown(f'<p class="hint"><b>{esc(shown.name if shown else "Your scan")}</b> · drop a new file to replace it</p>', unsafe_allow_html=True)
        action_column.button("Clear", on_click=_clear, key="clear-button", width="stretch")
    toolbar_slot = st.empty()
    image_slot = st.empty()
    caption_slot = st.empty()

# Resolve the active query: an upload wins over a demo scan.
query_bytes, query_name = None, None
if uploaded is not None:
    query_bytes, query_name = uploaded.getvalue(), uploaded.name
elif st.session_state.get("demo_case") and gallery is not None:
    try:
        demo_path = gallery.demo_path(st.session_state["demo_case"])
        query_bytes, query_name = demo_path.read_bytes(), f"Sample scan {st.session_state['demo_case']}"
    except Exception as error:
        with scan_panel:
            st.error(f"The sample scan could not be opened: {error}")

if query_bytes is None:
    st.session_state.pop("result", None)
    with results_panel:
        st.markdown(panel_head("Results"), unsafe_allow_html=True)
        st.markdown(
            '<div class="empty"><p class="empty-title">No scan loaded</p>'
            '<p class="empty-text">Upload one axial MRI slice, or open a sample scan. The prediction, tumour type, heat map '
            "and similar reference cases appear here automatically.</p></div>",
            unsafe_allow_html=True,
        )
        if gallery_problem:
            st.markdown(
                ui_notice("Reference library unavailable.", f"{esc(gallery_problem)} Predictions and heat maps still work.", "warn"),
                unsafe_allow_html=True,
            )
        if demo_cases:
            st.button("Try a demo scan", on_click=_start_demo, args=(demo_cases[0],), key="demo-button", type="primary", width="stretch")
            st.markdown('<p class="label">Sample scans</p><p class="hint">Held out from the reference library, so they are searched like a new upload.</p>',
                        unsafe_allow_html=True)
            with st.container(key="sample-grid"):
                picks = demo_cases[:6]
                for row_start in range(0, len(picks), 3):
                    for column, case_id in zip(st.columns(3), picks[row_start:row_start + 3]):
                        with column:
                            st.markdown(ui_viewer(data_uri(gallery.demo_path(case_id), 256), alt=f"Sample scan {case_id}", size="thumb"), unsafe_allow_html=True)
                            st.button(case_id, key=f"demo-{case_id}", on_click=_start_demo, args=(case_id,), width="stretch")
    render_footer()
    st.stop()

try:
    query_image = read_image(query_bytes)
except ValueError as error:
    st.session_state.pop("result", None)
    with scan_panel:
        st.error(str(error))
    st.stop()

query_id = hashlib.sha256(query_bytes).hexdigest()
result = st.session_state.get("result")
if result is not None and result["id"] != query_id:  # a new scan replaces every earlier result
    result = None
    st.session_state.pop("result", None)
    st.session_state.pop("selected_case", None)
    st.session_state.pop("scan-view", None)

# The scan stays visible while the models run.
image_slot.markdown(ui_viewer(data_uri(query_image), tag="Original", alt="Uploaded MRI slice", meta=f"{query_image.width} × {query_image.height} px"),
                    unsafe_allow_html=True)
caption_slot.markdown(f'<p class="caption"><b>{esc(query_name)}</b> · {query_image.width} × {query_image.height} px</p>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- 1-2. prediction and tumour type
with results_panel:
    head_slot = st.empty()
    head_slot.markdown(panel_head("Results", "Running"), unsafe_allow_html=True)
    if result is None:
        try:
            with st.spinner("Loading models (the first run downloads the weights)"):
                model_set = get_models()
            started = time.perf_counter()
            with st.spinner("Analysing scan"):
                analysis = model_set.analyse(query_image)
            print(f"[brain-mri] inference run for query {query_id[:10]}", flush=True)
        except Exception as error:
            head_slot.markdown(panel_head("Results", "Failed"), unsafe_allow_html=True)
            st.error(f"The scan could not be analysed: {error}")
            st.stop()
        result = {"id": query_id, "analysis": analysis, "search": None, "search_error": None, "searched": False,
                  "timings": {"analyse": time.perf_counter() - started}}
        st.session_state["result"] = result
    analysis = result["analysis"]
    head_slot.markdown(panel_head("Results", f"Analysed in {result['timings']['analyse']:.1f} s"), unsafe_allow_html=True)

    tumour_pct = analysis.tumour_probability * 100
    tone = "positive" if analysis.has_tumour else "negative"
    icon = ('<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true"><circle cx="8" cy="8" r="6.5" fill="none" '
            'stroke="currentColor" stroke-width="1.6"/><circle cx="8" cy="8" r="2.6"/></svg>' if analysis.has_tumour else
            '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">'
            '<circle cx="8" cy="8" r="6.5"/><path d="M5.2 8.2l1.9 1.9 3.8-4" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    st.markdown(
        f'<div class="verdict"><p class="label">Model prediction</p><p class="badge {tone}">{icon}{esc(analysis.label)}</p></div>'
        + ui_kv([
            ("Model confidence", f'<b>{analysis.confidence * 100:.1f}%</b>{ui_bar(analysis.confidence)}'),
            ("Detector tumour score", f'{tumour_pct:.1f}% <span class="muted">· threshold 50%</span>'),
        ]),
        unsafe_allow_html=True,
    )

    type_value = esc(analysis.tumour_type) if analysis.tumour_type else "Not reported"
    type_conf = f'<span class="muted">{analysis.type_confidence * 100:.1f}%</span>' if analysis.type_confidence is not None else ""
    table = ""
    if analysis.has_tumour and analysis.type_probabilities:
        top = max(analysis.type_probabilities, key=analysis.type_probabilities.get)
        table = '<table class="probs" aria-label="Type model probabilities"><tbody>' + "".join(
            f'<tr class="{"top" if name == top else ""}"><th scope="row">{esc(name)}</th><td>{ui_bar(value)}</td><td class="num">{value * 100:.1f}%</td></tr>'
            for name, value in analysis.type_probabilities.items()
        ) + "</tbody></table>"
    st.markdown(
        f'<div class="section"><p class="label">Predicted tumour type</p><p class="type-value">{type_value} {type_conf}</p>{table}</div>',
        unsafe_allow_html=True,
    )
    if analysis.tumour_type:
        from inference import TUMOR_TYPE_INFO

        st.markdown(f'<p class="note">{esc(TUMOR_TYPE_INFO[analysis.tumour_type])}</p>', unsafe_allow_html=True)
    elif analysis.type_note:
        st.markdown(f'<p class="note">{esc(analysis.type_note)}</p>', unsafe_allow_html=True)
    st.markdown('<p class="fine">Model outputs, not a diagnosis.</p>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- 3. Grad-CAM view switch (never reruns the models)
if analysis.overlay is not None:
    with toolbar_slot.container():
        view = st.segmented_control(
            "Scan view", ["Original", "Grad-CAM overlay", "Side by side"], default="Grad-CAM overlay",
            key="scan-view", label_visibility="collapsed",
        ) or "Original"
    if view == "Grad-CAM overlay":
        image_slot.markdown(ui_viewer(data_uri(analysis.overlay), tag="Grad-CAM", alt="Grad-CAM overlay on the uploaded slice",
                                      meta=f"Tumour score {tumour_pct:.1f}%", legend=True), unsafe_allow_html=True)
    elif view == "Side by side":
        image_slot.markdown(
            '<div class="viewer-pair">'
            + ui_viewer(data_uri(query_image), tag="Original", alt="Uploaded MRI slice", size="half")
            + ui_viewer(data_uri(analysis.overlay), tag="Grad-CAM", alt="Grad-CAM overlay", size="half", legend=True)
            + "</div>",
            unsafe_allow_html=True,
        )
    if view != "Original":
        caption_slot.markdown(
            '<p class="caption">Grad-CAM: warmer colours mark the regions that most raised the tumour score. It is not a tumour outline.</p>',
            unsafe_allow_html=True,
        )
else:
    caption_slot.markdown(
        ui_notice("Heat map unavailable.", f"The prediction is fine, but the Grad-CAM overlay failed ({esc(analysis.overlay_error)}).", "warn"),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- 4. reference cases
if gallery is not None and not result["searched"]:
    started = time.perf_counter()
    with st.spinner("Finding reference cases"):
        try:
            if analysis.embedding is None:
                raise RuntimeError(analysis.embedding_error or "no image features were produced for this scan")
            from inference import EMBEDDING_NAME

            if gallery.feature_name != EMBEDDING_NAME:
                raise RuntimeError(
                    "the reference index was built with different image features than this app uses; "
                    "rebuild it with: python -m retrieval.gallery2d index"
                )
            result["search"] = gallery.search(analysis.embedding, query_image, top_k=TOP_K)
        except Exception as error:
            result["search_error"] = str(error)
    result["searched"] = True
    result["timings"]["references"] = time.perf_counter() - started

with st.container(key="refs-panel"):
    aside = f"{len(gallery.case_ids)} reference cases · UPENN-GBM" if gallery else ""
    st.markdown(
        panel_head("Similar cases", esc(aside))
        + '<p class="hint">Ranked by 2D visual similarity (1.00 = identical). It compares appearance only and says nothing about this scan\'s diagnosis.</p>',
        unsafe_allow_html=True,
    )
    if gallery is None:
        st.markdown(ui_notice("Reference library unavailable.", f"{esc(gallery_problem)} The analysis above is unaffected.", "warn"), unsafe_allow_html=True)
    elif result["search_error"]:
        st.markdown(ui_notice("Reference search failed.", f"{esc(result['search_error'])} The analysis above is unaffected.", "warn"), unsafe_allow_html=True)
    elif result["search"] is not None:
        search = result["search"]
        predicted = analysis.tumour_type or ("no tumour" if not analysis.has_tumour else None)
        if analysis.tumour_type == "Glioma":
            st.markdown(ui_notice("Library scope.", f"All {len(gallery.case_ids)} reference cases are glioblastoma from UPENN-GBM."), unsafe_allow_html=True)
        else:
            lead = f"The model predicted {esc(predicted.lower())} for this scan, but the" if predicted else "The"
            st.markdown(
                ui_notice("Library scope.", f"{lead} reference library only contains glioblastoma. These are the nearest-looking "
                          "glioblastoma slices, not examples of the predicted class.", "warn"),
                unsafe_allow_html=True,
            )
        if search.excluded:
            st.markdown(f'<p class="note">This image is identical to reference case {esc(", ".join(search.excluded))}, which was left out of its own results.</p>',
                        unsafe_allow_html=True)
        if not search.matches:
            st.markdown(ui_notice("No reference cases were returned.", ""), unsafe_allow_html=True)

matches = result["search"].matches if result.get("search") and not result["search_error"] else []
if matches:
    case_ids = [match.case_id for match in matches]
    selected_id = st.session_state.get("selected_case")
    if selected_id not in case_ids:
        selected_id = case_ids[0]  # the top result opens by default
    with st.container(key="ref-grid"):
        for column, match in zip(st.columns(len(matches)), matches):
            with column:
                is_selected = match.case_id == selected_id
                st.markdown(
                    f'<div class="ref-card{" selected" if is_selected else ""}">'
                    f'<div class="ref-thumb"><img src="{data_uri(match.image_path, 320)}" alt="Reference case {esc(match.case_id)}"></div>'
                    f'<div class="ref-body"><span class="sr-only">Rank {match.rank}</span>'
                    f'<div class="ref-title"><span class="ref-rank">#{match.rank}</span>{esc(match.details["Documented diagnosis"])}</div>'
                    f'<div class="ref-score"><span>Similarity</span><b>{match.similarity:.2f}</b></div>{ui_bar(match.similarity)}'
                    f'<div class="ref-id">{esc(match.details.get("Source ID", match.case_id))}</div></div></div>',
                    unsafe_allow_html=True,
                )
                st.button("Selected" if is_selected else "Compare", key=f"select-{match.case_id}",
                          type="primary" if is_selected else "secondary", on_click=_select_case, args=(match.case_id,), width="stretch")

    from retrieval.gallery2d import SOURCE_CITATION

    chosen = next(match for match in matches if match.case_id == selected_id)
    with st.container(key="compare-panel"):
        st.markdown(
            panel_head("Comparison", f'2D visual similarity <b class="num">{chosen.similarity:.2f}</b>')
            + f'<p class="hint">Uploaded scan vs reference rank {chosen.rank}</p>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2, gap="medium")
        with left:
            rows = [("Model prediction", esc(analysis.label)), ("Model confidence", f"{analysis.confidence * 100:.1f}%")]
            if analysis.tumour_type:
                rows.append(("Predicted tumour type", f"{esc(analysis.tumour_type)} ({analysis.type_confidence * 100:.1f}%)"))
            st.markdown(
                f'<div class="cmp-head"><p class="label">Uploaded scan · model prediction</p><p class="cmp-title">{esc(analysis.label)}</p></div>'
                + ui_viewer(data_uri(query_image, 720), tag="Uploaded", alt="Uploaded MRI slice", size="cmp")
                + ui_kv(rows)
                + '<p class="fine">Model outputs only. No clinical details are known for this scan.</p>',
                unsafe_allow_html=True,
            )
        with right:
            rows = [(k, esc(v)) for k, v in chosen.details.items()]
            st.markdown(
                f'<div class="cmp-head"><p class="label">Reference case · documented diagnosis</p>'
                f'<p class="cmp-title">{esc(chosen.details["Documented diagnosis"])} · rank {chosen.rank}</p></div>'
                + ui_viewer(data_uri(chosen.image_path, 720), tag="Reference", alt=f"Reference case {chosen.case_id}", size="cmp")
                + ui_kv(rows)
                + f'<p class="fine">{esc(SOURCE_CITATION)}. Missing fields are left out.</p>',
                unsafe_allow_html=True,
            )

render_footer()
if APP_ACCESS_PASSWORD and st.button("Lock app", key="lock-button"):
    st.session_state.pop("app_access_granted", None)
    st.rerun()

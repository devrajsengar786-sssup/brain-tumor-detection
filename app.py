import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import imutils
import h5py
from huggingface_hub import hf_hub_download

# =========================
# HUGGING FACE MODEL REPO
# =========================
# All 6 .h5 files are downloaded from this repo instead of being bundled
# in the GitHub repo (keeps the repo small, and models can be updated by
# re-uploading to HF without redeploying the app).
HF_REPO_ID = "mvdu/brain-tumor-models"  

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# THEME / CUSTOM CSS
# =========================
PRIMARY = "#a78bfa"       # soft violet - main accent
PRIMARY_DARK = "#7c3aed"
BG_DARK = "#0f1117"
CARD_BG = "#1a1d29"
CARD_BORDER = "#2d2f3f"
GREEN = "#4ade80"
RED = "#f87171"
TEXT_MUTED = "#9ca3af"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG_DARK};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG};
        border-right: 1px solid {CARD_BORDER};
    }}

    /* Generic card */
    .app-card {{
        background-color: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}

    .app-card h4 {{
        margin-top: 0;
        color: {PRIMARY};
    }}

    /* Header */
    .app-header {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.2rem;
    }}
    .app-header .brain-icon {{
        font-size: 2.6rem;
        filter: drop-shadow(0 0 12px rgba(167, 139, 250, 0.5));
    }}
    .app-header h1 {{
        margin: 0;
        background: linear-gradient(90deg, {PRIMARY} 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.1rem;
    }}
    .app-subtitle {{
        color: {TEXT_MUTED};
        margin-top: 0;
        margin-bottom: 1.5rem;
    }}

    /* Result cards */
    .result-card {{
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-top: 1rem;
        border: 1px solid;
    }}
    .result-card.tumor {{
        background-color: rgba(248, 113, 113, 0.08);
        border-color: {RED};
    }}
    .result-card.no-tumor {{
        background-color: rgba(74, 222, 128, 0.08);
        border-color: {GREEN};
    }}
    .result-label {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {TEXT_MUTED};
        margin-bottom: 0.3rem;
    }}
    .result-value {{
        font-size: 1.8rem;
        font-weight: 700;
    }}
    .result-value.tumor {{ color: {RED}; }}
    .result-value.no-tumor {{ color: {GREEN}; }}

    /* Sidebar model info card */
    .sidebar-info {{
        background-color: rgba(167, 139, 250, 0.08);
        border: 1px solid {PRIMARY_DARK};
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.8rem;
    }}
    .sidebar-info .metric-row {{
        display: flex;
        justify-content: space-between;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }}
    .sidebar-info .metric-label {{ color: {TEXT_MUTED}; }}
    .sidebar-info .metric-val {{ color: {PRIMARY}; font-weight: 600; }}

    /* Uploader area */
    div[data-testid="stFileUploader"] {{
        border: 2px dashed {CARD_BORDER};
        border-radius: 16px;
        padding: 1rem;
        background-color: {CARD_BG};
    }}

    .disclaimer {{
        color: {TEXT_MUTED};
        font-size: 0.8rem;
        border-top: 1px solid {CARD_BORDER};
        padding-top: 0.8rem;
        margin-top: 2rem;
    }}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL METADATA
# (accuracy values computed directly from each model's confusion matrix on
#  the held-out validation set — from the final training run in
#  brain_tumor_detection.ipynb. val_loss for MultiClass models is read from
#  the final epoch of the training curves; val_loss for Binary models isn't
#  available as no training-curve chart was exported for that section, so
#  it's left as None and hidden in the UI rather than guessed.)
# =========================
MODEL_INFO = {
    "Binary - VGG16": {
        "file": "binary_vgg16.h5",
        "task": "Binary",
        "architecture": "VGG16 (Transfer Learning, frozen ImageNet base)",
        "img_size": (224, 224),
        "accuracy": 97.83,
        "val_loss": 0.08,
        "dataset": "akar49/MRI_Classification (Hugging Face) — notumor/tumor, ~3.6k images, 80/20 split, dedup-checked",
        "chart_type": "history",
        "history_chart": "assets/vgg16_b1.png",
        "confusion_chart": "assets/vgg16_b.png",
    },
    "Binary - Custom CNN": {
        "file": "binary_custom_cnn.h5",
        "task": "Binary",
        "architecture": "Custom 3-block CNN (trained from scratch, class-weighted)",
        "img_size": (224, 224),
        "accuracy": 96.43,
        "val_loss": 0.10,
        "dataset": "akar49/MRI_Classification (Hugging Face) — notumor/tumor, ~3.6k images, 80/20 split, dedup-checked",
        "chart_type": "history",
        "history_chart": "assets/cnn_b1.png",
        "confusion_chart": "assets/cnn_b.png",
    },
    "Binary - ResNet50": {
        "file": "binary_resnet50.h5",
        "task": "Binary",
        "architecture": "ResNet50 (fine-tuned, last 10 layers unfrozen)",
        "img_size": (224, 224),
        "accuracy": 99.07,
        "val_loss": 0.04,
        "dataset": "akar49/MRI_Classification (Hugging Face) — notumor/tumor, ~3.6k images, 80/20 split, dedup-checked",
        "chart_type": "history",
        "history_chart": "assets/resnet50_b1.png",
        "confusion_chart": "assets/resnet50_b.png",
    },
    "MultiClass - Custom CNN": {
        "file": "multiclass_custom_cnn.h5",
        "task": "MultiClass",
        "architecture": "Custom 3-block CNN (trained from scratch, augmented)",
        "img_size": (128, 128),
        "accuracy": 71.04,
        "val_loss": 0.72,
        "dataset": "masoudnickparvar/brain-tumor-mri-dataset (Kaggle) — 4 classes, dynamically balanced, Training+Testing merged",
        "chart_type": "history",
        "history_chart": "assets/cnn_m.png",
        "confusion_chart": "assets/cnn_m1.png",
    },
    "MultiClass - ResNet50": {
        "file": "multiclass_resnet50.h5",
        "task": "MultiClass",
        "architecture": "ResNet50 (fine-tuned, last 10 layers unfrozen)",
        "img_size": (128, 128),
        "accuracy": 96.18,
        "val_loss": 0.15,
        "dataset": "masoudnickparvar/brain-tumor-mri-dataset (Kaggle) — 4 classes, dynamically balanced, Training+Testing merged",
        "chart_type": "history",
        "history_chart": "assets/resnet50_m.png",
        "confusion_chart": "assets/resnet50_m1.png",
    },
    "MultiClass - VGG16": {
        "file": "multiclass_vgg16.h5",
        "task": "MultiClass",
        "architecture": "VGG16 (Transfer Learning, frozen ImageNet base)",
        "img_size": (128, 128),
        "accuracy": 83.19,
        "val_loss": 0.49,
        "dataset": "masoudnickparvar/brain-tumor-mri-dataset (Kaggle) — 4 classes, dynamically balanced, Training+Testing merged",
        "chart_type": "history",
        "history_chart": "assets/vgg16_m.png",
        "confusion_chart": "assets/vgg16_m1.png",
    },
}

binary_classes = ["No Tumor", "Tumor"]
multiclass_names = ["Healthy", "Glioma", "Meningioma", "Pituitary"]
# Used as the automatic second-stage classifier when a binary model predicts Tumor.
# ResNet50 has the strongest recorded multi-class validation result in MODEL_INFO.
DEFAULT_TYPE_MODEL = "MultiClass - ResNet50"

# The multi-class models were trained only to distinguish these broad image
# categories. These descriptions are educational context, not diagnoses.
TUMOR_TYPE_INFO = {
    "Glioma": {
        "summary": "Gliomas are a group of tumours that form in glial cells, which support and protect nerve cells in the central nervous system.",
    },
    "Meningioma": {
        "summary": "Meningiomas form in the meninges, the thin layers of tissue that cover the brain and spinal cord.",
    },
    "Pituitary": {
        "summary": "Pituitary tumours are growths of abnormal cells in the pituitary gland, a hormone-producing gland at the base of the brain.",
    },
}

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def build_binary_resnet50_architecture():
    """
    Rebuilds the exact ResNet50 Binary architecture from the training
    notebook, in native Python code. Used instead of loading the full
    .h5 file for this specific model, because it contains a Lambda layer
    whose function was pickled on a different machine/Python version —
    unpickling that across mismatched Python versions can hard-crash the
    process rather than raise a catchable error. Defining the Lambda's
    function natively here sidesteps that entirely; we only load the
    trained numeric weights from the .h5 file afterwards.

    Layer names below are hardcoded to match the exact names stored
    inside binary_resnet50.h5 (verified with inspect_resnet50.py), so
    load_weights_manually() below matches every layer correctly regardless
    of Keras's auto-naming counters in this session.
    """
    resnet50_base_model = ResNet50(
        weights=None, include_top=False, input_shape=(224, 224, 3)
    )
    return models.Sequential([
        layers.Lambda(
            lambda pixel_values: resnet50_preprocess(pixel_values * 255.0),
            input_shape=(224, 224, 3),
            name="lambda"
        ),
        resnet50_base_model,  # default name is already "resnet50"
        layers.GlobalAveragePooling2D(name="global_average_pooling2d_5"),
        layers.Dense(256, activation='relu', name="dense_10"),
        layers.Dropout(0.4, name="dropout_5"),
        layers.Dense(1, activation='sigmoid', name="dense_11")
    ], name='ResNet50_BrainTumor')


@st.cache_resource
def build_multiclass_resnet50_architecture():
    """
    Same idea as build_binary_resnet50_architecture() above, but for
    multiclass_resnet50.h5. The updated training notebook now fine-tunes
    the last 10 layers of ResNet50 AND adds the same kind of Lambda
    preprocessing layer for the multi-class task (previously it was fully
    frozen with no Lambda layer) — so this model is now exposed to the
    exact same Lambda-unpickling risk described above.

    ⚠️ Layer names below were verified with inspect_h5_model.py against the
    actual trained multiclass_resnet50.h5 — these are the real names, not
    placeholders.
    """
    resnet50_base_model = ResNet50(
        weights=None, include_top=False, input_shape=(128, 128, 3)
    )
    return models.Sequential([
        layers.Lambda(
            lambda pixel_values: resnet50_preprocess(pixel_values * 255.0),
            input_shape=(128, 128, 3),
            name="lambda_1"
        ),
        resnet50_base_model,                                              # default name "resnet50"
        layers.GlobalAveragePooling2D(name="global_average_pooling2d_4"),
        layers.Dense(256, activation='relu', name="dense_8"),
        layers.Dropout(0.4, name="dropout_4"),
        layers.Dense(4, activation='softmax', name="dense_9")
    ], name='MC_ResNet50')


def _find_weights_group(h5_group, target_layer_name):
    """
    Given an h5py group that should contain a layer's weight datasets,
    find the subgroup that actually holds them -- handling the extra
    nesting level some Keras saves add (e.g. dense_10/ResNet50_BrainTumor/
    dense_10/{kernel,bias} instead of just dense_10/{kernel,bias}).
    """
    for key in h5_group.keys():
        if isinstance(h5_group[key], h5py.Dataset):
            return h5_group

    if target_layer_name in h5_group and isinstance(h5_group[target_layer_name], h5py.Group):
        result = _find_weights_group(h5_group[target_layer_name], target_layer_name)
        if result is not None:
            return result

    for key in h5_group.keys():
        sub = h5_group[key]
        if isinstance(sub, h5py.Group):
            result = _find_weights_group(sub, target_layer_name)
            if result is not None:
                return result
    return None


def _set_layer_weights_by_name(layer, h5_layer_group):
    """Load one layer's weights, matching each weight dataset by its exact name."""
    if not layer.weights:
        return
    weights_group = _find_weights_group(h5_layer_group, layer.name)
    if weights_group is None:
        return
    values = []
    for w in layer.weights:
        weight_name = w.name.split("/")[-1].split(":")[0]
        if weight_name not in weights_group:
            raise KeyError(
                f"Weight '{weight_name}' not found for layer '{layer.name}'"
            )
        values.append(weights_group[weight_name][()])
    layer.set_weights(values)


def load_weights_manually(model, h5_path):
    """
    Walks model_weights/<layer_name>/... in the h5 file and assigns weights
    to each layer (including sub-layers of nested Functional models) by
    matching exact weight names.

    We use this instead of Keras's `model.load_weights(path, by_name=True)`
    because that call is unreliable here: the ResNet50 base is nested
    inside a Sequential model, and Keras's legacy by_name loader does not
    correctly recurse into that nested Functional sub-model -- it ends up
    matching weights by position instead of by name for the inner layers,
    silently assigning the wrong weight array to the wrong layer (or, if
    shapes happen to differ, raising a "Shape mismatch" ValueError like:
    "Weight expects shape (7, 7, 3, 64). Received saved weight with shape
    (1, 1, 2048, 512)"). Matching by exact name at every level avoids this
    entirely.
    """
    with h5py.File(h5_path, "r") as f:
        model_weights_root = f["model_weights"]
        for layer in model.layers:
            if layer.name not in model_weights_root:
                continue
            layer_group = model_weights_root[layer.name]
            if hasattr(layer, "layers"):  # nested Functional/Sequential sub-model
                for sublayer in layer.layers:
                    if sublayer.name in layer_group:
                        _set_layer_weights_by_name(sublayer, layer_group[sublayer.name])
            else:
                _set_layer_weights_by_name(layer, layer_group)


@st.cache_resource
def load_model(path):
    """
    `path` is the model's filename as stored in the HF repo (e.g.
    "binary_vgg16.h5"). It's downloaded from Hugging Face Hub first —
    hf_hub_download caches it on disk, so this only re-downloads if the
    file isn't already cached locally.
    """
    try:
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=path)

        if path == "binary_resnet50.h5":
            model = build_binary_resnet50_architecture()
            # See load_weights_manually() docstring for why we don't use
            # model.load_weights(local_path, by_name=True) here.
            load_weights_manually(model, local_path)
            return model

        if path == "multiclass_resnet50.h5":
            model = build_multiclass_resnet50_architecture()
            load_weights_manually(model, local_path)
            return model

        # Keras 3 blocks deserializing Lambda layers with a raw Python
        # function by default (safety guard against untrusted files).
        # Older TF/Keras versions don't have this kwarg at all, so we
        # fall back gracefully if it's not supported.
        try:
            return tf.keras.models.load_model(local_path, safe_mode=False)
        except TypeError:
            return tf.keras.models.load_model(local_path)
    except Exception as e:
        st.error(f"⚠️ Failed to load model '{path}': {e}")
        st.stop()

# =========================
# IMAGE PREPROCESSING
# =========================
def crop_brain_roi(mri_image):
    """
    Removes the black padding around an MRI scan and returns only the
    brain region. Exactly mirrors the preprocessing used to train the
    Binary models in the training notebook — required so inference
    matches training conditions.
    """
    gray_image = cv2.cvtColor(mri_image, cv2.COLOR_RGB2GRAY)
    blurred_gray = cv2.GaussianBlur(gray_image, (5, 5), 0)

    binary_mask = cv2.threshold(blurred_gray, 45, 255, cv2.THRESH_BINARY)[1]
    binary_mask = cv2.erode(binary_mask, None, iterations=2)
    binary_mask = cv2.dilate(binary_mask, None, iterations=2)

    all_contours = cv2.findContours(
        binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    all_contours = imutils.grab_contours(all_contours)

    if len(all_contours) == 0:
        return mri_image  # Nothing to crop — return as-is

    largest_contour = max(all_contours, key=cv2.contourArea)

    leftmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    topmost_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    bottommost_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

    cropped_brain = mri_image[
        topmost_point[1]: bottommost_point[1],
        leftmost_point[0]: rightmost_point[0]
    ]

    # Guard against a degenerate crop (empty slice) — fall back to original
    if cropped_brain.size == 0:
        return mri_image

    return cropped_brain


def preprocess_image(image, target_size, apply_crop=True):
    """
    Mirrors the notebook inference pipeline exactly: PIL RGB -> OpenCV BGR,
    crop the brain ROI, resize, then normalise.  The returned BGR image is the
    same image the model sees and is used as the Grad-CAM overlay background.
    """
    bgr_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if apply_crop:
        bgr_image = crop_brain_roi(bgr_image)

    resized_bgr_image = cv2.resize(bgr_image, target_size, interpolation=cv2.INTER_CUBIC)
    model_input = resized_bgr_image.astype("float32") / 255.0
    return np.expand_dims(model_input, axis=0), resized_bgr_image


def find_last_conv_layer(model):
    """Returns the nested model and its final Conv2D layer for Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return model, layer
        if isinstance(layer, models.Model):
            nested_result = find_last_conv_layer(layer)
            if nested_result is not None:
                return nested_result
    return None


def build_gradcam_model(model):
    """
    Builds a model that exposes the final convolutional activation and the
    original prediction. VGG16 and ResNet50 are nested inside Sequential
    wrappers, so their inner activations need to be connected back through
    the remaining classifier layers explicitly.
    """
    conv_result = find_last_conv_layer(model)
    if conv_result is None:
        raise ValueError("This model has no convolutional layer for Grad-CAM.")

    conv_container, conv_layer = conv_result
    if conv_container is model:
        return models.Model(model.inputs, [conv_layer.output, model.outputs[0]])

    current_tensor = model.inputs[0]
    conv_output = None
    found_container = False

    for layer in model.layers:
        if layer is conv_container:
            nested_gradcam_model = models.Model(
                conv_container.inputs,
                [conv_layer.output, conv_container.outputs[0]]
            )
            conv_output, current_tensor = nested_gradcam_model(current_tensor)
            found_container = True
        else:
            current_tensor = layer(current_tensor)

    if not found_container:
        raise ValueError("Could not connect the model's convolutional layer for Grad-CAM.")

    return models.Model(model.inputs, [conv_output, current_tensor])


def create_gradcam_heatmap(model, model_input, task, predicted_index=None):
    """Produces a normalised Grad-CAM heatmap for the predicted tumour class."""
    gradcam_model = build_gradcam_model(model)

    with tf.GradientTape() as tape:
        conv_output, predictions = gradcam_model(model_input)
        if task == "Binary":
            # Binary models use one sigmoid unit: the positive class is Tumor.
            target_score = predictions[:, 0]
        else:
            target_score = predictions[:, predicted_index]

    gradients = tape.gradient(target_score, conv_output)
    if gradients is None:
        raise ValueError("Grad-CAM gradients could not be calculated for this model.")

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    activation_map = conv_output[0] * channel_weights
    heatmap = tf.reduce_sum(activation_map, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if float(max_value) == 0:
        return np.zeros(heatmap.shape, dtype=np.float32)
    return (heatmap / max_value).numpy()


def overlay_heatmap(image_bgr, heatmap, alpha=0.45):
    """Blends a Grad-CAM heatmap onto the cropped, resized image used by the model."""
    resized_heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return heatmap_bgr, overlay_bgr


def show_tumor_type(
    task,
    predicted_class,
    inferred_tumor_type=None,
    inferred_type_confidence=None,
):
    """Renders safe educational context for the model's image-class result."""
    st.markdown("##### 🧬 Tumour Type")

    if task == "MultiClass":
        tumor_type = predicted_class
        type_confidence = inferred_type_confidence
        type_source = "the selected MultiClass model"
    elif predicted_class != "Tumor":
        st.info(
            "The binary screening model predicted No Tumor, so no tumour type is reported."
        )
        return
    elif inferred_tumor_type is None:
        st.warning(
            "The binary model predicted Tumor, but the follow-up MultiClass model "
            "could not provide a tumour type. Please retry the image or select a "
            "MultiClass model directly."
        )
        return
    else:
        tumor_type = inferred_tumor_type
        type_confidence = inferred_type_confidence
        type_source = f"the automatic {DEFAULT_TYPE_MODEL} follow-up model"

    if tumor_type == "Healthy":
        st.warning(
            "The binary and MultiClass models disagree: the binary model predicted "
            "Tumor while the follow-up model predicted Healthy. No tumour type is "
            "reported for this disagreement."
        )
        return

    type_column, confidence_column = st.columns(2)
    with type_column:
        st.metric("Model-predicted tumour type", tumor_type)
    with confidence_column:
        if type_confidence is not None:
            st.metric("Type confidence", f"{type_confidence * 100:.2f}%")
        else:
            st.metric("Type confidence", "—")
    st.caption(f"Type result produced by {type_source}.")
    st.write(TUMOR_TYPE_INFO[tumor_type]["summary"])

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Select Model")
    selected_model = st.selectbox(
        "Choose Model",
        list(MODEL_INFO.keys()),
        label_visibility="collapsed"
    )
    info = MODEL_INFO[selected_model]
    val_loss_display = f"{info['val_loss']:.2f}" if info['val_loss'] is not None else "—"

    st.markdown(f"""
    <div class="sidebar-info">
        <div class="metric-row"><span class="metric-label">Task</span><span class="metric-val">{info['task']}</span></div>
        <div class="metric-row"><span class="metric-label">Architecture</span><span class="metric-val">{info['architecture'].split('(')[0].strip()}</span></div>
        <div class="metric-row"><span class="metric-label">Accuracy</span><span class="metric-val">{info['accuracy']}%</span></div>
        <div class="metric-row"><span class="metric-label">Val Loss</span><span class="metric-val">{val_loss_display}</span></div>
        <div class="metric-row"><span class="metric-label">Input Size</span><span class="metric-val">{info['img_size'][0]}×{info['img_size'][1]}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.8rem; margin-top:1rem;'>"
        f"Dataset: {info['dataset']}</p>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='disclaimer'>⚠️ This tool is for educational/research purposes only and is not a substitute for professional medical diagnosis.</div>", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="app-header">
    <div class="brain-icon">🧠</div>
    <h1>Brain Tumor Detection</h1>
</div>
<p class="app-subtitle">Upload an MRI scan and choose a model from the sidebar to run the analysis.</p>
""", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab_detect, tab_compare, tab_retrieve = st.tabs(
    ["🔍 Detection", "📊 Model Comparison", "📚 Similar Cases"]
)

# ---------- TAB 1: DETECTION ----------
with tab_detect:
    model = load_model(info["file"])

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Upload MRI Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded MRI", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 🧾 Result")

        if uploaded_file is None:
            st.write("Upload an image first to see the result here.")
        else:
            processed_image, processed_bgr_image = preprocess_image(
                image, info["img_size"], apply_crop=True
            )
            prediction = model.predict(processed_image, verbose=0)

            if info["task"] == "Binary":
                confidence = float(prediction[0][0])
                if confidence > 0.5:
                    predicted_class = binary_classes[1]
                    final_confidence = confidence
                else:
                    predicted_class = binary_classes[0]
                    final_confidence = 1 - confidence
                is_tumor = predicted_class == "Tumor"
            else:
                predicted_index = int(np.argmax(prediction))
                predicted_class = multiclass_names[predicted_index]
                final_confidence = float(np.max(prediction))
                is_tumor = predicted_class != "Healthy"

            # A binary screening model cannot identify the tumour subtype. When it
            # flags Tumor, run the strongest available multi-class model as a second
            # stage so users do not have to upload the image a second time.
            inferred_tumor_type = predicted_class if info["task"] == "MultiClass" else None
            inferred_type_confidence = final_confidence if info["task"] == "MultiClass" else None
            if info["task"] == "Binary" and is_tumor:
                type_model_info = MODEL_INFO[DEFAULT_TYPE_MODEL]
                with st.spinner("Classifying the predicted tumour type..."):
                    type_model = load_model(type_model_info["file"])
                    type_input, _ = preprocess_image(
                        image, type_model_info["img_size"], apply_crop=True
                    )
                    type_prediction = type_model.predict(type_input, verbose=0)
                type_index = int(np.argmax(type_prediction))
                inferred_tumor_type = multiclass_names[type_index]
                inferred_type_confidence = float(np.max(type_prediction))

            css_class = "tumor" if is_tumor else "no-tumor"
            st.markdown(f"""
            <div class="result-card {css_class}">
                <div class="result-label">Result</div>
                <div class="result-value {css_class}">{predicted_class}</div>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Confidence", f"{final_confidence * 100:.2f}%")
            st.progress(min(final_confidence, 1.0))
            show_tumor_type(
                info["task"],
                predicted_class,
                inferred_tumor_type=inferred_tumor_type,
                inferred_type_confidence=inferred_type_confidence,
            )

            if is_tumor:
                st.markdown("##### 🔥 Model Attention Map")
                st.caption(
                    "Grad-CAM highlights image regions that most influenced this "
                    "tumour prediction. It is not a clinical tumour segmentation "
                    "or a confirmed tumour boundary."
                )
                try:
                    heatmap = create_gradcam_heatmap(
                        model,
                        processed_image,
                        info["task"],
                        predicted_index=predicted_index if info["task"] == "MultiClass" else None
                    )
                    heatmap_bgr, overlay_bgr = overlay_heatmap(processed_bgr_image, heatmap)

                    heatmap_col, overlay_col = st.columns(2)
                    with heatmap_col:
                        st.image(
                            cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB),
                            caption="Grad-CAM attention",
                            use_container_width=True
                        )
                    with overlay_col:
                        st.image(
                            cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
                            caption="Attention overlay on model input",
                            use_container_width=True
                        )
                except Exception as error:
                    st.warning(f"The prediction was generated, but its attention map could not be created: {error}")
            else:
                st.info("No tumour class was predicted, so a tumour-specific attention map is not shown.")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------- TAB 2: MODEL COMPARISON ----------
with tab_compare:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 All Models Comparison")

    df = pd.DataFrame([
        {
            "Model": name,
            "Task": d["task"],
            "Architecture": d["architecture"],
            "Accuracy (%)": d["accuracy"],
            "Val Loss": d["val_loss"] if d["val_loss"] is not None else "—",
        }
        for name, d in MODEL_INFO.items()
    ]).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_bin, col_multi = st.columns(2)

    with col_bin:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("##### Binary Models — Accuracy")
        bin_df = df[df["Task"] == "Binary"].set_index("Model")[["Accuracy (%)"]]
        st.bar_chart(bin_df, color=PRIMARY)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_multi:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("##### MultiClass Models — Accuracy")
        multi_df = df[df["Task"] == "MultiClass"].set_index("Model")[["Accuracy (%)"]]
        st.bar_chart(multi_df, color=PRIMARY_DARK)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Training history / confusion matrix charts (from the notebook) ----------
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Training Results")
    st.caption("Accuracy/loss curves and confusion matrix recorded for each model on the validation set — taken directly from the training notebook.")

    hist_task = st.radio(
        "Task",
        ["Binary", "MultiClass"],
        horizontal=True,
        label_visibility="collapsed",
        key="history_task_filter"
    )

    task_models = {name: d for name, d in MODEL_INFO.items() if d["task"] == hist_task}
    hist_model_name = st.selectbox(
        "Model",
        list(task_models.keys()),
        key="history_model_select"
    )
    selected_info = task_models[hist_model_name]

    st.markdown(f"###### {hist_model_name} — Training curves (accuracy & loss)")
    st.image(selected_info["history_chart"], use_container_width=True)

    with st.expander("Show confusion matrix too"):
        st.image(selected_info["confusion_chart"], use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- TAB 3: SIMILAR-CASE RETRIEVAL ----------
with tab_retrieve:
    try:
        from retrieval.quick_ui import render_similarity_page
    except ImportError:
        st.info(
            "The Similar Cases page could not be loaded. Install the application "
            "dependencies with `pip install -r requirements.txt`, then restart Streamlit."
        )
    else:
        render_similarity_page()

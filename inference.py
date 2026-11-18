"""Model loading, preprocessing, Grad-CAM and the single-image analysis pipeline.

This module has no Streamlit dependency so it can be tested and reused by the gallery
builder. The model architectures, manual weight loader, brain-ROI crop, preprocessing
and Grad-CAM maths are the ones the app has always used; they were moved here unchanged.

One upload feeds two things:

* **Inference** - the brain-ROI crop goes to the binary detector (tumour / no tumour,
  plus the Grad-CAM heat map) and, when a tumour is detected, the second-stage
  four-class model supplies the tumour type.
* **Retrieval** - the same crop goes through the second-stage model's ResNet50 backbone;
  its pooled activations are the image embedding compared with the reference slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import h5py
import imutils
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess

# All .h5 files are downloaded from this Hugging Face repo and cached on disk.
HF_REPO_ID = "mvdu/brain-tumor-models"

BINARY_CLASSES = ["No Tumor", "Tumor"]
MULTICLASS_NAMES = ["Healthy", "Glioma", "Meningioma", "Pituitary"]

# Validation accuracy recorded in the training notebook (see README "Results").
# The detector and the type model are the strongest recorded model for each task.
BINARY_MODEL = {"file": "binary_resnet50.h5", "img_size": (224, 224), "name": "ResNet50 binary detector", "val_accuracy": 99.07}
TYPE_MODEL = {"file": "multiclass_resnet50.h5", "img_size": (128, 128), "name": "ResNet50 four-class model", "val_accuracy": 96.18}

# Identifies the retrieval feature; the gallery index must be built with the same one.
EMBEDDING_NAME = f"{TYPE_MODEL['file']}:resnet50-global-average-pool:brain-crop-128"

TUMOR_TYPE_INFO = {
    "Glioma": "Gliomas are a group of tumours that form in glial cells, which support and protect nerve cells in the central nervous system.",
    "Meningioma": "Meningiomas form in the meninges, the thin layers of tissue that cover the brain and spinal cord.",
    "Pituitary": "Pituitary tumours are growths of abnormal cells in the pituitary gland, a hormone-producing gland at the base of the brain.",
}


# --------------------------------------------------------------------------- model loading
def build_binary_resnet50_architecture():
    """
    Rebuilds the exact ResNet50 Binary architecture from the training notebook in
    native Python. The saved .h5 contains a Lambda layer whose function was pickled on
    another machine/Python version; unpickling it can hard-crash the process, so only
    the numeric weights are loaded from the file. Layer names match binary_resnet50.h5.
    """
    resnet50_base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    return models.Sequential([
        layers.Lambda(lambda pixel_values: resnet50_preprocess(pixel_values * 255.0), input_shape=(224, 224, 3), name="lambda"),
        resnet50_base_model,  # default name is already "resnet50"
        layers.GlobalAveragePooling2D(name="global_average_pooling2d_5"),
        layers.Dense(256, activation="relu", name="dense_10"),
        layers.Dropout(0.4, name="dropout_5"),
        layers.Dense(1, activation="sigmoid", name="dense_11"),
    ], name="ResNet50_BrainTumor")


def build_multiclass_resnet50_architecture():
    """Same approach for multiclass_resnet50.h5; layer names match the trained file."""
    resnet50_base_model = ResNet50(weights=None, include_top=False, input_shape=(128, 128, 3))
    return models.Sequential([
        layers.Lambda(lambda pixel_values: resnet50_preprocess(pixel_values * 255.0), input_shape=(128, 128, 3), name="lambda_1"),
        resnet50_base_model,
        layers.GlobalAveragePooling2D(name="global_average_pooling2d_4"),
        layers.Dense(256, activation="relu", name="dense_8"),
        layers.Dropout(0.4, name="dropout_4"),
        layers.Dense(4, activation="softmax", name="dense_9"),
    ], name="MC_ResNet50")


def _find_weights_group(h5_group, target_layer_name):
    """Find the subgroup holding a layer's weight datasets (handles extra Keras nesting)."""
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
            raise KeyError(f"Weight '{weight_name}' not found for layer '{layer.name}'")
        values.append(weights_group[weight_name][()])
    layer.set_weights(values)


def load_weights_manually(model, h5_path):
    """
    Assign weights layer by layer, matching exact names (including the nested ResNet50).
    Keras's ``load_weights(by_name=True)`` does not recurse into the nested Functional
    sub-model correctly and can silently assign the wrong arrays.
    """
    def numeric_suffix(name: str) -> int:
        tail = name.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    with h5py.File(h5_path, "r") as f:
        model_weights_root = f["model_weights"]
        # The classifier-head layer names depend on Keras's counters at training time and
        # change whenever a model is retrained and re-uploaded (binary_resnet50.h5 stores
        # dense_4/dense_5, not the dense_10/dense_11 this code used to hardcode, which left
        # the head randomly initialised). Head Dense layers are therefore matched in order.
        saved_dense = sorted((k for k in model_weights_root if k.startswith("dense")), key=numeric_suffix)
        model_dense = [layer for layer in model.layers if isinstance(layer, layers.Dense)]
        dense_group = dict(zip([layer.name for layer in model_dense], saved_dense)) if len(saved_dense) == len(model_dense) else {}

        unloaded = []
        for layer in model.layers:
            group_name = layer.name if layer.name in model_weights_root else dense_group.get(layer.name)
            if group_name is None:
                if layer.weights:
                    unloaded.append(layer.name)
                continue
            layer_group = model_weights_root[group_name]
            if hasattr(layer, "layers"):
                for sublayer in layer.layers:
                    if sublayer.name in layer_group:
                        _set_layer_weights_by_name(sublayer, layer_group[sublayer.name])
                    elif sublayer.weights:
                        unloaded.append(f"{layer.name}/{sublayer.name}")
            elif layer.weights:
                weights_group = _find_weights_group(layer_group, group_name)
                if weights_group is None:
                    unloaded.append(layer.name)
                    continue
                layer.set_weights([weights_group[w.name.split("/")[-1].split(":")[0]][()] for w in layer.weights])
        if unloaded:
            raise ValueError(
                f"{len(unloaded)} layer(s) of '{model.name}' have no weights in the file "
                f"(for example {unloaded[:3]}); refusing to run a partly initialised model."
            )


def load_model(filename: str):
    """Download (or reuse the cached) weights file and return the ready Keras model."""
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    if filename == "binary_resnet50.h5":
        model = build_binary_resnet50_architecture()
        load_weights_manually(model, local_path)
        return model
    if filename == "multiclass_resnet50.h5":
        model = build_multiclass_resnet50_architecture()
        load_weights_manually(model, local_path)
        return model
    try:
        return tf.keras.models.load_model(local_path, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(local_path)


# --------------------------------------------------------------------------- preprocessing
def brain_roi_box(bgr_image) -> tuple[int, int, int, int] | None:
    """Bounding box (top, bottom, left, right) of the brain, or None when nothing to crop."""
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2GRAY)
    blurred_gray = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary_mask = cv2.threshold(blurred_gray, 45, 255, cv2.THRESH_BINARY)[1]
    binary_mask = cv2.erode(binary_mask, None, iterations=2)
    binary_mask = cv2.dilate(binary_mask, None, iterations=2)
    all_contours = imutils.grab_contours(cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    if len(all_contours) == 0:
        return None
    largest_contour = max(all_contours, key=cv2.contourArea)
    left = int(largest_contour[:, :, 0].min())
    right = int(largest_contour[:, :, 0].max())
    top = int(largest_contour[:, :, 1].min())
    bottom = int(largest_contour[:, :, 1].max())
    if bottom <= top or right <= left:
        return None
    return top, bottom, left, right


def crop_brain_roi(mri_image):
    """Remove the black padding around a scan; mirrors the training notebook's crop."""
    box = brain_roi_box(mri_image)
    if box is None:
        return mri_image
    top, bottom, left, right = box
    return mri_image[top:bottom, left:right]


def preprocess_image(image: Image.Image, target_size, apply_crop=True):
    """PIL RGB -> OpenCV BGR, crop the brain ROI, resize, scale to 0-1 (as in training)."""
    bgr_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if apply_crop:
        bgr_image = crop_brain_roi(bgr_image)
    resized_bgr_image = cv2.resize(bgr_image, target_size, interpolation=cv2.INTER_CUBIC)
    model_input = resized_bgr_image.astype("float32") / 255.0
    return np.expand_dims(model_input, axis=0), resized_bgr_image


# --------------------------------------------------------------------------- Grad-CAM
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
    """Expose the final convolutional activation alongside the prediction."""
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
            nested_gradcam_model = models.Model(conv_container.inputs, [conv_layer.output, conv_container.outputs[0]])
            conv_output, current_tensor = nested_gradcam_model(current_tensor)
            found_container = True
        else:
            current_tensor = layer(current_tensor)
    if not found_container:
        raise ValueError("Could not connect the model's convolutional layer for Grad-CAM.")
    return models.Model(model.inputs, [conv_output, current_tensor])


def create_gradcam_heatmap(gradcam_model, model_input, task="Binary", predicted_index=None):
    """Normalised Grad-CAM heat map for the tumour score (binary) or a class index."""
    with tf.GradientTape() as tape:
        conv_output, predictions = gradcam_model(model_input)
        target_score = predictions[:, 0] if task == "Binary" else predictions[:, predicted_index]
    gradients = tape.gradient(target_score, conv_output)
    if gradients is None:
        raise ValueError("Grad-CAM gradients could not be calculated for this model.")
    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.maximum(tf.reduce_sum(conv_output[0] * channel_weights, axis=-1), 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) == 0:
        return np.zeros(heatmap.shape, dtype=np.float32)
    return (heatmap / max_value).numpy()


def overlay_on_original(image: Image.Image, heatmap: np.ndarray, max_alpha: float = 0.6) -> np.ndarray:
    """
    Paint the heat map over the *original* image so both views share one size.

    The heat map covers the brain-ROI crop the model saw, so it is resized to that box.
    Opacity follows the heat-map strength, leaving low-attention areas of the scan untouched.
    """
    rgb = np.array(image.convert("RGB"))
    box = brain_roi_box(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) or (0, rgb.shape[0], 0, rgb.shape[1])
    top, bottom, left, right = box
    full = np.zeros(rgb.shape[:2], dtype=np.float32)
    full[top:bottom, left:right] = cv2.resize(heatmap.astype(np.float32), (right - left, bottom - top), interpolation=cv2.INTER_CUBIC)
    full = np.clip(full, 0, 1)
    colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * full), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    alpha = (full * max_alpha)[..., None]
    return np.uint8(rgb * (1 - alpha) + colour * alpha)


# --------------------------------------------------------------------------- analysis
@dataclass
class Analysis:
    has_tumour: bool
    label: str  # "Tumour detected" / "No tumour detected"
    confidence: float  # model confidence in the predicted label
    tumour_probability: float
    tumour_type: str | None = None  # second-stage type, when a tumour is detected and the models agree
    type_confidence: float | None = None
    type_note: str | None = None  # why no type is reported
    type_probabilities: dict[str, float] | None = None  # all four classes of the second-stage model
    overlay: np.ndarray | None = None  # RGB Grad-CAM overlay at the original resolution
    overlay_error: str | None = None
    embedding: np.ndarray | None = None
    embedding_error: str | None = None


@dataclass
class ModelSet:
    binary_model: object
    type_model: object
    _gradcam_model: object | None = None

    @classmethod
    def load(cls, loader: Callable[[str], object] = load_model) -> "ModelSet":
        return cls(loader(BINARY_MODEL["file"]), loader(TYPE_MODEL["file"]))

    # The type model is Sequential[Lambda, ResNet50, GlobalAveragePooling, Dense, Dropout, Dense].
    def _type_forward(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        model_input, _ = preprocess_image(image, TYPE_MODEL["img_size"])
        tensor = tf.convert_to_tensor(model_input)
        pooled = None
        for position, layer in enumerate(self.type_model.layers):
            tensor = layer(tensor, training=False)
            if position == 2:
                pooled = tensor
        return pooled.numpy()[0], tensor.numpy()[0]

    def embed(self, image: Image.Image) -> np.ndarray:
        """Image embedding used for retrieval (pooled ResNet50 activations of the brain crop)."""
        return self._type_forward(image)[0]

    def analyse(self, image: Image.Image) -> Analysis:
        binary_input, _ = preprocess_image(image, BINARY_MODEL["img_size"])
        tumour_probability = float(self.binary_model.predict(binary_input, verbose=0)[0][0])
        has_tumour = tumour_probability > 0.5
        result = Analysis(
            has_tumour=has_tumour,
            label="Tumour detected" if has_tumour else "No tumour detected",
            confidence=tumour_probability if has_tumour else 1 - tumour_probability,
            tumour_probability=tumour_probability,
        )
        try:
            embedding, type_probabilities = self._type_forward(image)
            result.embedding = embedding
            result.type_probabilities = {name: float(value) for name, value in zip(MULTICLASS_NAMES, type_probabilities)}
        except Exception as error:
            type_probabilities = None
            result.embedding_error = str(error)
        if not has_tumour:
            result.type_note = "No tumour was detected, so no tumour type is reported."
        elif type_probabilities is None:
            result.type_note = "The tumour-type model could not be run for this image."
        else:
            type_index = int(np.argmax(type_probabilities))
            if MULTICLASS_NAMES[type_index] == "Healthy":
                result.type_note = (
                    "The two models disagree: the detector found a tumour but the type model "
                    "classified the scan as healthy. No tumour type is reported."
                )
            else:
                result.tumour_type = MULTICLASS_NAMES[type_index]
                result.type_confidence = float(type_probabilities[type_index])
        try:
            if self._gradcam_model is None:
                self._gradcam_model = build_gradcam_model(self.binary_model)
            heatmap = create_gradcam_heatmap(self._gradcam_model, binary_input, "Binary")
            result.overlay = overlay_on_original(image, heatmap)
        except Exception as error:
            result.overlay_error = str(error)
        return result

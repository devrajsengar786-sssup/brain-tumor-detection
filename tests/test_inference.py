"""Inference pipeline tests with tiny stand-in Keras models (no downloaded weights)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models

import inference
from inference import ModelSet, brain_roi_box, crop_brain_roi, load_weights_manually, overlay_on_original, preprocess_image


def scan(size=(200, 160), offset=0) -> Image.Image:
    """A bright ellipse ("brain") with a brighter blob on a black background."""
    yy, xx = np.mgrid[: size[1], : size[0]]
    brain = ((xx - size[0] / 2) / (size[0] * 0.33)) ** 2 + ((yy - size[1] / 2) / (size[1] * 0.38)) ** 2 <= 1
    blob = (xx - size[0] * 0.6 - offset) ** 2 + (yy - size[1] * 0.45) ** 2 <= 14 ** 2
    pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
    pixels[brain] = 110
    pixels[blob & brain] = 230
    return Image.fromarray(pixels, mode="L").convert("RGB")


def tiny_binary(bias: float):
    model = models.Sequential([
        layers.Input((224, 224, 3)),
        layers.Conv2D(4, 3, strides=8, activation="relu", name="conv_a"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation="sigmoid", name="out"),
    ])
    kernel, _ = model.get_layer("out").get_weights()
    model.get_layer("out").set_weights([np.abs(kernel), np.array([bias], dtype=np.float32)])
    return model


def tiny_type(winner: int):
    backbone = models.Sequential([layers.Input((128, 128, 3)), layers.Conv2D(6, 3, strides=8, activation="relu")], name="resnet50")
    model = models.Sequential([
        layers.Input((128, 128, 3)),
        layers.Lambda(lambda x: x * 1.0, name="lambda_1"),
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dense(5, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(4, activation="softmax", name="types"),
    ])
    kernel, _ = model.get_layer("types").get_weights()
    bias = np.full(4, -20.0, dtype=np.float32)
    bias[winner] = 20.0
    model.get_layer("types").set_weights([kernel * 0, bias])
    return model


class PreprocessingTests(unittest.TestCase):
    def test_crop_matches_box_and_removes_padding(self):
        bgr = np.array(scan())[:, :, ::-1]
        top, bottom, left, right = brain_roi_box(bgr)
        cropped = crop_brain_roi(bgr)
        self.assertEqual(cropped.shape[:2], (bottom - top, right - left))
        self.assertLess(cropped.shape[0], bgr.shape[0])
        self.assertLess(cropped.shape[1], bgr.shape[1])

    def test_blank_image_is_not_cropped(self):
        blank = np.zeros((50, 60, 3), dtype=np.uint8)
        self.assertIsNone(brain_roi_box(blank))
        self.assertEqual(crop_brain_roi(blank).shape, blank.shape)

    def test_preprocess_shapes_and_range(self):
        model_input, resized = preprocess_image(scan(), (128, 128))
        self.assertEqual(model_input.shape, (1, 128, 128, 3))
        self.assertEqual(resized.shape, (128, 128, 3))
        self.assertTrue(0.0 <= model_input.min() and model_input.max() <= 1.0)

    def test_overlay_keeps_original_size_and_untouched_background(self):
        image = scan((200, 160))
        heat = np.zeros((7, 7), dtype=np.float32)
        heat[3, 4] = 1.0
        overlay = overlay_on_original(image, heat)
        original = np.array(image)
        self.assertEqual(overlay.shape, original.shape)
        top, bottom, left, right = brain_roi_box(original[:, :, ::-1])
        outside = np.ones(original.shape[:2], dtype=bool)
        outside[top:bottom, left:right] = False
        np.testing.assert_array_equal(overlay[outside], original[outside])
        self.assertTrue((overlay != original).any())
        np.testing.assert_array_equal(overlay_on_original(image, np.zeros((7, 7), np.float32)), original)


class WeightLoaderTests(unittest.TestCase):
    """Regression: head Dense layers saved under different names were silently left random."""

    def _model(self):
        return models.Sequential([
            layers.Input((4,)),
            layers.Dense(3, name="dense_10"),
            layers.Dense(1, name="dense_11"),
        ], name="head_only")

    def _write(self, path: Path, names: list[str]):
        shapes = {0: ((4, 3), (3,)), 1: ((3, 1), (1,))}
        with h5py.File(path, "w") as f:
            root = f.create_group("model_weights")
            for position, name in enumerate(names):
                group = root.create_group(name).create_group(name)
                group.create_dataset("kernel", data=np.full(shapes[position][0], position + 1.0, dtype=np.float32))
                group.create_dataset("bias", data=np.full(shapes[position][1], position + 0.5, dtype=np.float32))

    def test_renamed_head_layers_are_loaded_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.h5"
            self._write(path, ["dense_4", "dense_5"])
            model = self._model()
            load_weights_manually(model, path)
        np.testing.assert_array_equal(model.get_layer("dense_10").get_weights()[0], np.full((4, 3), 1.0))
        np.testing.assert_array_equal(model.get_layer("dense_11").get_weights()[1], np.array([1.5]))

    def test_missing_layer_weights_raise_instead_of_running_untrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.h5"
            self._write(path, ["dense_4"])
            with self.assertRaisesRegex(ValueError, "partly initialised"):
                load_weights_manually(self._model(), path)


class AnalysisTests(unittest.TestCase):
    def test_tumour_path_reports_type_overlay_and_embedding(self):
        analysis = ModelSet(tiny_binary(8.0), tiny_type(1)).analyse(scan())
        self.assertTrue(analysis.has_tumour)
        self.assertEqual(analysis.label, "Tumour detected")
        self.assertGreater(analysis.confidence, 0.5)
        self.assertEqual(analysis.tumour_type, "Glioma")
        self.assertAlmostEqual(analysis.type_confidence, 1.0, places=4)
        self.assertEqual(analysis.overlay.shape, (160, 200, 3))
        self.assertIsNone(analysis.overlay_error)
        self.assertEqual(analysis.embedding.shape, (6,))

    def test_no_tumour_path_reports_no_type_but_still_embeds(self):
        analysis = ModelSet(tiny_binary(-8.0), tiny_type(2)).analyse(scan())
        self.assertFalse(analysis.has_tumour)
        self.assertEqual(analysis.label, "No tumour detected")
        self.assertGreater(analysis.confidence, 0.5)  # confidence in the predicted label
        self.assertIsNone(analysis.tumour_type)
        self.assertIn("No tumour", analysis.type_note)
        self.assertIsNotNone(analysis.embedding)

    def test_model_disagreement_reports_no_type(self):
        analysis = ModelSet(tiny_binary(8.0), tiny_type(0)).analyse(scan())
        self.assertTrue(analysis.has_tumour)
        self.assertIsNone(analysis.tumour_type)
        self.assertIn("disagree", analysis.type_note)

    def test_heat_map_failure_keeps_the_prediction(self):
        class NoConv:
            layers = []

            def predict(self, *_args, **_kwargs):
                return np.array([[0.9]])

        analysis = ModelSet(NoConv(), tiny_type(1)).analyse(scan())
        self.assertTrue(analysis.has_tumour)
        self.assertIsNone(analysis.overlay)
        self.assertIn("convolutional", analysis.overlay_error)
        self.assertEqual(analysis.tumour_type, "Glioma")

    def test_embedding_depends_only_on_the_image(self):
        models_ = ModelSet(tiny_binary(8.0), tiny_type(1))
        np.testing.assert_allclose(models_.embed(scan()), models_.embed(scan()))
        self.assertFalse(np.allclose(models_.embed(scan()), models_.embed(scan(offset=-40))))

    def test_defaults_are_the_best_recorded_models(self):
        self.assertEqual(inference.BINARY_MODEL["file"], "binary_resnet50.h5")
        self.assertEqual(inference.TYPE_MODEL["file"], "multiclass_resnet50.h5")
        self.assertEqual(inference.BINARY_MODEL["img_size"], (224, 224))
        self.assertEqual(inference.TYPE_MODEL["img_size"], (128, 128))
        self.assertEqual(inference.MULTICLASS_NAMES, ["Healthy", "Glioma", "Meningioma", "Pituitary"])


if __name__ == "__main__":
    unittest.main()

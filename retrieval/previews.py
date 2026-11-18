"""Small PNG previews: the axial slice with the largest tumour area, with subregion outlines."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage

from .volumes import LoadedCase


# Outline colours per subregion label (1 necrotic core, 2 oedema, 3/4 enhancing).
LABEL_COLOURS = {1: (255, 214, 10), 2: (64, 196, 255), 3: (255, 64, 96), 4: (255, 64, 96)}
PANEL_CHANNELS = ((1, "T1-Gd"), (3, "FLAIR"))


def _window(slice_: np.ndarray, brain: np.ndarray) -> np.ndarray:
    values = slice_[brain] if brain.any() else slice_.ravel()
    low, high = np.percentile(values, (1, 99)) if values.size else (0.0, 1.0)
    return np.clip((slice_ - low) / max(high - low, 1e-6), 0, 1)


def tumour_slice_index(case: LoadedCase) -> int:
    return int(np.argmax(case.mask.sum(axis=(0, 1))))


def preview_array(case: LoadedCase, size: int = 192) -> np.ndarray:
    """Side-by-side T1-Gd and FLAIR panels (RGB uint8) at the largest-tumour axial slice."""
    z = tumour_slice_index(case)
    labels = case.labels if case.labels is not None else case.mask.astype(np.int16)
    brain = case.brain_mask[:, :, z] if case.brain_mask is not None else np.ones(case.mask.shape[:2], bool)
    panels = []
    for channel, _ in PANEL_CHANNELS:
        grey = (_window(case.images[:, :, z, channel], brain) * 255).astype(np.uint8)
        rgb = np.repeat(grey[..., None], 3, axis=-1)
        label_slice = labels[:, :, z]
        for value, colour in LABEL_COLOURS.items():
            region = label_slice == value
            if region.any():
                rgb[region & ~ndimage.binary_erosion(region)] = colour
        # Array axes are (x, y); transpose to rows=y. Put anterior at the top: flip only
        # when increasing y index points anterior (UPENN's affine has y decreasing).
        panel = np.transpose(rgb, (1, 0, 2))
        panels.append(np.flipud(panel) if case.affine[1, 1] > 0 else panel)
    image = Image.fromarray(np.concatenate(panels, axis=1))
    scale = size / image.height
    return np.asarray(image.resize((int(image.width * scale), size), Image.Resampling.BILINEAR))


def preview_png(case: LoadedCase, size: int = 192) -> bytes:
    buffer = BytesIO()
    Image.fromarray(preview_array(case, size)).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()

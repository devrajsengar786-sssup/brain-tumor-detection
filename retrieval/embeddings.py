"""Compact 3D encoder utilities for whole-MRI and tumour-region representations."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from .volumes import LoadedCase


def resize_case_for_encoder(
    case: LoadedCase,
    output_shape: tuple[int, int, int] = (64, 64, 64),
    tumor_only: bool = False,
) -> np.ndarray:
    """Prepare a normalized 4-channel 3D tensor for a trained encoder."""
    volume = case.images * case.mask[..., np.newaxis] if tumor_only else case.images
    factors = tuple(target / source for target, source in zip(output_shape, volume.shape[:3])) + (1.0,)
    resized = zoom(volume, factors, order=1)
    return resized.astype(np.float32)


def build_3d_autoencoder(
    input_shape: tuple[int, int, int, int] = (64, 64, 64, 4),
    embedding_dimension: int = 128,
):
    """
    Build a small trainable 3D autoencoder and its reusable encoder component.

    Train separate whole-image and tumour-masked encoders on the training split only.
    The resulting embeddings form the generic-deep and tumour-region baselines used
    in the retrieval ablation study.
    """
    import tensorflow as tf

    encoder_input = tf.keras.Input(shape=input_shape, name="mri_volume")
    tensor = encoder_input
    for filters in (16, 32, 64):
        tensor = tf.keras.layers.Conv3D(filters, 3, strides=2, padding="same", activation="relu")(tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
    tensor = tf.keras.layers.GlobalAveragePooling3D()(tensor)
    embedding = tf.keras.layers.Dense(embedding_dimension, name="embedding")(tensor)
    encoder = tf.keras.Model(encoder_input, embedding, name="mri_3d_encoder")

    decoder_input = tf.keras.Input(shape=(embedding_dimension,), name="embedding_input")
    base_shape = tuple(dimension // 8 for dimension in input_shape[:3])
    tensor = tf.keras.layers.Dense(int(np.prod(base_shape) * 64), activation="relu")(decoder_input)
    tensor = tf.keras.layers.Reshape((*base_shape, 64))(tensor)
    for filters in (64, 32, 16):
        tensor = tf.keras.layers.Conv3DTranspose(filters, 3, strides=2, padding="same", activation="relu")(tensor)
    reconstruction = tf.keras.layers.Conv3D(input_shape[-1], 1, activation="linear", name="reconstruction")(tensor)
    decoder = tf.keras.Model(decoder_input, reconstruction, name="mri_3d_decoder")

    autoencoder = tf.keras.Model(encoder_input, decoder(encoder(encoder_input)), name="mri_3d_autoencoder")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
    return autoencoder, encoder


def encode_cases(model, cases: list[LoadedCase], tumor_only: bool = False) -> np.ndarray:
    """Generate embeddings without retraining or fitting on query/test cases."""
    if not cases:
        raise ValueError("At least one case is required to create embeddings.")
    tensors = np.stack([resize_case_for_encoder(case, tumor_only=tumor_only) for case in cases])
    return np.asarray(model.predict(tensors, verbose=0), dtype=np.float32)

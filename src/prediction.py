"""
prediction.py
-------------
Inference helpers for the Chest X-Ray Pneumonia Classifier.

Provides high-level wrappers around a loaded Keras model so that callers
only need to supply a model and an image source (bytes, path, or directory).

Typical usage
-------------
>>> from src.prediction import predict_from_path
>>> result = predict_from_path(model, "chest_xray.jpg")
>>> print(result)
{'class': 'PNEUMONIA', 'confidence': 0.9341, 'probabilities': {'NORMAL': 0.0659, 'PNEUMONIA': 0.9341}}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing import load_and_preprocess_image, preprocess_single_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = ["NORMAL", "PNEUMONIA"]
"""Ordered list of class labels.  Index 0 → NORMAL, index 1 → PNEUMONIA."""

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_result(predictions: np.ndarray) -> dict[str, Any]:
    """Convert a raw model output array into a structured prediction dict.

    Parameters
    ----------
    predictions:
        2-D float array of shape ``(1, num_classes)`` as returned by
        ``model.predict()``.

    Returns
    -------
    dict
        ``{"class": str, "confidence": float, "probabilities": {str: float}}``
    """
    probs = predictions[0]  # shape: (num_classes,)
    class_index = int(np.argmax(probs))
    confidence = float(probs[class_index])
    probability_map = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

    return {
        "class": CLASS_NAMES[class_index],
        "confidence": confidence,
        "probabilities": probability_map,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_image(model: Any, image_array: np.ndarray) -> dict[str, Any]:
    """Run inference on a pre-processed image array.

    Parameters
    ----------
    model:
        A loaded Keras model (e.g. from :func:`src.model.load_model`).
        Must accept input of shape ``(batch, 224, 224, 3)``.
    image_array:
        Float32 numpy array of shape ``(1, 224, 224, 3)`` with pixel
        values normalised to ``[0, 1]``.

    Returns
    -------
    dict
        ``{"class": str, "confidence": float, "probabilities": {str: float}}``

        - **class** – the predicted label (``"NORMAL"`` or ``"PNEUMONIA"``).
        - **confidence** – the probability assigned to the predicted class
          (between 0 and 1).
        - **probabilities** – per-class softmax probabilities.

    Raises
    ------
    ValueError
        If *image_array* does not have shape ``(1, H, W, 3)``.
    """
    if image_array.ndim != 4 or image_array.shape[0] != 1:
        raise ValueError(
            f"image_array must have shape (1, H, W, 3), got {image_array.shape}"
        )

    predictions = model.predict(image_array, verbose=0)
    return _build_result(predictions)


def predict_from_path(model: Any, image_path: str | Path) -> dict[str, Any]:
    """Load an image from *image_path*, preprocess it, and run inference.

    This is a convenience wrapper around :func:`load_and_preprocess_image`
    and :func:`predict_image` for single-file predictions.

    Parameters
    ----------
    model:
        A loaded Keras model.
    image_path:
        Filesystem path to the image (JPEG or PNG).

    Returns
    -------
    dict
        ``{"class": str, "confidence": float, "probabilities": {str: float}}``
        (same structure as :func:`predict_image`).

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    ValueError
        If the image cannot be opened or decoded.
    """
    image_path = Path(image_path)
    # load_and_preprocess_image returns (H, W, 3); add batch dim
    arr = load_and_preprocess_image(image_path)        # (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)                  # (1, 224, 224, 3)
    return predict_image(model, arr)


def batch_predict(model: Any, image_dir: str | Path) -> list[dict[str, Any]]:
    """Run inference on every image found in *image_dir*.

    Traverses *image_dir* (non-recursively) and predicts each file whose
    extension is ``.jpg``, ``.jpeg``, or ``.png``.  Results are returned
    in filesystem iteration order (which may vary across OS / filesystems).

    Parameters
    ----------
    model:
        A loaded Keras model.
    image_dir:
        Path to a directory containing image files.

    Returns
    -------
    list[dict]
        One dict per image, each with:

        - **file** (``str``) – basename of the image file.
        - **path** (``str``) – absolute path to the image file.
        - **class** (``str``) – predicted label.
        - **confidence** (``float``) – confidence in the prediction.
        - **probabilities** (``dict[str, float]``) – per-class probabilities.
        - **error** (``str | None``) – ``None`` on success; error message on failure.

    Raises
    ------
    NotADirectoryError
        If *image_dir* is not a directory.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    results: list[dict[str, Any]] = []

    for entry in sorted(image_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        base_result: dict[str, Any] = {
            "file": entry.name,
            "path": str(entry.resolve()),
            "class": None,
            "confidence": None,
            "probabilities": None,
            "error": None,
        }

        try:
            prediction = predict_from_path(model, entry)
            base_result.update(prediction)
        except Exception as exc:  # pragma: no cover
            base_result["error"] = str(exc)

        results.append(base_result)

    return results

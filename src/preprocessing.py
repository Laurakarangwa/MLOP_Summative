"""
preprocessing.py
----------------
Image preprocessing utilities for the Chest X-Ray Pneumonia Classifier.

Provides functions for loading, normalising, and augmenting images for
both training pipelines and single-image inference.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional TensorFlow / Keras imports – gracefully handled so the module
# can be imported in environments where TF is not yet installed.
# ---------------------------------------------------------------------------
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False
    ImageDataGenerator = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TARGET_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_BATCH_SIZE: int = 32


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess_image(
    path: str | Path,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """Load an image from *path*, resize it, and normalise pixel values to [0, 1].

    Parameters
    ----------
    path:
        Filesystem path to the image file (JPEG, PNG, etc.).
    target_size:
        ``(height, width)`` tuple that the image will be resized to.
        Defaults to ``(224, 224)``.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(height, width, 3)`` with values in ``[0, 1]``.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    ValueError
        If the image cannot be opened or converted to RGB.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Cannot open image at {path}: {exc}") from exc

    img = img.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # shape: (height, width, 3)


def create_data_generators(
    train_dir: str | Path,
    test_dir: str | Path,
    img_size: int = 224,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple:
    """Create Keras ``ImageDataGenerator`` flows for training and evaluation.

    The **training** generator applies the following augmentations:
    - Horizontal flip
    - Rotation up to ±15 °
    - Zoom up to ±10 %
    - Brightness jitter in [0.8, 1.2]
    - Pixel rescaling to [0, 1]

    The **test/validation** generator only rescales pixels to [0, 1] –
    no augmentation is applied to avoid data leakage.

    Both generators produce batches of shape
    ``(batch_size, img_size, img_size, 3)`` with *categorical* labels
    (class mode = ``"categorical"``).

    Parameters
    ----------
    train_dir:
        Root directory for training images.  Must contain one subdirectory
        per class (e.g. ``NORMAL/`` and ``PNEUMONIA/``).
    test_dir:
        Root directory for test / validation images.  Same sub-directory
        structure expected.
    img_size:
        Side length in pixels.  Images are resized to ``(img_size, img_size)``.
    batch_size:
        Number of images per batch.

    Returns
    -------
    (train_generator, test_generator)
        A tuple of two Keras ``DirectoryIterator`` objects ready for use
        with ``model.fit()`` and ``model.evaluate()``.

    Raises
    ------
    RuntimeError
        If TensorFlow / Keras is not installed.
    """
    if not _TF_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required for create_data_generators. "
            "Install it with: pip install tensorflow"
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    target_size = (img_size, img_size)

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, test_generator


def get_class_distribution(data_dir: str | Path) -> dict[str, int]:
    """Count the number of images for each class in *data_dir*.

    Expects *data_dir* to contain subdirectories named ``NORMAL`` and
    ``PNEUMONIA``.  Any file with a recognised image extension
    (``.jpg``, ``.jpeg``, ``.png``) inside a class subdirectory is counted.

    Parameters
    ----------
    data_dir:
        Root directory containing ``NORMAL/`` and ``PNEUMONIA/`` subdirs.

    Returns
    -------
    dict
        ``{"NORMAL": <int>, "PNEUMONIA": <int>}``
        Missing directories are reported as ``0`` rather than raising.
    """
    data_dir = Path(data_dir)
    image_extensions = {".jpg", ".jpeg", ".png"}
    distribution: dict[str, int] = {}

    for class_name in ("NORMAL", "PNEUMONIA"):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            distribution[class_name] = 0
            continue

        count = sum(
            1
            for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        )
        distribution[class_name] = count

    return distribution


def preprocess_single_image(
    image_bytes: bytes,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """Preprocess raw image bytes for inference.

    Decodes *image_bytes* (e.g. the content of an uploaded file), resizes
    to *target_size*, normalises pixel values to ``[0, 1]``, and adds a
    batch dimension so the result is ready for ``model.predict()``.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the image (JPEG, PNG, etc.).
    target_size:
        ``(height, width)`` to resize the image to.  Defaults to
        ``(224, 224)``.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(1, height, width, 3)`` ready for
        ``model.predict()``.

    Raises
    ------
    ValueError
        If *image_bytes* cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    img = img.resize((target_size[1], target_size[0]))  # PIL: (width, height)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = np.expand_dims(arr, axis=0)               # (1, H, W, 3)
    return arr

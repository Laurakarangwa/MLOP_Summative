"""
model.py
--------
MobileNetV2 transfer-learning model factory and persistence helpers for the
Chest X-Ray Pneumonia Classifier.

All public functions handle TensorFlow imports internally so that this module
can be imported in environments where TF might not yet be on ``sys.path``
(the import will raise ``RuntimeError`` only when a function is actually
called without TF present).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# TensorFlow / Keras – lazy import
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    from tensorflow.keras.applications import MobileNetV2  # type: ignore
    from tensorflow.keras.metrics import Precision, Recall, AUC  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_tf() -> None:
    """Raise ``RuntimeError`` when TensorFlow is not installed."""
    if not _TF_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required. Install it with: pip install tensorflow"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    learning_rate: float = 1e-4,
) -> "keras.Model":
    """Build and compile a MobileNetV2 transfer-learning model.

    Architecture
    ------------
    1. **Base**: ``MobileNetV2`` pre-trained on ImageNet, with the
       classification head removed (``include_top=False``).  All base
       layers are **frozen** initially.
    2. **Head**::

           GlobalAveragePooling2D
           → Dense(128, relu)
           → Dropout(0.3)
           → Dense(64, relu)
           → Dropout(0.2)
           → Dense(num_classes, softmax)

    Compilation
    -----------
    - Optimiser: ``Adam`` with *learning_rate*
    - Loss: ``categorical_crossentropy``
    - Metrics: ``accuracy``, ``Precision``, ``Recall``, ``AUC``

    Parameters
    ----------
    input_shape:
        Shape of a single input image ``(H, W, C)``.
        Defaults to ``(224, 224, 3)``.
    num_classes:
        Number of output classes.  Defaults to ``2`` (NORMAL / PNEUMONIA).
    learning_rate:
        Learning rate passed to Adam.  Defaults to ``1e-4``.

    Returns
    -------
    keras.Model
        Compiled Keras model ready for ``model.fit()``.
    """
    _require_tf()

    # ---- Base model --------------------------------------------------------
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False  # freeze all base layers initially

    # ---- Custom head -------------------------------------------------------
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_0.3")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.2, name="dropout_0.2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="chest_xray_mobilenetv2")

    # ---- Compilation -------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )

    return model


def load_model(model_path: str | Path) -> "keras.Model":
    """Load a saved Keras model from *model_path*.

    Supports both ``.h5`` (legacy HDF5) and ``.keras`` (new native Keras
    format) files.  Custom metric objects (``Precision``, ``Recall``,
    ``AUC``) are registered via ``custom_objects`` so they survive
    serialisation round-trips.

    Parameters
    ----------
    model_path:
        Path to the saved model file.

    Returns
    -------
    keras.Model
        The loaded, compiled Keras model.

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist.
    RuntimeError
        If TensorFlow is not installed.
    """
    _require_tf()

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    custom_objects = {
        "Precision": Precision,
        "Recall": Recall,
        "AUC": AUC,
    }

    loaded = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    return loaded


def save_model(model: "keras.Model", model_path: str | Path) -> None:
    """Save *model* to *model_path*.

    Parent directories are created automatically.  The format is inferred
    from the file extension:
    - ``.h5`` → HDF5 (legacy, broad compatibility)
    - ``.keras`` → new native Keras format (recommended for TF ≥ 2.12)

    Parameters
    ----------
    model:
        A compiled (or fit) Keras model instance.
    model_path:
        Destination path.  Extension determines the save format.

    Raises
    ------
    RuntimeError
        If TensorFlow is not installed.
    """
    _require_tf()

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))


def unfreeze_top_layers(model: "keras.Model", num_layers: int = 20) -> "keras.Model":
    """Unfreeze the last *num_layers* layers of the MobileNetV2 base for fine-tuning.

    Fine-tuning works by making the top layers of the frozen base trainable
    so that they can adapt to the chest X-ray domain while the lower,
    more generic feature extractors remain frozen.

    After calling this function you should **recompile** the model with a
    lower learning rate (e.g. ``1e-5``) before continuing training.

    Parameters
    ----------
    model:
        A Keras model whose first layer is a ``MobileNetV2`` base model
        (as returned by :func:`build_model`).
    num_layers:
        Number of layers to unfreeze, counting from the end of the base
        model.  Defaults to ``20``.

    Returns
    -------
    keras.Model
        The same model object with the requested layers set to trainable.
        The model is **not** recompiled; call ``model.compile(...)``
        afterward to apply a new learning rate.

    Raises
    ------
    RuntimeError
        If TensorFlow is not installed or the model structure is unexpected.
    """
    _require_tf()

    # Locate the MobileNetV2 base layer (second layer after the Input)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise RuntimeError(
            "Could not find a nested Keras model (base) inside the provided model. "
            "Make sure the model was built with build_model()."
        )

    # Unfreeze the whole base first so we can selectively re-freeze
    base_model.trainable = True

    # Re-freeze every layer except the last `num_layers`
    freeze_until = max(0, len(base_model.layers) - num_layers)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    return model

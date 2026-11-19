"""
Chest X-Ray Pneumonia Classifier — FastAPI Backend
====================================================
Provides REST endpoints for model inference, dataset management,
retraining orchestration, and metrics reporting.

Endpoints
---------
GET  /health           → liveness probe
GET  /status           → model metadata & accuracy
GET  /metrics          → full metrics from models/metrics.json
POST /predict          → single-image inference
POST /upload           → batch image upload with label, logged to SQLite
POST /retrain          → kick off background fine-tuning
GET  /retrain/status   → poll retraining progress
GET  /visualizations   → aggregated chart data for the UI
"""

from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Optional TensorFlow import — handled gracefully so the API starts even when
# TF is not yet installed (e.g., during development / CI without GPU drivers).
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf  # noqa: F401 — imported for type hints & retraining
    TF_AVAILABLE = True
except ImportError:
    tf = None  # type: ignore[assignment]
    TF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("chest_xray_api")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[1]
MODEL_PATH: Path = ROOT / "models" / "chest_xray_model.h5"
METRICS_PATH: Path = ROOT / "models" / "metrics.json"
DATA_DIR: Path = ROOT / "data"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
DB_PATH: Path = DATA_DIR / "uploads.db"
PREDICTIONS_LOG: Path = DATA_DIR / "predictions_log.json"
TRAIN_DIR: Path = ROOT / "data" / "train"
CLASS_NAMES: List[str] = ["NORMAL", "PNEUMONIA"]

# ---------------------------------------------------------------------------
# Global mutable state
# ---------------------------------------------------------------------------
model: Any = None           # Loaded Keras model (or None)
APP_START_TIME: float = time.time()
RETRAIN_STATUS: Dict[str, str] = {"status": "idle", "message": ""}

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Chest X-Ray MLOps API",
    description="Inference, upload, and retraining API for the chest X-ray pneumonia classifier.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _init_db() -> None:
    """Create the uploads table in SQLite if it does not already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                filename  TEXT    NOT NULL,
                label     TEXT    NOT NULL,
                timestamp TEXT    NOT NULL
            )
            """
        )
        conn.commit()
    logger.info("SQLite DB ready at %s", DB_PATH)


def _log_upload(filename: str, label: str) -> None:
    """Append an upload record to the SQLite database."""
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO uploads (filename, label, timestamp) VALUES (?, ?, ?)",
            (filename, label, ts),
        )
        conn.commit()


def _get_upload_history() -> List[Dict[str, Any]]:
    """Return per-day upload counts grouped by label."""
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute(
            """
            SELECT substr(timestamp, 1, 10) AS date, label, COUNT(*) AS cnt
            FROM uploads
            GROUP BY date, label
            ORDER BY date DESC
            LIMIT 60
            """
        ).fetchall()
    return [{"date": r[0], "label": r[1], "count": r[2]} for r in rows]

# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------

def _log_prediction(result: Dict[str, Any]) -> None:
    """Append a prediction result as a JSON line to predictions_log.json."""
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {**result, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _load_predictions_log(limit: int = 200) -> List[Dict[str, Any]]:
    """Return the last *limit* prediction log entries."""
    if not PREDICTIONS_LOG.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries[-limit:]

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a normalised numpy array ready for inference.

    Parameters
    ----------
    image_bytes:
        Raw bytes of a JPEG/PNG image.

    Returns
    -------
    np.ndarray
        Shape (1, 224, 224, 3), dtype float32, values in [0, 1].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """
    Runs once when the FastAPI application starts.

    Initialises the SQLite database, creates required directories, and
    attempts to load the Keras model from MODEL_PATH.  If the model file
    does not exist yet (e.g., before the first training run) the app starts
    without a model — the ``/predict`` endpoint will return an appropriate
    error.
    """
    global model, APP_START_TIME
    APP_START_TIME = time.time()

    # Ensure directory tree exists
    for d in [DATA_DIR, UPLOADS_DIR, ROOT / "models"]:
        d.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        (UPLOADS_DIR / cls).mkdir(parents=True, exist_ok=True)

    _init_db()

    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available — running without model inference.")
        return

    if MODEL_PATH.exists():
        try:
            model = tf.keras.models.load_model(str(MODEL_PATH))
            logger.info("Model loaded from %s", MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            model = None
    else:
        logger.warning("Model file not found at %s — model is None.", MODEL_PATH)
        model = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness probe")
async def health() -> JSONResponse:
    """
    Returns the current health of the API service.

    Response fields
    ---------------
    status : str
        Always ``"ok"`` while the service is running.
    uptime_seconds : float
        Seconds since the application started.
    model_loaded : bool
        ``True`` when a Keras model is currently in memory.
    """
    return JSONResponse({
        "status": "ok",
        "uptime_seconds": round(time.time() - APP_START_TIME, 2),
        "model_loaded": model is not None,
    })


@app.get("/status", summary="Model status & metadata")
async def status() -> JSONResponse:
    """
    Returns high-level metadata about the trained model.

    Response fields
    ---------------
    model_path : str
        Absolute path to the model file (may not exist yet).
    model_exists : bool
        Whether the model file is present on disk.
    last_trained : str | null
        ISO-8601 timestamp of when the model file was last modified,
        or ``null`` if the file does not exist.
    accuracy : float | null
        Top-level accuracy from ``models/metrics.json``, or ``null``.
    """
    last_trained: Optional[str] = None
    if MODEL_PATH.exists():
        mtime = MODEL_PATH.stat().st_mtime
        last_trained = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    accuracy: Optional[float] = None
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            accuracy = metrics.get("accuracy")
        except (json.JSONDecodeError, OSError):
            pass

    return JSONResponse({
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "last_trained": last_trained,
        "accuracy": accuracy,
    })


@app.get("/metrics", summary="Full model metrics")
async def metrics() -> JSONResponse:
    """
    Returns the full contents of ``models/metrics.json``.

    Expected fields in the JSON file
    ---------------------------------
    accuracy, loss, f1, precision, recall, roc_auc

    Raises
    ------
    404
        If ``metrics.json`` does not exist yet.
    """
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found — train the model first.",
        )
    try:
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt metrics.json: {exc}") from exc
    return JSONResponse(data)


@app.post("/predict", summary="Single-image inference")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Run pneumonia classification on an uploaded chest X-ray image.

    Parameters
    ----------
    file : UploadFile
        JPEG or PNG chest X-ray image.

    Response fields
    ---------------
    class : str
        Predicted class — ``"NORMAL"`` or ``"PNEUMONIA"``.
    confidence : float
        Probability of the predicted class (0–1).
    probabilities : dict
        Per-class probabilities ``{"NORMAL": float, "PNEUMONIA": float}``.

    Raises
    ------
    503
        If TensorFlow is unavailable or no model is loaded.
    422
        If the uploaded file cannot be decoded as an image.
    """
    if not TF_AVAILABLE:
        raise HTTPException(status_code=503, detail="TensorFlow not installed.")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — train first.")

    image_bytes = await file.read()
    try:
        img_array = _preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not decode image: {exc}"
        ) from exc

    try:
        preds = model.predict(img_array, verbose=0)  # shape (1, 2)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {exc}"
        ) from exc

    probs = preds[0].tolist()
    pred_idx = int(np.argmax(probs))
    result = {
        "class": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 6),
        "probabilities": {
            CLASS_NAMES[0]: round(float(probs[0]), 6),
            CLASS_NAMES[1]: round(float(probs[1]), 6),
        },
    }
    _log_prediction(result)
    return JSONResponse(result)


@app.post("/upload", summary="Upload labelled images for retraining")
async def upload(
    files: List[UploadFile] = File(...),
    label: str = Form(...),
) -> JSONResponse:
    """
    Save one or more chest X-ray images to the uploads directory and
    record the upload in the SQLite database.

    Parameters
    ----------
    files : list[UploadFile]
        One or more JPEG/PNG images.
    label : str
        Class label — must be ``"NORMAL"`` or ``"PNEUMONIA"``.

    Response fields
    ---------------
    saved : int
        Number of files successfully saved.
    label : str
        The label applied to all files in this batch.

    Raises
    ------
    422
        If ``label`` is not one of the accepted values.
    """
    label = label.upper().strip()
    if label not in CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"label must be one of {CLASS_NAMES}, got '{label}'.",
        )

    dest_dir = UPLOADS_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for upload_file in files:
        content = await upload_file.read()
        if not content:
            continue
        # Sanitise filename to avoid path traversal
        safe_name = Path(upload_file.filename or "image.jpg").name
        ts_prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        dest_name = f"{ts_prefix}_{safe_name}"
        dest_path = dest_dir / dest_name
        dest_path.write_bytes(content)
        _log_upload(dest_name, label)
        saved += 1
        logger.info("Saved upload: %s → %s", safe_name, dest_path)

    return JSONResponse({"saved": saved, "label": label})


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

def _background_retrain() -> None:
    """
    Fine-tune the existing model on uploaded images + original training data.

    Strategy
    --------
    1. Load the existing model (if present) or build a fresh one.
    2. Collect image paths from ``data/uploads/`` and ``data/train/``.
    3. Use Keras ``ImageDataGenerator`` for augmented mini-batches.
    4. Fine-tune for a small number of epochs.
    5. Evaluate on a validation split and persist metrics.
    6. Save model back to MODEL_PATH.

    Status is kept in the global ``RETRAIN_STATUS`` dict so ``GET
    /retrain/status`` can poll it.
    """
    global model, RETRAIN_STATUS

    RETRAIN_STATUS["status"] = "running"
    RETRAIN_STATUS["message"] = "Loading model…"
    logger.info("Background retraining started.")

    if not TF_AVAILABLE:
        RETRAIN_STATUS["status"] = "failed"
        RETRAIN_STATUS["message"] = "TensorFlow not available."
        return

    try:
        # ---- 1. Load or build model ----------------------------------------
        if MODEL_PATH.exists():
            RETRAIN_STATUS["message"] = "Loading existing model…"
            current_model = tf.keras.models.load_model(str(MODEL_PATH))
            logger.info("Retrain: existing model loaded.")
        else:
            RETRAIN_STATUS["message"] = "Building new MobileNetV2 model…"
            base = tf.keras.applications.MobileNetV2(
                weights="imagenet", include_top=False, input_shape=(224, 224, 3)
            )
            base.trainable = False
            x = base.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(64, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            out = tf.keras.layers.Dense(2, activation="softmax")(x)
            current_model = tf.keras.Model(inputs=base.input, outputs=out)
            current_model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.info("Retrain: new model built.")

        # ---- 2. Determine data sources ------------------------------------
        RETRAIN_STATUS["message"] = "Preparing data generators…"
        # Prefer uploads if they exist, fall back to training directory
        data_root: Optional[Path] = None
        for candidate in [UPLOADS_DIR, TRAIN_DIR]:
            has_data = any(
                list((candidate / cls).glob("*.jpg"))
                + list((candidate / cls).glob("*.jpeg"))
                + list((candidate / cls).glob("*.png"))
                for cls in CLASS_NAMES
                if (candidate / cls).exists()
            )
            if has_data:
                data_root = candidate
                break

        if data_root is None:
            RETRAIN_STATUS["status"] = "failed"
            RETRAIN_STATUS["message"] = "No training images found in uploads/ or data/train/."
            return

        logger.info("Retrain: using data root %s", data_root)

        # ---- 3. Data generators -------------------------------------------
        datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            horizontal_flip=True,
            rotation_range=15,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
        )
        train_gen = datagen_train.flow_from_directory(
            str(data_root),
            target_size=(224, 224),
            batch_size=16,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )
        val_gen = datagen_train.flow_from_directory(
            str(data_root),
            target_size=(224, 224),
            batch_size=16,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

        if train_gen.samples == 0:
            RETRAIN_STATUS["status"] = "failed"
            RETRAIN_STATUS["message"] = "Data generator found 0 samples."
            return

        # ---- 4. Fine-tune --------------------------------------------------
        RETRAIN_STATUS["message"] = "Fine-tuning…"
        epochs = int(os.getenv("RETRAIN_EPOCHS", "5"))
        history = current_model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=0,
        )

        # ---- 5. Evaluate & persist metrics ---------------------------------
        RETRAIN_STATUS["message"] = "Evaluating & saving…"
        val_loss, val_acc = current_model.evaluate(val_gen, verbose=0)[:2]
        metrics_data: Dict[str, Any] = {
            "accuracy": round(float(val_acc), 6),
            "loss": round(float(val_loss), 6),
            "val_accuracy": round(float(val_acc), 6),
            "val_loss": round(float(val_loss), 6),
            "epochs_trained": epochs,
            "history": {
                "accuracy": [round(v, 6) for v in history.history.get("accuracy", [])],
                "val_accuracy": [round(v, 6) for v in history.history.get("val_accuracy", [])],
                "loss": [round(v, 6) for v in history.history.get("loss", [])],
                "val_loss": [round(v, 6) for v in history.history.get("val_loss", [])],
            },
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        # Merge with existing metrics if present
        if METRICS_PATH.exists():
            try:
                existing = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                existing.update(metrics_data)
                metrics_data = existing
            except (json.JSONDecodeError, OSError):
                pass
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
        logger.info("Metrics saved to %s", METRICS_PATH)

        # ---- 6. Save model -------------------------------------------------
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        current_model.save(str(MODEL_PATH))
        logger.info("Model saved to %s", MODEL_PATH)

        # Hot-swap the in-memory model
        model = current_model

        RETRAIN_STATUS["status"] = "completed"
        RETRAIN_STATUS["message"] = (
            f"Retraining complete — val_accuracy={val_acc:.4f}, val_loss={val_loss:.4f}"
        )
        logger.info("Background retraining completed successfully.")

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Retraining failed: %s", exc)
        RETRAIN_STATUS["status"] = "failed"
        RETRAIN_STATUS["message"] = str(exc)


@app.post("/retrain", summary="Trigger background model retraining")
async def retrain(background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Kick off a background fine-tuning job on uploaded images.

    Returns immediately with ``{"status": "retraining_started"}``.
    Poll ``GET /retrain/status`` to track progress.

    Raises
    ------
    409
        If a retraining job is already running.
    503
        If TensorFlow is not available.
    """
    if not TF_AVAILABLE:
        raise HTTPException(status_code=503, detail="TensorFlow not available.")
    if RETRAIN_STATUS["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="A retraining job is already running.",
        )

    RETRAIN_STATUS["status"] = "running"
    RETRAIN_STATUS["message"] = "Job queued…"

    t = threading.Thread(target=_background_retrain, daemon=True, name="retrain")
    t.start()
    logger.info("Retraining thread started.")

    return JSONResponse({"status": "retraining_started"})


@app.get("/retrain/status", summary="Poll retraining progress")
async def retrain_status() -> JSONResponse:
    """
    Returns the current status of the retraining pipeline.

    Response fields
    ---------------
    status : str
        One of ``"idle"``, ``"running"``, ``"completed"``, ``"failed"``.
    message : str
        Human-readable description of the current state.
    """
    return JSONResponse(RETRAIN_STATUS)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

@app.get("/visualizations", summary="Aggregated chart data")
async def visualizations() -> JSONResponse:
    """
    Returns aggregated data for the UI's visualisation page.

    Response fields
    ---------------
    class_distribution : dict
        ``{"NORMAL": int, "PNEUMONIA": int}`` — count of images in
        ``data/train/``.
    upload_history : list[dict]
        Per-day upload counts: ``[{"date": str, "label": str, "count": int}, …]``.
    confidence_distribution : list[float]
        Confidence scores from the most recent 200 predictions.
    """
    # Class distribution from training data
    class_dist: Dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
    for cls in CLASS_NAMES:
        cls_dir = TRAIN_DIR / cls
        if cls_dir.exists():
            class_dist[cls] = sum(
                1 for _ in cls_dir.iterdir()
                if _.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )

    # Upload history from DB
    upload_history = _get_upload_history()

    # Confidence distribution from prediction log
    recent_preds = _load_predictions_log(limit=200)
    confidence_dist = [
        round(float(p.get("confidence", 0)), 4)
        for p in recent_preds
        if "confidence" in p
    ]

    return JSONResponse({
        "class_distribution": class_dist,
        "upload_history": upload_history,
        "confidence_distribution": confidence_dist,
    })

"""Add this router to the existing API with app.include_router(router)."""

import io
import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.cnn_model import (
    ARCHITECTURE, CLASS_NAMES, PREPROCESSING, AssignmentCNN, make_transform,
)


router = APIRouter(tags=["Image classification"])
logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.environ.get(
    "CNN_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models" / "cnn_cifar10.pt")
))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
ACCEPTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ACCEPTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


class Prediction(BaseModel):
    class_name: str
    class_index: int
    probability: float


@lru_cache(maxsize=1)
def load_classifier():
    """Load lazily so the API's existing endpoints work before CNN training."""
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Classifier is unavailable. Train it with: uv run python train_cnn.py",
        )
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if checkpoint["architecture"] != ARCHITECTURE:
            raise ValueError("Unexpected CNN architecture")
        if checkpoint["preprocessing"] != PREPROCESSING:
            raise ValueError("Checkpoint preprocessing does not match the API")
        classes = checkpoint["class_names"]
        if not isinstance(classes, (list, tuple)) or len(classes) != 10:
            raise ValueError("Checkpoint must provide 10 class names")
        if set(classes) != set(CLASS_NAMES):
            raise ValueError("Checkpoint labels are not the CIFAR10 classes")
        model = AssignmentCNN()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return model, classes, make_transform()
    except (OSError, KeyError, ValueError, TypeError, RuntimeError,
            pickle.UnpicklingError, EOFError) as exc:
        logger.exception("Could not load the CNN checkpoint")
        raise HTTPException(status_code=503, detail="Classifier checkpoint cannot be loaded.") from exc


def classify_bytes(data: bytes) -> Prediction:
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(data)) as uploaded:
            if uploaded.format not in ACCEPTED_FORMATS:
                raise HTTPException(status_code=415, detail="Use a JPEG, PNG, WebP, or BMP image.")
            if uploaded.width * uploaded.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=400, detail="The image exceeds 16 million pixels.")
            image = ImageOps.exif_transpose(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The file is not a valid readable image.") from exc

    model, classes, transform = load_classifier()
    batch = transform(image).unsqueeze(0)  # [1, 3, 64, 64]
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1)[0]
        index = int(probabilities.argmax().item())
    return Prediction(
        class_name=classes[index], class_index=index,
        probability=float(probabilities[index].item()),
    )


@router.post("/classify", response_model=Prediction)
async def classify(file: UploadFile = File(...)) -> Prediction:
    """Return a CIFAR10 label and its softmax probability for an uploaded image."""
    try:
        if file.content_type and file.content_type not in ACCEPTED_TYPES:
            raise HTTPException(status_code=415, detail="Use a JPEG, PNG, WebP, or BMP image.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Maximum image upload size is 5 MiB.")
    # Pillow decoding and PyTorch inference run off the async event-loop thread.
    return await run_in_threadpool(classify_bytes, data)

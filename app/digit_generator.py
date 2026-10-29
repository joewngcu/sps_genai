"""Serve a generated MNIST image from the Assignment 3 checkpoint."""

import io
import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from app.gan_model import ARCHITECTURE, LATENT_DIM, Generator


router = APIRouter(tags=["Handwritten digit generation"])
logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.environ.get(
    "GAN_MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "models" / "gan_mnist.pt"),
)).expanduser().resolve()
TRAIN_COMMAND = "uv run python train_gan.py"


class DigitGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1, strict=True)


@lru_cache(maxsize=1)
def load_generator() -> Generator:
    """Load lazily so the other API endpoints work without GAN weights."""
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Digit generator is unavailable. Train it with: {TRAIN_COMMAND}",
        )
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if checkpoint["architecture"] != ARCHITECTURE:
            raise ValueError("Unexpected GAN architecture")
        if checkpoint["latent_dim"] != LATENT_DIM:
            raise ValueError("Unexpected latent dimension")
        epochs = checkpoint["epochs_completed"]
        if type(epochs) is not int or epochs < 1:
            raise ValueError("The generator has no completed training epochs")
        # Build without random parameter initialization, then assign the saved
        # CPU tensors. Even the first request leaves the global RNG untouched.
        with torch.device("meta"):
            model = Generator()
        model.load_state_dict(checkpoint["generator_state_dict"], strict=True, assign=True)
        model.eval()
        return model
    except (OSError, KeyError, ValueError, TypeError, RuntimeError,
            pickle.UnpicklingError, EOFError, AttributeError, IndexError) as exc:
        logger.exception("Could not load the GAN checkpoint")
        raise HTTPException(
            status_code=503,
            detail=f"Digit generator checkpoint cannot be loaded. Train it with: {TRAIN_COMMAND}",
        ) from exc


@router.post(
    "/generate-digit",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "A generated 28x28 grayscale digit image."}},
)
def generate_digit(request: DigitGenerationRequest | None = None) -> Response:
    """Generate one random digit; the unconditional GAN cannot select a class."""
    model = load_generator()
    request = request or DigitGenerationRequest()
    # A private CPU generator makes a seeded request reproducible without
    # changing PyTorch's global random state or another request's sequence.
    rng = torch.Generator(device="cpu")
    if request.seed is None:
        rng.seed()
    else:
        rng.manual_seed(request.seed)
    noise = torch.randn(1, LATENT_DIM, generator=rng)
    with torch.inference_mode():
        generated = model(noise)[0, 0]
        pixels = ((generated.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels.numpy()).save(output, format="PNG")
    return Response(
        content=output.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )

"""Checks for the assignment architecture and its actual PNG API response.

Run from the repository root with: python -m unittest discover -s tests -v
"""

import importlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from app import digit_generator
from app.gan_model import ARCHITECTURE, LATENT_DIM, Discriminator, Generator


class ArchitectureTests(unittest.TestCase):
    def test_exact_assignment_layers_and_shapes(self):
        generator = Generator().eval()
        discriminator = Discriminator().eval()
        self.assertEqual(LATENT_DIM, 100)
        linear = [m for m in generator.modules() if isinstance(m, nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in linear], [(100, 7 * 7 * 128)])
        transpose = [m for m in generator.modules() if isinstance(m, nn.ConvTranspose2d)]
        self.assertEqual(
            [(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.output_padding)
             for m in transpose],
            [(128, 64, (4, 4), (2, 2), (1, 1), (0, 0)),
             (64, 1, (4, 4), (2, 2), (1, 1), (0, 0))],
        )
        self.assertEqual([m.num_features for m in generator.modules() if isinstance(m, nn.BatchNorm2d)], [64])
        self.assertEqual(sum(isinstance(m, nn.ReLU) for m in generator.modules()), 1)
        self.assertEqual(sum(isinstance(m, nn.Tanh) for m in generator.modules()), 1)

        conv = [m for m in discriminator.modules() if isinstance(m, nn.Conv2d)]
        self.assertEqual(
            [(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding) for m in conv],
            [(1, 64, (4, 4), (2, 2), (1, 1)), (64, 128, (4, 4), (2, 2), (1, 1))],
        )
        self.assertEqual([m.num_features for m in discriminator.modules() if isinstance(m, nn.BatchNorm2d)], [128])
        self.assertEqual([m.negative_slope for m in discriminator.modules() if isinstance(m, nn.LeakyReLU)], [0.2, 0.2])
        self.assertEqual(
            [(m.in_features, m.out_features) for m in discriminator.modules() if isinstance(m, nn.Linear)],
            [(7 * 7 * 128, 1)],
        )
        self.assertEqual(sum(isinstance(m, nn.Sigmoid) for m in discriminator.modules()), 1)
        with torch.inference_mode():
            generated = generator(torch.randn(4, LATENT_DIM))
            probability = discriminator(generated)
        self.assertEqual(tuple(generated.shape), (4, 1, 28, 28))
        self.assertEqual(probability.numel(), 4)
        self.assertTrue(torch.isfinite(generated).all())
        self.assertTrue((generated >= -1).all() and (generated <= 1).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())


class DigitAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoint_path = Path(cls.directory.name) / "test_generator.pt"
        # Synthetic weights test serialization and inference, not training quality.
        cls.checkpoint = {
            "architecture": ARCHITECTURE,
            "latent_dim": LATENT_DIM,
            "generator_state_dict": Generator().state_dict(),
            "epochs_completed": 1,
        }
        torch.save(cls.checkpoint, cls.checkpoint_path)
        app = FastAPI()
        app.include_router(digit_generator.router)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.directory.cleanup()

    def setUp(self):
        self.path_patch = patch.object(digit_generator, "MODEL_PATH", self.checkpoint_path)
        self.path_patch.start()
        digit_generator.load_generator.cache_clear()

    def tearDown(self):
        digit_generator.load_generator.cache_clear()
        self.path_patch.stop()

    def test_returns_grayscale_png_and_seed_repeats(self):
        first = self.client.post("/generate-digit", json={"seed": 42})
        second = self.client.post("/generate-digit", json={"seed": 42})
        different = self.client.post("/generate-digit", json={"seed": 43})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.headers["content-type"], "image/png")
        self.assertEqual(first.content, second.content)
        self.assertNotEqual(first.content, different.content)
        with Image.open(io.BytesIO(first.content)) as image:
            self.assertEqual(image.size, (28, 28))
            self.assertEqual(image.mode, "L")

    def test_default_body_and_empty_json_work(self):
        self.assertEqual(self.client.post("/generate-digit").status_code, 200)
        self.assertEqual(self.client.post("/generate-digit", json={}).status_code, 200)

    def test_seed_does_not_change_global_rng(self):
        before = torch.random.get_rng_state().clone()
        response = self.client.post("/generate-digit", json={"seed": 123})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_invalid_requests_are_rejected(self):
        for body in ({"seed": -1}, {"seed": 2**63}, {"seed": 1.5},
                     {"seed": "123"}, {"seed": True}, {"digit": 7}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/generate-digit", json=body).status_code, 422)

    def test_missing_checkpoint_is_unavailable(self):
        with patch.object(digit_generator, "MODEL_PATH", Path(self.directory.name) / "missing.pt"):
            response = self.client.post("/generate-digit")
        self.assertEqual(response.status_code, 503)
        self.assertIn("train_gan.py", response.json()["detail"])

    @patch.object(digit_generator.logger, "exception")
    def test_corrupt_and_incompatible_checkpoints_are_unavailable(self, log_exception):
        bad_path = Path(self.directory.name) / "bad.pt"
        bad_path.write_bytes(b"this is not a PyTorch checkpoint")
        with patch.object(digit_generator, "MODEL_PATH", bad_path):
            self.assertEqual(self.client.post("/generate-digit").status_code, 503)
        variants = [
            {**self.checkpoint, "architecture": "other"},
            {**self.checkpoint, "latent_dim": 99},
            {**self.checkpoint, "epochs_completed": 0},
            {**self.checkpoint, "generator_state_dict": {}},
        ]
        for checkpoint in variants:
            with self.subTest(keys=list(checkpoint)):
                torch.save(checkpoint, bad_path)
                with patch.object(digit_generator, "MODEL_PATH", bad_path):
                    self.assertEqual(self.client.post("/generate-digit").status_code, 503)
        self.assertEqual(log_exception.call_count, 5)

    def test_router_is_registered_with_existing_endpoints(self):
        # The embedding model's multi-gigabyte spaCy load is unrelated to routing.
        stub = types.ModuleType("app.embedding_model")
        stub.calculate_embedding = lambda word: None
        with patch.dict("sys.modules", {"app.embedding_model": stub}):
            main = importlib.import_module("app.main")
        routes = {(path, method.upper()) for path, operations in main.app.openapi()["paths"].items()
                  for method in operations}
        self.assertTrue({("/", "GET"), ("/generate", "POST"),
                         ("/embedding", "POST"), ("/classify", "POST"),
                         ("/generate-digit", "POST")}.issubset(routes))


if __name__ == "__main__":
    unittest.main()

"""Train the Assignment 3 GAN on all 60,000 MNIST training images."""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import certifi
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from app.gan_model import (
    ARCHITECTURE, LATENT_DIM, PREPROCESSING, Discriminator, Generator,
    initialize_weights,
)


ROOT = Path(__file__).resolve().parent


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, generator: Generator, discriminator: Discriminator,
                    optimizer_g, optimizer_d, epoch: int, config: dict,
                    loader_rng: torch.Generator) -> None:
    checkpoint = {
        "architecture": ARCHITECTURE,
        "latent_dim": LATENT_DIM,
        "preprocessing": PREPROCESSING,
        "epochs_completed": epoch,
        "training_config": config,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "loader_rng_state": loader_rng.get_state(),
    }
    # Replace atomically so serving never reads a partially written checkpoint.
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--resume", type=Path, help="Resume a checkpoint from this script")
    parser.add_argument("--max-batches", type=int, help="Limit batches per epoch for a smoke test only")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        parser.error("epochs, batch-size, lr, and threads must be positive")
    if args.seed < 0 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("seed must be nonnegative and max-batches must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "gan_samples"
    sample_dir.mkdir(exist_ok=True)

    # Python.org builds on macOS may not have a system CA bundle configured.
    # Keep certificate verification enabled using the project's certifi bundle.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    dataset = MNIST(root=args.data_dir, train=True, download=True)
    # MNIST is already 28x28 grayscale. Convert once, rather than decoding a PIL
    # image at every step. Map uint8 [0,255] to the generator's Tanh range [-1,1].
    images = dataset.data.unsqueeze(1).float().div_(127.5).sub_(1.0)
    loader_rng = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(images), batch_size=args.batch_size,
                        shuffle=True, generator=loader_rng, num_workers=0,
                        pin_memory=device.type == "cuda", drop_last=False)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()  # Discriminator.forward returns probabilities.
    config = {
        "dataset": "MNIST", "split": "train", "training_images": len(images),
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "adam_betas": [0.5, 0.999], "seed": args.seed,
        "device": str(device), "torch_version": str(torch.__version__),
        "max_batches": args.max_batches,
    }
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint["architecture"] != ARCHITECTURE:
            raise ValueError("Cannot resume a different GAN architecture")
        if checkpoint["training_config"] != config:
            raise ValueError("Resume with the same training configuration and device")
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        start_epoch = int(checkpoint["epochs_completed"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        loader_rng.set_state(checkpoint["loader_rng_state"].cpu())
    if args.epochs <= start_epoch:
        parser.error("epochs must exceed the checkpoint's completed epochs")

    # CPU RNG for noise also makes saved RNG state portable across accelerators.
    fixed_noise = torch.randn(64, LATENT_DIM,
                             generator=torch.Generator().manual_seed(args.seed + 1)).to(device)
    (args.output_dir / "gan_training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    metrics_path = args.output_dir / "gan_metrics.csv"
    append_metrics = bool(args.resume) and metrics_path.exists()
    with metrics_path.open("a" if append_metrics else "w", newline="") as metrics_file:
        fields = ["epoch", "images_seen", "d_loss", "g_loss", "d_real", "d_fake", "seconds"]
        writer = csv.DictWriter(metrics_file, fieldnames=fields)
        if not append_metrics:
            writer.writeheader()
        print(f"Training on {device}: {len(images)} images, {len(loader)} batches/epoch", flush=True)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            started = time.perf_counter()
            generator.train()
            discriminator.train()
            totals = {name: 0.0 for name in ("d_loss", "g_loss", "d_real", "d_fake")}
            seen = 0
            for batch_index, (real_images,) in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                real_images = real_images.to(device)
                batch_size = real_images.shape[0]
                ones = torch.ones(batch_size, 1, device=device)
                zeros = torch.zeros(batch_size, 1, device=device)
                noise = torch.randn(batch_size, LATENT_DIM).to(device)
                fake_images = generator(noise)

                # D learns real=1 and generated=0. Detach prevents this update
                # from changing G through the fake-image computation graph.
                optimizer_d.zero_grad(set_to_none=True)
                real_scores = discriminator(real_images)
                fake_scores = discriminator(fake_images.detach())
                d_loss = (criterion(real_scores, ones) + criterion(fake_scores, zeros)) / 2
                d_loss.backward()
                optimizer_d.step()

                # G learns to make D label generated images as real. Gradients
                # pass through D to G, while D's weights stay fixed for this step.
                discriminator.requires_grad_(False)
                optimizer_g.zero_grad(set_to_none=True)
                g_loss = criterion(discriminator(fake_images), ones)
                g_loss.backward()
                optimizer_g.step()
                discriminator.requires_grad_(True)

                seen += batch_size
                for name, value in (("d_loss", d_loss), ("g_loss", g_loss),
                                    ("d_real", real_scores.mean()), ("d_fake", fake_scores.mean())):
                    totals[name] += float(value.detach().item()) * batch_size

            generator.eval()
            with torch.inference_mode():
                samples = generator(fixed_noise).cpu().add(1).div(2).clamp(0, 1)
            save_image(samples, sample_dir / f"epoch_{epoch:03d}.png", nrow=8)
            save_checkpoint(args.output_dir / "gan_mnist.pt", generator, discriminator,
                            optimizer_g, optimizer_d, epoch, config, loader_rng)
            row = {"epoch": epoch, "images_seen": seen,
                   **{key: round(value / seen, 6) for key, value in totals.items()},
                   "seconds": round(time.perf_counter() - started, 3)}
            writer.writerow(row)
            metrics_file.flush()
            print(json.dumps(row), flush=True)
    print(f"Saved trained checkpoint: {args.output_dir / 'gan_mnist.pt'}", flush=True)


if __name__ == "__main__":
    main()

"""Train on CIFAR10; select by validation accuracy; evaluate the test set once."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from app.cnn_model import ARCHITECTURE, PREPROCESSING, AssignmentCNN, make_transform


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device auto or --device cpu.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device auto or --device cpu.")
    return torch.device(requested)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_size
    return {"loss": total_loss / count, "accuracy": correct / count}


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_size
    return {"loss": total_loss / count, "accuracy": correct / count}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Fixed seeds make the data split repeatable; hardware/software can still
    # change floating-point results between runs.
    transform = make_transform()
    training_data = CIFAR10(
        root=str(args.data_dir), train=True, download=True, transform=transform
    )
    test_data = CIFAR10(
        root=str(args.data_dir), train=False, download=True, transform=transform
    )
    if args.download_only:
        print(f"CIFAR10 ready: {len(training_data)} training and {len(test_data)} test images.")
        return
    if not 1 <= args.validation_size < len(training_data):
        raise ValueError("validation-size must leave at least one training image")

    train_size = len(training_data) - args.validation_size
    train_data, validation_data = random_split(
        training_data,
        [train_size, args.validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    device = choose_device(args.device)
    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "cnn_cifar10.pt"
    best_accuracy, best_epoch = -1.0, 0
    history = []
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}; parameters: {parameter_count:,}; train/validation/test: "
          f"{len(train_data)}/{len(validation_data)}/{len(test_data)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        training = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation = evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train": training, "validation": validation})
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train loss={training['loss']:.4f}, accuracy={training['accuracy']:.4f}; "
            f"validation loss={validation['loss']:.4f}, accuracy={validation['accuracy']:.4f}",
            flush=True,
        )
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            checkpoint = {
                "architecture": ARCHITECTURE,
                "model_state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
                "class_names": list(training_data.classes),
                "preprocessing": PREPROCESSING,
                "training": {
                    "epoch": epoch, "seed": args.seed,
                    "train_size": train_size, "validation_size": args.validation_size,
                    "validation_accuracy": best_accuracy,
                    "optimizer": "Adam", "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                },
            }
            torch.save(checkpoint, checkpoint_path)

    # Select using validation data only. The held-out test data is evaluated here.
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test = evaluate(model, test_loader, criterion, device)
    metrics = {
        "dataset": "CIFAR10", "architecture": ARCHITECTURE,
        "seed": args.seed, "device": str(device),
        "parameters": parameter_count,
        "train_size": len(train_data), "validation_size": len(validation_data),
        "test_size": len(test_data), "epochs_run": args.epochs,
        "selected_epoch": best_epoch, "selection_metric": "validation_accuracy",
        "best_validation_accuracy": best_accuracy,
        "test": test, "history": history,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Selected epoch {best_epoch}; test accuracy={test['accuracy']:.4f}, "
          f"test loss={test['loss']:.4f}", flush=True)
    print(f"Saved {checkpoint_path} and {metrics_path}", flush=True)


if __name__ == "__main__":
    main()

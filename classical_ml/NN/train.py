import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import ImprovedCNN

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 20
PATIENCE = 5

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_transforms(kind):
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    if kind == "baseline":
        train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15,
                saturation=0.15, hue=0.05
            ),
            transforms.ToTensor(),
        ])
    return train_tf, eval_tf

def get_split(data_root, split_path):
    raw = datasets.ImageFolder(data_root)
    indices = np.arange(len(raw))
    labels = np.asarray(raw.targets)

    if split_path.exists():
        s = np.load(split_path)
        return raw.classes, s["train_indices"], s["val_indices"]

    train_idx, val_idx = train_test_split(
        indices, test_size=0.15,
        random_state=SEED, stratify=labels
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split_path, train_indices=train_idx, val_indices=val_idx)
    return raw.classes, train_idx, val_idx

def epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    loss_sum = correct = total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            out = model(x)
            loss = criterion(out, y)

            if training:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/train")
    p.add_argument("--augmentation", choices=["baseline", "colorjitter"], required=True)
    p.add_argument("--output-dir", default="checkpoints")
    args = p.parse_args()

    seed_everything()
    device = get_device()
    root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = get_transforms(args.augmentation)
    classes, train_idx, val_idx = get_split(
        root, out / "split_indices.npz"
    )

    train_ds = datasets.ImageFolder(root, transform=train_tf)
    val_ds = datasets.ImageFolder(root, transform=eval_tf)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = ImprovedCNN(len(classes)).to(device)
    params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    print("=" * 80)
    print(f"{args.augmentation.upper()} CNN — LOCAL")
    print("=" * 80)
    print(f"Device            : {device}")
    if device.type == "cuda":
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
    print(f"Training images   : {len(train_idx)}")
    print(f"Validation images : {len(val_idx)}")
    print(f"Parameters        : {params}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Learning rate     : {LR}")
    print(f"Maximum epochs    : {MAX_EPOCHS}")
    print(f"Early stopping    : {PATIENCE}")
    print("=" * 80)

    total_start = time.time()

    for e in range(1, MAX_EPOCHS + 1):
        start = time.time()

        tr_loss, tr_acc = epoch(
            model, train_loader, criterion, optimizer, device, True
        )
        va_loss, va_acc = epoch(
            model, val_loader, criterion, None, device, False
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = e
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            marker = " <-- BEST"
        else:
            stale += 1
            marker = ""

        elapsed = time.time() - start
        history.append([e, tr_loss, tr_acc, va_loss, va_acc, elapsed])

        print(
            f"Epoch [{e:02d}/{MAX_EPOCHS:02d}] | "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.4f} | "
            f"Time: {elapsed:.1f}s{marker}"
        )

        if stale >= PATIENCE:
            print(f"\nEarly stopping after {e} epochs.")
            break

    model.load_state_dict(best_state)

    checkpoint = out / f"{args.augmentation}_cnn.pth"
    history_file = out / f"{args.augmentation}_history.csv"

    torch.save(model.state_dict(), checkpoint)
    pd.DataFrame(
        history,
        columns=["epoch", "train_loss", "train_accuracy",
                 "val_loss", "val_accuracy", "epoch_time_seconds"]
    ).to_csv(history_file, index=False)

    total_time = time.time() - total_start

    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best epoch        : {best_epoch}")
    print(f"Best validation   : {best_acc:.4f}")
    print(f"Total training    : {total_time / 60:.2f} min")
    print(f"Checkpoint saved  : {checkpoint}")
    print("=" * 80)

if __name__ == "__main__":
    main()

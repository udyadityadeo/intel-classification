import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ImprovedCNN


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "test"


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    device = get_device()

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        args.data_root,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = ImprovedCNN(
        len(dataset.classes)
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    labels = []
    predictions = []

    loss_sum = 0.0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            loss_sum += (
                loss.item() * x.size(0)
            )

            labels.extend(
                y.cpu().numpy()
            )

            predictions.extend(
                output.argmax(1)
                .cpu()
                .numpy()
            )

            total += y.size(0)

    test_loss = loss_sum / total
    test_accuracy = accuracy_score(
        labels,
        predictions,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print("=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Device        : {device}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            labels,
            predictions,
            target_names=dataset.classes,
            digits=4,
        )
    )

    print("Confusion Matrix:")

    print(
        pd.DataFrame(
            cm,
            index=dataset.classes,
            columns=dataset.classes,
        )
    )


if __name__ == "__main__":
    main()
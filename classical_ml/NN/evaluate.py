import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ImprovedCNN

def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/test")
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    dev = device()
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    ds = datasets.ImageFolder(args.data_root, transform=tf)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    model = ImprovedCNN(len(ds.classes))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(dev).eval()

    criterion = nn.CrossEntropyLoss()
    labels, preds = [], []
    loss_sum = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            labels.extend(y.cpu().numpy())
            preds.extend(out.argmax(1).cpu().numpy())
            total += y.size(0)

    cm = confusion_matrix(labels, preds)

    print("=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Device        : {dev}")
    print(f"Test Loss     : {loss_sum / total:.4f}")
    print(f"Test Accuracy : {accuracy_score(labels, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=ds.classes, digits=4))
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=ds.classes, columns=ds.classes))

if __name__ == "__main__":
    main()

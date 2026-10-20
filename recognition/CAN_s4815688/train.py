"""Train the U-Net / CAN baseline on HipMRI 2D prostate slices."""

import argparse
import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, PROSTATE, make_datasets, pad_batch
from metrics import dice_per_class, iou_per_class, update_confusion
from modules import build_model


class CombinedLoss(nn.Module):
    def __init__(self, n_classes, class_weights=None, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        union = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * intersection + 1) / (union + 1)
        present = onehot.sum(dims) > 0
        foreground = dice[1:][present[1:]]
        dice_loss = 1 - foreground.mean() if foreground.numel() else logits.new_zeros(())
        return self.ce_weight * ce + self.dice_weight * dice_loss


def class_weights(dataset):
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for mask in dataset.labels:
        counts += np.bincount(mask.reshape(-1), minlength=len(CLASS_NAMES))
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq + 1e-8)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.2, 5.0)
    print("class pixels:", dict(zip(CLASS_NAMES, counts.astype(int).tolist())))
    print("class weights:", np.round(weights, 3).tolist())
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total, seen = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.size(0)
        seen += images.size(0)
    return total / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    n_classes = len(CLASS_NAMES)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    total, seen = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        update_confusion(cm, logits.argmax(1).cpu().numpy(), masks.cpu().numpy(), n_classes)
        total += loss.item() * images.size(0)
        seen += images.size(0)
    return total / max(seen, 1), dice_per_class(cm), iou_per_class(cm), cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    parser.add_argument("--model", choices=("unet", "can"), default="unet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set, val_set, _ = make_datasets(args.data_root, limit=args.limit)
    print(f"train {len(train_set)} slices, val {len(val_set)} slices")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True,
        collate_fn=pad_batch,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=pad_batch,
    )

    model = build_model(args.model, n_classes=len(CLASS_NAMES), base=args.base).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    criterion = CombinedLoss(
        len(CLASS_NAMES), class_weights=class_weights(train_set).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_dice, best_epoch, stale = -1.0, 0, 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice, val_iou, _ = evaluate(model, val_loader, criterion, device)

        score = float(val_dice[PROSTATE])
        if np.isnan(score):
            score = float(np.nanmean(val_dice[1:]))
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_prostate_dice": score,
            "val_foreground_dice": float(np.nanmean(val_dice[1:])),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"{epoch:2d} train {train_loss:.4f} val {val_loss:.4f} "
            f"prostate dice {score:.4f} fg dice {row['val_foreground_dice']:.4f} "
            f"({time.time() - start:.0f}s)"
        )

        if score > best_dice:
            best_dice, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "classes": CLASS_NAMES,
                    "epoch": epoch,
                    "val_dice": score,
                },
                os.path.join(args.out, f"{args.model}_best.pt"),
            )
        else:
            stale += 1
            if stale >= args.patience:
                print("early stop")
                break

    with open(os.path.join(args.out, f"{args.model}_history.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_prostate_dice"] for row in history])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val prostate dice")
    figure.tight_layout()
    figure.savefig(os.path.join(args.out, f"{args.model}_history.png"), dpi=150)

    print(f"best epoch {best_epoch}, prostate dice {best_dice:.4f}")


if __name__ == "__main__":
    main()

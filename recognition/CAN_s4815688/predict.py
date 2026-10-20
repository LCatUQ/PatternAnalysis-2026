"""Evaluate a checkpoint on a split and save per-slice records and figures."""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, PROSTATE, discover, HipMRISlices, pad_batch
from metrics import dice_binary, dice_per_class, iou_per_class, update_confusion
from modules import build_model


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model"],
        n_classes=len(CLASS_NAMES),
        base=checkpoint.get("args", {}).get("base", 32),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def overlay(axis, image, mask):
    axis.imshow(image, cmap="gray")
    rgba = np.zeros(mask.shape + (4,))
    rgba[mask == PROSTATE] = (1.0, 0.15, 0.15, 0.55)
    rgba[mask == 3] = (1.0, 1.0, 0.1, 0.30)
    rgba[mask == 4] = (0.1, 1.0, 0.3, 0.30)
    axis.imshow(rgba)
    axis.axis("off")


def diff_overlay(axis, image, pred, target):
    axis.imshow(image, cmap="gray")
    rgba = np.zeros(target.shape + (4,))
    rgba[(pred == PROSTATE) & (target == PROSTATE)] = (1.0, 1.0, 1.0, 0.65)
    rgba[(pred == PROSTATE) & (target != PROSTATE)] = (1.0, 0.1, 0.1, 0.65)
    rgba[(pred != PROSTATE) & (target == PROSTATE)] = (0.1, 0.4, 1.0, 0.65)
    axis.imshow(rgba)
    axis.axis("off")


def save_failure_panel(dataset, rows, path):
    figure, axes = plt.subplots(len(rows), 4, figsize=(10, 2.7 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for axis_row, row in zip(axes, rows):
        index = int(row["dataset_index"])
        image = dataset.images[index].astype(np.float32)
        target = dataset.labels[index]
        pred = row["pred_mask"]
        overlay(axis_row[0], image, np.zeros_like(target))
        overlay(axis_row[1], image, target)
        overlay(axis_row[2], image, pred)
        diff_overlay(axis_row[3], image, pred, target)
        axis_row[0].set_title(f"{row['case']}w{row['week']} s{row['slice']}", fontsize=8)
        axis_row[1].set_title("ground truth", fontsize=8)
        axis_row[2].set_title(f"prediction, dice {row['dice']:.3f}", fontsize=8)
        axis_row[3].set_title("TP / FP / FN", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_position_plot(rows, path):
    slices = [row for row in rows if np.isfinite(row["dice"])]
    if not slices:
        return
    position = np.array([row["position"] for row in slices])
    dice = np.array([row["dice"] for row in slices])
    bins = np.linspace(0, 1, 11)
    centres, means = [], []
    for low, high in zip(bins[:-1], bins[1:]):
        chosen = dice[(position >= low) & (position < high)]
        if chosen.size:
            centres.append((low + high) / 2)
            means.append(chosen.mean())

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(position, dice, s=9, alpha=0.35, label="slices")
    axis.plot(centres, means, color="black", marker="o", label="bin mean")
    axis.set_xlabel("normalised slice position (apex to base)")
    axis.set_ylabel("prostate dice")
    axis.set_ylim(-0.05, 1.05)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_risk_coverage(rows, path):
    slices = [row for row in rows if np.isfinite(row["dice"]) and np.isfinite(row["confidence"])]
    if not slices:
        return
    confidence = np.array([row["confidence"] for row in slices])
    dice = np.array([row["dice"] for row in slices])
    order = np.argsort(-confidence)
    dice = dice[order]
    coverage = np.arange(1, len(dice) + 1) / len(dice)
    accepted = np.cumsum(dice) / np.arange(1, len(dice) + 1)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(coverage, accepted, color="black")
    axis.set_xlabel("coverage (highest-confidence fraction retained)")
    axis.set_ylabel("mean dice on retained slices")
    axis.set_ylim(0, 1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    parser.add_argument("--split", default="test", choices=("validate", "test"))
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--tag", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    tag = args.tag or checkpoint["model"]

    refs = discover(args.data_root, args.split)
    dataset = HipMRISlices(refs, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=pad_batch,
    )

    n_classes = len(CLASS_NAMES)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    rows = []
    per_case = defaultdict(list)
    max_slice = defaultdict(int)
    for ref in refs:
        key = (ref.case, ref.week)
        max_slice[key] = max(max_slice[key], ref.index)
    inference_seconds = 0.0
    seen = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for batch_index, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.time() - start

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1).cpu().numpy().astype(np.uint8)
        confidence = probabilities.max(dim=1).values.cpu().numpy()
        targets = masks.numpy().astype(np.uint8)

        for i in range(images.size(0)):
            index = batch_index * args.batch_size + i
            ref = dataset.refs[index]
            height, width = dataset.labels[index].shape
            pred = predictions[i][:height, :width]
            target = targets[i][:height, :width]
            confidence_map = confidence[i][:height, :width]
            update_confusion(confusion, pred, target, n_classes)
            dice = dice_binary(pred, target, PROSTATE)

            region = (pred == PROSTATE) | (target == PROSTATE)
            conf = float(confidence_map[region].mean()) if region.any() else float("nan")

            pred_area = int((pred == PROSTATE).sum())
            leak = 0.0
            if pred_area:
                leak = float(((pred == PROSTATE) & ((target == 3) | (target == 4))).sum() / pred_area)

            per_case[(ref.case, ref.week)].append(dice)
            rows.append({
                "case": ref.case,
                "week": ref.week,
                "slice": ref.index,
                "position": ref.index / max(max_slice[(ref.case, ref.week)], 1),
                "dice": dice,
                "confidence": conf,
                "leak": leak,
                "gt_area": int((target == PROSTATE).sum()),
                "pred_area": pred_area,
                "dataset_index": index,
                "pred_mask": pred,
            })
            seen += images.size(0)

    dice_classes = dice_per_class(confusion)
    iou_classes = iou_per_class(confusion)
    valid_dice = np.array([row["dice"] for row in rows if np.isfinite(row["dice"])])

    print(f"\n{tag} on {args.split}: {seen} slices")
    print(f"{'class':10s} {'dice':>7s} {'iou':>7s}")
    for name, dice, iou in zip(CLASS_NAMES, dice_classes, iou_classes):
        print(f"{name:10s} {dice:7.3f} {iou:7.3f}")
    print(f"prostate dice mean {valid_dice.mean():.3f} median {np.median(valid_dice):.3f} "
          f"n={valid_dice.size}")

    print("\nper case:")
    for (case, week), values in sorted(per_case.items()):
        finite = [value for value in values if np.isfinite(value)]
        if finite:
            print(f"  case {case} week {week}: {np.mean(finite):.3f} over {len(finite)} slices")

    peak_memory = (
        torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    )
    milliseconds = 1000 * inference_seconds / max(seen, 1)
    print(f"\nparameters {sum(p.numel() for p in model.parameters()):,}  "
          f"peak vram {peak_memory:.0f} MB  {milliseconds:.1f} ms/slice")

    failures = sorted(
        (row for row in rows if np.isfinite(row["dice"])),
        key=lambda row: row["dice"],
    )
    chosen, cases = [], set()
    for row in failures:
        key = (row["case"], row["week"])
        if key not in cases:
            chosen.append(row)
            cases.add(key)
        if len(chosen) == 5:
            break
    if chosen:
        save_failure_panel(dataset, chosen, os.path.join(args.out, f"{tag}_failures.png"))

    for row in rows:
        row.pop("pred_mask")

    with open(os.path.join(args.out, f"{tag}_slices.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "tag": tag,
        "split": args.split,
        "slices": seen,
        "classes": CLASS_NAMES,
        "dice": {name: float(value) for name, value in zip(CLASS_NAMES, dice_classes)},
        "iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou_classes)},
        "prostate_dice_mean": float(valid_dice.mean()),
        "prostate_dice_median": float(np.median(valid_dice)),
        "leak_mean": float(np.nanmean([row["leak"] for row in rows])),
        "params": sum(p.numel() for p in model.parameters()),
        "peak_vram_mb": peak_memory,
        "ms_per_slice": milliseconds,
    }
    with open(os.path.join(args.out, f"{tag}_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    save_position_plot(rows, os.path.join(args.out, f"{tag}_position.png"))
    save_risk_coverage(rows, os.path.join(args.out, f"{tag}_risk_coverage.png"))


if __name__ == "__main__":
    main()

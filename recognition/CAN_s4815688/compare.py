"""Compare two evaluated models: per-class scores, slice position, risk-coverage."""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_ORDER = ("background", "body", "bone", "bladder", "rectum", "prostate")


def load_slices(path):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "case": row["case"],
                "week": int(row["week"]),
                "slice": int(row["slice"]),
                "position": float(row["position"]),
                "dice": float(row["dice"]),
                "confidence": float(row["confidence"]),
                "leak": float(row["leak"]),
                "gt_area": int(row["gt_area"]),
                "pred_area": int(row["pred_area"]),
            })
    return rows


def finite(rows):
    return [row for row in rows if np.isfinite(row["dice"])]


def key(row):
    return (row["case"], row["week"], row["slice"])


def position_curve(rows):
    rows = finite(rows)
    edges = np.linspace(0, 1, 11)
    centres, means, counts = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = [row["dice"] for row in rows if low <= row["position"] < high]
        if chosen:
            centres.append((low + high) / 2)
            means.append(float(np.mean(chosen)))
            counts.append(len(chosen))
    return np.array(centres), np.array(means), np.array(counts)


def risk_curve(rows):
    rows = [row for row in finite(rows) if np.isfinite(row["confidence"])]
    rows.sort(key=lambda row: -row["confidence"])
    dice = np.array([row["dice"] for row in rows])
    coverage = np.arange(1, len(dice) + 1) / len(dice)
    accepted = np.cumsum(dice) / np.arange(1, len(dice) + 1)
    return coverage, accepted, rows


def operating_point(rows, coverage=0.9):
    coverage_curve, accepted, ordered = risk_curve(rows)
    index = max(0, min(len(accepted) - 1, int(round(coverage * len(accepted))) - 1))
    return {
        "coverage": float(coverage_curve[index]),
        "dice": float(accepted[index]),
        "threshold": float(ordered[index]["confidence"]),
    }


def bootstrap_ci(values, rounds=5000, seed=0):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(rounds, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/eval")
    parser.add_argument("--tags", nargs=2, default=("unet", "can"))
    args = parser.parse_args()

    tags = args.tags
    slices = {tag: load_slices(os.path.join(args.dir, f"{tag}_slices.csv")) for tag in tags}
    summaries = {}
    for tag in tags:
        with open(os.path.join(args.dir, f"{tag}_summary.json")) as handle:
            summaries[tag] = json.load(handle)

    paired = {}
    for tag in tags:
        paired[tag] = {key(row): row for row in finite(slices[tag])}
    shared = sorted(set(paired[tags[0]]) & set(paired[tags[1]]))

    lines = [f"# {tags[0]} vs {tags[1]}", ""]
    lines += ["## Per-class test scores", "",
              f"| class | {tags[0]} dice | {tags[1]} dice | delta |",
              "|---|---|---|---|"]
    for name in CLASS_ORDER:
        left = summaries[tags[0]]["dice"].get(name, float("nan"))
        right = summaries[tags[1]]["dice"].get(name, float("nan"))
        lines.append(f"| {name} | {left:.3f} | {right:.3f} | {right - left:+.3f} |")

    lines += ["", "## Prostate", "",
              f"| metric | {tags[0]} | {tags[1]} |",
              "|---|---|---|"]
    for label, field in (("mean dice", "prostate_dice_mean"), ("median dice", "prostate_dice_median")):
        lines.append(f"| {label} | {summaries[tags[0]][field]:.3f} | {summaries[tags[1]][field]:.3f} |")

    if shared:
        diffs = np.array([
            paired[tags[1]][item]["dice"] - paired[tags[0]][item]["dice"] for item in shared
        ])
        low, high = bootstrap_ci(diffs)
        wins = int((diffs > 0.01).sum())
        losses = int((diffs < -0.01).sum())
        ties = len(diffs) - wins - losses
        lines += [
            f"| paired slices | {len(shared)} | |",
            f"| mean paired difference | {diffs.mean():+.3f} [{low:+.3f}, {high:+.3f}] | |",
            f"| wins / ties / losses | {wins} / {ties} / {losses} | |",
        ]

    lines += ["", "## Operating point at 90% coverage", "",
              f"| metric | {tags[0]} | {tags[1]} |",
              "|---|---|---|"]
    points = {tag: operating_point(slices[tag]) for tag in tags}
    lines += [
        f"| retained slices | {points[tags[0]]['coverage']:.0%} | {points[tags[1]]['coverage']:.0%} |",
        f"| dice on retained | {points[tags[0]]['dice']:.3f} | {points[tags[1]]['dice']:.3f} |",
        f"| confidence threshold | {points[tags[0]]['threshold']:.3f} | {points[tags[1]]['threshold']:.3f} |",
    ]

    lines += ["", "## Boundary leakage", "",
              f"| metric | {tags[0]} | {tags[1]} |",
              "|---|---|---|"]
    leaks = {tag: np.array([row["leak"] for row in slices[tag]]) for tag in tags}
    lines.append(f"| mean leak fraction | {leaks[tags[0]].mean():.3f} | {leaks[tags[1]].mean():.3f} |")
    lines.append(f"| slices with leak > 0.1 | "
                 f"{np.mean(leaks[tags[0]] > 0.1):.1%} | {np.mean(leaks[tags[1]] > 0.1):.1%} |")

    lines += ["", "## Resources", "",
              f"| metric | {tags[0]} | {tags[1]} |",
              "|---|---|---|",
              f"| parameters | {summaries[tags[0]]['params']:,} | {summaries[tags[1]]['params']:,} |",
              f"| peak vram (MB) | {summaries[tags[0]]['peak_vram_mb']:.0f} | {summaries[tags[1]]['peak_vram_mb']:.0f} |",
              f"| ms per slice | {summaries[tags[0]]['ms_per_slice']:.1f} | {summaries[tags[1]]['ms_per_slice']:.1f} |"]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    positions = np.arange(len(CLASS_ORDER))
    width = 0.38
    for offset, tag in enumerate(tags):
        scores = [summaries[tag]["dice"].get(name, np.nan) for name in CLASS_ORDER]
        axes[0, 0].bar(positions + (offset - 0.5) * width, scores, width, label=tag)
    axes[0, 0].set_xticks(positions, CLASS_ORDER, rotation=30, ha="right")
    axes[0, 0].set_ylabel("dice")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()

    for tag in tags:
        centres, means, _ = position_curve(slices[tag])
        axes[0, 1].plot(centres, means, marker="o", label=tag)
    axes[0, 1].set_xlabel("normalised slice position (apex to base)")
    axes[0, 1].set_ylabel("prostate dice")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    for tag in tags:
        coverage, accepted, _ = risk_curve(slices[tag])
        axes[1, 0].plot(coverage, accepted, label=tag)
    axes[1, 0].set_xlabel("coverage retained")
    axes[1, 0].set_ylabel("mean dice on retained")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    if shared:
        axes[1, 1].hist(diffs, bins=30)
        axes[1, 1].axvline(0, color="black", linewidth=1)
        axes[1, 1].axvline(diffs.mean(), color="red", linestyle="--",
                           label=f"mean {diffs.mean():+.3f}")
        axes[1, 1].set_xlabel(f"{tags[1]} minus {tags[0]} dice")
        axes[1, 1].set_ylabel("slices")
        axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(os.path.join(args.dir, "comparison.png"), dpi=150)
    plt.close(figure)

    with open(os.path.join(args.dir, "comparison.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

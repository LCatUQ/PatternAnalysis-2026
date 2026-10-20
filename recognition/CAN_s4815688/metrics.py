"""Confusion-matrix metrics for multi-class segmentation."""

import numpy as np


def update_confusion(cm, pred, target, n_classes):
    pred = pred.reshape(-1).astype(np.int64)
    target = target.reshape(-1).astype(np.int64)
    valid = (target >= 0) & (target < n_classes)
    idx = target[valid] * n_classes + pred[valid]
    cm += np.bincount(idx, minlength=n_classes**2).reshape(n_classes, n_classes)


def dice_per_class(cm):
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1)
    return np.divide(2 * tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)


def iou_per_class(cm):
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    return np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)


def dice_binary(pred, target, label):
    pred = pred == label
    target = target == label
    denom = pred.sum() + target.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (pred & target).sum() / denom)

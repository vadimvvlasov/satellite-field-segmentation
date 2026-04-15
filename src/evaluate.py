"""Evaluation metrics for field boundary segmentation.

Metrics:
- IoU (Intersection over Union) — strict
- MCC (Matthews Correlation Coefficient)
- F1 score
"""

import numpy as np


def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute IoU for binary segmentation masks."""
    pred_bin = (pred > threshold).astype(bool)
    target_bin = (target > threshold).astype(bool)
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    return float(intersection / union) if union > 0 else 1.0


def compute_mcc(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Matthews Correlation Coefficient."""
    pred_bin = (pred > threshold).astype(bool)
    target_bin = (target > threshold).astype(bool)
    tp = np.logical_and(pred_bin, target_bin).sum()
    fp = np.logical_and(pred_bin, ~target_bin).sum()
    tn = np.logical_and(~pred_bin, ~target_bin).sum()
    fn = np.logical_and(~pred_bin, target_bin).sum()

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def compute_f1(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute F1 score."""
    pred_bin = (pred > threshold).astype(bool)
    target_bin = (target > threshold).astype(bool)
    tp = np.logical_and(pred_bin, target_bin).sum()
    fp = np.logical_and(pred_bin, ~target_bin).sum()
    fn = np.logical_and(~pred_bin, target_bin).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute all metrics at once."""
    return {
        "iou": compute_iou(pred, target),
        "mcc": compute_mcc(pred, target),
        "f1": compute_f1(pred, target),
    }

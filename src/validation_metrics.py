
import numpy as np


def calculate_metrics(ground_truth, prediction):
    """Calculate binary segmentation metrics from two masks."""
    ground_truth = np.asarray(ground_truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)

    if ground_truth.shape != prediction.shape:
        raise ValueError("Ground-truth and prediction masks must have the same shape.")

    true_positive = int(np.logical_and(ground_truth, prediction).sum())
    false_positive = int(np.logical_and(~ground_truth, prediction).sum())
    false_negative = int(np.logical_and(ground_truth, ~prediction).sum())

    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    return {
        "iou": true_positive / union if union else 1.0,
        "dice": (
            2 * true_positive / dice_denominator
            if dice_denominator else 1.0
        ),
        "precision": (
            true_positive / precision_denominator
            if precision_denominator else 1.0
        ),
        "recall": (
            true_positive / recall_denominator
            if recall_denominator else 1.0
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }

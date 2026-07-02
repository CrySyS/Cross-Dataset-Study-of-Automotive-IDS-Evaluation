
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    roc_curve as sk_roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    auc as sk_auc,
    average_precision_score,
)

# -------------------------------------------------------------------
# Basic wrappers used by cli.py
# -------------------------------------------------------------------

def roc_curve(y_true, scores):
    """
    Thin wrapper around sklearn.metrics.roc_curve to keep a stable API.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    return sk_roc_curve(y_true, scores, pos_label=1)


def auc(x, y):
    """
    Wrapper around sklearn.metrics.auc returning a Python float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(sk_auc(x, y))


def youden_threshold(scores, labels) -> float:
    """
    Compute Youden's J = TPR - FPR and return the threshold that maximizes it.

    Args:
        scores: anomaly scores (higher = more anomalous / more likely attack)
        labels: 0 = benign, 1 = attack
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)

    if np.unique(y).size < 2:
        # Degenerate case: only one class present
        return float(np.median(s))

    fpr, tpr, thr = sk_roc_curve(y, s, pos_label=1)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k])


def best_f1_threshold(scores, labels) -> float:
    """
    Return the score threshold that maximizes F1.

    We evaluate F1 at each threshold produced by roc_curve.
    For large datasets (>1M samples), we subsample thresholds to avoid O(n*m) blowup.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)

    if np.unique(y).size < 2:
        return float(np.median(s))

    fpr, tpr, thr = sk_roc_curve(y, s, pos_label=1)

    # OPTIMIZATION: For large datasets, subsample thresholds
    # This prevents O(n * m) memory/CPU explosion where n=17.5M, m=100K+
    max_thresholds = 1000
    if len(thr) > max_thresholds:
        # Keep first, last, and evenly spaced samples
        indices = np.linspace(0, len(thr) - 1, max_thresholds, dtype=int)
        thr_sampled = thr[indices]
    else:
        thr_sampled = thr

    best_f1 = -1.0
    best_thr = float(np.median(s))

    for t in thr_sampled:
        y_pred = (s >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)

    return best_thr


def threshold_for_fpr(fpr, thresholds, target_fpr: float):
    """
    Given an ROC curve (fpr, thresholds) pick the threshold whose FPR is
    closest to target_fpr.

    Returns:
        float threshold, or None if fpr/thresholds are empty.
    """
    fpr = np.asarray(fpr, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if fpr.size == 0 or thresholds.size == 0:
        return None

    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thresholds[idx])




def confusion_dict(y_true, y_pred_bin) -> Dict[str, float]:
    """
    Confusion matrix + basic rates + precision/recall/F1.

    Matches what cli.py expects:
      - tn, fp, fn, tp
      - precision, recall, f1
      - tpr (recall), fpr, tnr, fnr
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_bin = np.asarray(y_pred_bin, dtype=int)

    uniq_true, cnt_true = np.unique(y_true, return_counts=True)
    if not np.all(np.isin(uniq_true, [0, 1])):
        dist = {int(v): int(c) for v, c in zip(uniq_true, cnt_true)}
        raise ValueError(
            f"confusion_dict expected binary y_true in {{0,1}}, got {dist}. "
            "This usually indicates label contamination in an evaluation subset."
        )

    uniq_pred, cnt_pred = np.unique(y_pred_bin, return_counts=True)
    if not np.all(np.isin(uniq_pred, [0, 1])):
        dist = {int(v): int(c) for v, c in zip(uniq_pred, cnt_pred)}
        raise ValueError(
            f"confusion_dict expected binary y_pred in {{0,1}}, got {dist}. "
            "For binary evaluation, predictions must be hard labels 0/1."
        )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_bin, average="binary", zero_division=0
    )

    # Rates
    pos = tp + fn  # number of attacks
    neg = tn + fp  # number of benign

    tpr = tp / pos if pos > 0 else 0.0       # recall
    fnr = fn / pos if pos > 0 else 0.0
    fpr = fp / neg if neg > 0 else 0.0
    tnr = tn / neg if neg > 0 else 0.0

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "fnr": float(fnr),
    }


def balanced_accuracy_from_conf(conf: Dict[str, Any]) -> Optional[float]:
    """Compute balanced accuracy = (TPR + TNR) / 2 from a confusion dict."""
    if conf is None:
        return None

    tpr = conf.get("tpr")
    tnr = conf.get("tnr")

    # If missing, derive from counts
    if tpr is None or tnr is None:
        tn = conf.get("tn")
        fp = conf.get("fp")
        fn = conf.get("fn")
        tp = conf.get("tp")

        if tp is not None and fn is not None:
            denom = tp + fn
            if denom:
                tpr = tp / denom
        if tn is not None and fp is not None:
            denom = tn + fp
            if denom:
                tnr = tn / denom

    if tpr is None or tnr is None:
        return None

    return float((tpr + tnr) / 2.0)



def three_sigma_threshold(scores, labels, use_only_normal: bool = True, k: float = 3.0) -> float:
    """
    Compute mu + k*sigma using ONLY benign samples by default
    (labels == 0).

    Args:
        scores: anomaly scores (higher = more anomalous)
        labels: 0 = benign, 1 = attack
        use_only_normal: if True, use only scores where label == 0
        k: multiplier for sigma (3.0 for "3-sigma rule")
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if use_only_normal:
        normal_scores = scores[labels == 0]
    else:
        normal_scores = scores

    if normal_scores.size > 0:
        mu = normal_scores.mean()
        sigma = normal_scores.std()
    else:
        mu = scores.mean()
        sigma = scores.std()

    thr = mu + k * sigma
    return float(thr)


# -------------------------------------------------------------------
# Your original helpers (kept, with fixed signatures where needed)
# -------------------------------------------------------------------

def roc_pr(scores: List[float], labels: List[int]) -> Dict[str, float]:
    """
    Convenience function returning ROC AUC and PR AUC.
    Not used by cli_eval, but kept for completeness.
    """
    y = np.array(labels, dtype=int)
    s = np.array(scores, dtype=float)

    if np.unique(y).size < 2:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
        }

    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
    }


def roc_auc(scores, labels) -> float:
    """
    Simple wrapper around roc_auc_score.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    return float(roc_auc_score(y, s)) if np.unique(y).size > 1 else float("nan")


def pick_threshold_from_validation(
    y_val: np.ndarray,
    s_val: np.ndarray,
    mode: str = "youden",
    fixed_fpr: float = 0.01,
) -> float:
    """
    Return decision threshold on scores (higher = worse) from validation.
    """
    y_val = np.asarray(y_val, dtype=int)
    s_val = np.asarray(s_val, dtype=float)

    if np.unique(y_val).size < 2:
        return float(np.median(s_val))

    fpr, tpr, thr = sk_roc_curve(y_val, s_val, pos_label=1)

    if mode == "youden":
        j = tpr - fpr
        return float(thr[np.argmax(j)])

    # fixed FPR
    idx = int(np.argmin(np.abs(fpr - fixed_fpr)))
    return float(thr[idx])


def confusion_rates(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    """
    Alternative confusion-rate view (Hit, False Alarm, etc).
    Kept for compatibility; not used directly by cli_eval.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_bin = np.asarray(y_pred_bin, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()
    CA, CN = tp + fn, tn + fp

    return {
        "Hit (TPR)": (tp / CA) if CA else 0.0,
        "False Alarm (FPR)": (fp / CN) if CN else 0.0,
        "Miss (FNR)": (fn / CA) if CA else 0.0,
        "Correct Reject (TNR)": (tn / CN) if CN else 0.0,
        "Precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
    }


def threshold_at_target_fpr(y_true, scores, target_fpr=0.065):
    """
    Original helper: choose threshold for a given target FPR by scanning
    the ROC curve. Not used by cli_eval directly, but can be handy.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, thr = sk_roc_curve(y_true, scores, pos_label=1)
    if fpr.size == 0:
        return float(np.median(scores)), float("nan"), float("nan")
    i = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thr[i]), float(fpr[i]), float(tpr[i])

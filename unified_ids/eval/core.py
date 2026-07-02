
"""
Core evaluation logic for Unified IDS.

Contains:
  - data loading
  - single (method, dataset) experiment runner
  - metric & ROC helpers
  - output directory helpers
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve as sk_roc_curve,
    auc,
    precision_recall_fscore_support,
)

from unified_ids.config.method_capabilities import METHOD_CAPS
from unified_ids.methods.registry import METHOD_REGISTRY


# --------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------


def prepare_out_dir(base_out: str, method: str, dataset_tag: str) -> Path:
    """Create and return output directory: base_out / dataset_tag / method."""
    out_dir = Path(base_out) / dataset_tag / method
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _infer_dataset_tag_from_glob(glob_path: str) -> str:
    """
    Best-effort dataset label inference from a glob path.

    We try to capture both the dataset root and the immediate parent folder
    (e.g., attack type) to avoid collisions when running multiple subsets
    of the same dataset. Example:
      data_parquet/05_DAGA_STABILI2022/ArbitrarySequenceReplay/infected/*.parquet
    → label: 05_DAGA_STABILI2022_ArbitrarySequenceReplay

    If anything fails, returns "unknown_glob".
    """
    try:
        # Trim the wildcard tail to get a concrete path prefix
        prefix = glob_path.split("*")[0]
        p = Path(prefix).resolve()

        # parent is typically "infected" or "clean"; grandparent is attack type
        parent = p.parent.name
        grandparent = p.parent.parent.name if p.parent else ""

        # If parent is clean/infected, use grandparent as the specific subset
        if parent in {"clean", "infected"} and grandparent:
            attack = grandparent
        else:
            attack = parent or "subset"

        dataset_root = p.parent.parent.parent.name if p.parent and p.parent.parent else "dataset"

        label = f"{dataset_root}_{attack}".replace(" ", "_")
        return label or "unknown_glob"
    except Exception:
        return "unknown_glob"


def experiment_already_done(base_out: str, dataset_tag: str, method: str) -> bool:
    """
    Return True if we already have results for (dataset_tag, method)
    in base_out.

    Uses presence of metrics.json as a completion signal.
    """
    metrics_path = Path(base_out) / dataset_tag / method / "metrics.json"
    return metrics_path.exists()


def infer_dataset_tag_from_df(df: pd.DataFrame) -> str:
    """Try to infer a dataset tag from canonical df (dataset column)."""
    if "dataset" in df.columns:
        vals = sorted(str(x) for x in df["dataset"].dropna().unique())
        if len(vals) == 1:
            if "attack_type" in df.columns:
                attack_types = sorted(str(x) for x in df["attack_type"].dropna().unique())
                if len(attack_types) == 1:
                    return f"{vals[0]}_{attack_types[0]}".replace(" ", "_")
            return vals[0]
        elif len(vals) > 1:
            head = "+".join(vals[:3])
            suffix = "+etc" if len(vals) > 3 else ""
            return f"mixed_{head}{suffix}"
    return "unknown"


# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------


def load_train_test(
    train_glob: str,
    test_glob: str,
    log,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test dataframes for a dataset using explicit globs.

    All datasets must have explicit benign-only training files and 
    separate test files. Heuristic splitting is deprecated.

    Args:
        train_glob: Glob pattern for benign-only training files
        test_glob: Glob pattern for test files (benign + attack)
        log: Logger instance

    Returns:
        (df_train, df_test) tuple of concatenated DataFrames
    """
    from unified_ids.dataio.loaders import read_parquet_glob

    if not train_glob or not test_glob:
        raise SystemExit("ERROR: train_glob and test_glob are required. "
                        "Provide explicit globs for benign-only training and test files.")
    
    log.info("Loading train data from: %s", train_glob)
    df_tr = read_parquet_glob(train_glob)
    
    log.info("Loading test data from: %s", test_glob)
    df_te = read_parquet_glob(test_glob)

    log.info(
        "Data loaded: train=%d rows, test=%d rows",
        len(df_tr),
        len(df_te),
    )
    
    # Validate data quality
    _validate_dataset_quality(df_tr, df_te, log)
    
    return df_tr, df_te


def _validate_dataset_quality(df_train: pd.DataFrame, df_test: pd.DataFrame, log):
    """
    Perform data quality checks on loaded datasets.
    
    Validates:
    - Label column presence and values
    - Attack type consistency with labels
    - Class balance in test set
    - Training set purity (should be benign-only)
    """
    # Check training data (should be benign-only)
    if "label" in df_train.columns:
        train_attacks = df_train["label"].sum()
        if train_attacks > 0:
            log.warning(
                "Training data quality issue: %d attack messages found (%.2f%%). "
                "Training should be benign-only.",
                int(train_attacks),
                100.0 * train_attacks / len(df_train)
            )
    
    # Check test data
    if "label" not in df_test.columns:
        log.error("Test data missing 'label' column!")
        return
    
    n_test = len(df_test)
    n_benign = int((df_test["label"] == 0).sum())
    n_attack = int((df_test["label"] == 1).sum())
    n_other = n_test - n_benign - n_attack
    
    log.info(
        "Test data composition: %d benign (%.1f%%), %d attack (%.1f%%), %d other labels",
        n_benign, 100.0 * n_benign / n_test,
        n_attack, 100.0 * n_attack / n_test,
        n_other
    )
    
    if n_attack == 0:
        log.warning("Test data has NO attack messages - metrics will be undefined!")
    elif n_benign == 0:
        log.warning("Test data has NO benign messages - metrics will be undefined!")
    
    # Check attack_type consistency
    if "attack_type" in df_test.columns:
        benign_mask = df_test["label"] == 0
        attack_mask = df_test["label"] == 1
        benign_attack_types = df_test.loc[benign_mask, "attack_type"]
        attack_attack_types = df_test.loc[attack_mask, "attack_type"]
        
        # IMPROVED: Check for actual data problems, not just presence of attack_type
        # It's OK for benign messages to have attack_type IF they're from a trace with attacks
        # (e.g., DAGA dataset: benign prefix messages in "MessageIDFuzzing" trace)
        
        # Check if benign messages have attack_type values
        benign_non_null = benign_attack_types.notna().sum()
        benign_empty = (benign_attack_types.astype(str).str.strip() == "").sum()
        benign_real_values = benign_non_null - benign_empty
        
        # Check if attack messages are missing attack_type values
        attack_non_null = attack_attack_types.notna().sum()
        attack_empty = (attack_attack_types.astype(str).str.strip() == "").sum()
        attack_missing = n_attack - attack_non_null
        
        # Only warn if EITHER:
        # 1. Attack messages are missing attack_type (indicates incomplete labeling)
        # 2. Benign and attack messages have completely different attack_type values
        #    (indicates benign messages shouldn't have attack_type at all)
        
        if attack_missing > 0:
            log.warning(
                "Data quality issue: %d attack messages (%.1f%%) missing attack_type value. "
                "These will be assigned 'attack_unknown' for per-attack-type evaluation.",
                int(attack_missing),
                100.0 * attack_missing / n_attack if n_attack > 0 else 0
            )
        
        # If benign messages have consistent attack_type with attacks (same values),
        # it's OK - they're from the same trace. Only warn if there's real inconsistency.
        if benign_real_values > 0 and n_attack > 0:
            benign_types = set(benign_attack_types[benign_attack_types.notna()].unique())
            attack_types = set(attack_attack_types[attack_attack_types.notna()].unique())
            
            # If benign messages have attack types that are NOT in the attack messages,
            # check if it's a real inconsistency (shared attack types but different benign types)
            # vs. just separate datasets (completely different benign and attack types).
            benign_only_types = benign_types - attack_types
            shared_types = benign_types & attack_types
            
            # Only warn if:
            # 1. There ARE shared attack types (indicating benign + attack in same file), AND
            # 2. Benign has additional types not in attacks (inconsistency within that context)
            # This filters out false positives like OTIDS where benign="AttackFree" and attack="DoS"
            if benign_only_types and shared_types:
                log.warning(
                    "Data quality inconsistency: Benign messages have attack_type values "
                    "not found in any attack messages: %s. "
                    "These will be normalized to 'benign' for evaluation.",
                    sorted(benign_only_types)
                )
            # Otherwise, if benign and attack messages share attack_type values,
            # it's legitimate (e.g., both are from same trace)


def _validate_per_attack_consistency(per_attack_results: Dict, 
                                      total_attack_count: int, 
                                      total_benign_count: int,
                                      log):
    """
    Validate that per-attack-type metrics are consistent with overall counts.
    
    This catches bugs where individual attack type counts don't sum correctly
    or where benign counts are inconsistent.
    """
    if not per_attack_results:
        return
    
    # Sum up attack counts across all attack types
    sum_attacks = sum(result.get("n_attack", 0) for result in per_attack_results.values())
    
    # Check benign consistency (should be same for all attack types)
    benign_counts = {result.get("n_benign", 0) for result in per_attack_results.values()}
    
    log.info(
        "Per-attack validation: %d attack types, sum_attacks=%d (expected %d), benign_counts=%s (expected %d)",
        len(per_attack_results),
        sum_attacks,
        total_attack_count,
        benign_counts,
        total_benign_count
    )
    
    # Validate benign count consistency
    if len(benign_counts) > 1:
        log.error(
            "INCONSISTENCY: Per-attack-type results have different n_benign values: %s. "
            "All attack types should use the same benign set!",
            benign_counts,
        )
    elif benign_counts and list(benign_counts)[0] != total_benign_count:
        log.error(
            "INCONSISTENCY: Per-attack n_benign=%d != overall n_benign=%d",
            list(benign_counts)[0],
            total_benign_count,
        )
    
    # Validate attack count sum
    if sum_attacks != total_attack_count:
        log.warning(
            "INCONSISTENCY: Sum of per-attack n_attack (%d) != total attack count (%d). "
            "Difference: %d. This may indicate overlapping attack types or missing data.",
            sum_attacks,
            total_attack_count,
            sum_attacks - total_attack_count,
        )
    
    # Check for suspicious results (e.g., single_class when it shouldn't be)
    for attack_type, result in per_attack_results.items():
        if "single_class" in result:
            log.error(
                "INCONSISTENCY: Attack type '%s' marked as single_class=%s but should have both benign and attack samples!",
                attack_type,
                result["single_class"],
            )


def _evaluate_scores_generic(
    scores: np.ndarray,
    labels: np.ndarray,
    metrics_mod,
    target_fprs: Tuple[float, ...],
) -> Dict[str, Any]:
    """
    Generic scoring evaluation for continuous anomaly scores.
    
    Computes ROC curves, AUC, and various threshold-based metrics
    (Youden, best F1, 3-sigma, target FPR).
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0=benign, 1=attack)
        metrics_mod: metrics module with roc_curve, auc, etc.
        target_fprs: FPR targets for threshold computation
        
    Returns:
        Dictionary with ROC curves, AUC, and threshold-based metrics.
        Returns single_class metadata if labels contain only one class.

    Note:
        Binary methods should bypass this via is_binary=True in
        evaluate_scores_window/message.
    """
    import logging
    log = logging.getLogger("unified_ids")
    
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # --- handle single-class evaluation cleanly ---
    uniq = np.unique(labels)
    if uniq.size < 2:
        n = int(labels.size)
        n_pos = int(labels.sum())
        n_neg = n - n_pos
        single_class = int(uniq[0]) if uniq.size == 1 else None

        # Return a JSON-serializable dict with consistent keys
        return {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "single_class": single_class,  # 0 = all benign, 1 = all attack
            "roc_auc": None,
            "roc_curve": None,
            "pr_auc": None,
            "pr_curve": None,
            "best_f1": None,
            "best_f1_threshold": None,
            "threshold_for_target_fpr": None,
            "note": "ROC/PR metrics undefined because labels contain only one class at this evaluation unit.",
        }

    # ROC
    log.info("Computing ROC curve for %d samples...", len(scores))
    fpr, tpr, thrs = metrics_mod.roc_curve(labels, scores)
    roc_auc = metrics_mod.auc(fpr, tpr)
    log.info("ROC curve computed: AUC=%.4f, %d thresholds", roc_auc, len(thrs))
    # Downsample ROC arrays for metrics.json to prevent huge files on large datasets
    MAX_ROC_POINTS = 4096
    if len(thrs) > MAX_ROC_POINTS:
        step = int(np.ceil(len(thrs) / MAX_ROC_POINTS))
        fpr_ds = fpr[::step]
        tpr_ds = tpr[::step]
        thrs_ds = thrs[::step]
    else:
        fpr_ds, tpr_ds, thrs_ds = fpr, tpr, thrs
    roc_pr = {
        "fpr": fpr_ds.tolist(),
        "tpr": tpr_ds.tolist(),
        "thresholds": thrs_ds.tolist(),
    }

    # Youden
    log.info("Computing Youden threshold...")
    thr_youden = metrics_mod.youden_threshold(scores, labels)
    yhat_youden = (scores >= thr_youden).astype(int)
    conf_youden = metrics_mod.confusion_dict(labels, yhat_youden)

    # Best F1
    log.info("Computing best F1 threshold (may take several minutes for large datasets)...")
    thr_best_f1 = metrics_mod.best_f1_threshold(scores, labels)
    log.info("Best F1 threshold computed: %.6f", thr_best_f1)
    yhat_best = (scores >= thr_best_f1).astype(int)
    conf_best = metrics_mod.confusion_dict(labels, yhat_best)
    ba_best = metrics_mod.balanced_accuracy_from_conf(conf_best)

    # Fixed FPR targets
    fixed = {}
    for tfpr in target_fprs:
        thr = metrics_mod.threshold_for_fpr(fpr, thrs, target_fpr=tfpr)
        if thr is None:
            continue
        yhat = (scores >= thr).astype(int)
        conf = metrics_mod.confusion_dict(labels, yhat)
        ba_fixed = metrics_mod.balanced_accuracy_from_conf(conf)
        fixed[f"fpr_{tfpr}"] = {
            "target_fpr": tfpr,
            "chosen_threshold": float(thr),
            "achieved_fpr": float(conf["fpr"]),
            "achieved_tpr": float(conf["tpr"]),
            "confusion": conf,
            "balanced_accuracy": ba_fixed,
        }

    # 3-sigma rule on benign scores
    thr_three_sigma = metrics_mod.three_sigma_threshold(scores, labels)
    yhat_sigma = (scores >= thr_three_sigma).astype(int)
    conf_sigma = metrics_mod.confusion_dict(labels, yhat_sigma)

    thresholds = {
        "youden": {
            "threshold": float(thr_youden),
            "confusion": conf_youden,
            "balanced_accuracy": metrics_mod.balanced_accuracy_from_conf(conf_youden),
        },
        "best_f1": {
            "threshold": float(thr_best_f1),
            "confusion": conf_best,
            "balanced_accuracy": ba_best,
        },
        "three_sigma": {
            "threshold": float(thr_three_sigma),
            "confusion": conf_sigma,
            "balanced_accuracy": metrics_mod.balanced_accuracy_from_conf(conf_sigma),
        },
    }
    thresholds.update(fixed)

    return {
        "roc_pr": roc_pr,
        "roc_auc": float(roc_auc),
        "thresholds": thresholds,
        "best_f1_threshold": float(thr_best_f1),  # Return for ROC plotting
        "balanced_accuracy_best_f1": ba_best,
    }


def evaluate_scores_window(
    scores: np.ndarray,
    labels: np.ndarray,
    metrics_mod,
    target_fprs: Tuple[float, ...] = (0.001, 0.01, 0.05),
    is_binary: bool = False,
) -> Dict[str, Any]:
    """
    Window-level evaluation wrapper.
    
    Args:
        scores: Window-level anomaly scores
        labels: Window-level ground truth labels
        metrics_mod: metrics module
        target_fprs: FPR targets for threshold computation
        is_binary: True if scores are binary (0/1), False for continuous
        
    Returns:
        Dictionary with window-level evaluation metrics
    """
    if is_binary:
        y_pred = np.asarray(scores, dtype=int)
        labels = np.asarray(labels, dtype=int)
        conf = metrics_mod.confusion_dict(labels, y_pred)
        ba = metrics_mod.balanced_accuracy_from_conf(conf)
        return {
            "roc_pr": None,
            "roc_auc": None,
            "thresholds": {},
            "confusion_at_best_f1": conf,
            "balanced_accuracy": ba,
            "note": "Binary method: no threshold sweep or ROC.",
        }
    return _evaluate_scores_generic(scores, labels, metrics_mod, target_fprs)


def evaluate_scores_message(
    scores: np.ndarray,
    labels: np.ndarray,
    metrics_mod,
    target_fprs: Tuple[float, ...] = (0.001, 0.01, 0.05),
    is_binary: bool = False,
) -> Dict[str, Any]:
    """
    Message-level evaluation wrapper.
    
    Args:
        scores: Message-level anomaly scores
        labels: Message-level ground truth labels
        metrics_mod: metrics module
        target_fprs: FPR targets for threshold computation
        is_binary: True if scores are binary (0/1), False for continuous
        
    Returns:
        Dictionary with message-level evaluation metrics, including
        interpretation guidance for binary methods
    """
    if is_binary:
        y_pred = np.asarray(scores, dtype=int)
        labels = np.asarray(labels, dtype=int)
        conf = metrics_mod.confusion_dict(labels, y_pred)
        ba = metrics_mod.balanced_accuracy_from_conf(conf)
        
        # Calculate additional useful metrics
        n_attack = int(np.sum(labels))
        n_benign = len(labels) - n_attack
        n_pred_attack = int(np.sum(y_pred))
        n_pred_benign = len(y_pred) - n_pred_attack
        
        return {
            "roc_pr": None,
            "roc_auc": None,
            "thresholds": {},
            "confusion_at_binary_pred": conf,
            "balanced_accuracy": ba,
            "note": "Binary method: predictions are 0/1 per message, no threshold tuning.",
            "interpretation": {
                "n_messages_total": len(labels),
                "n_messages_attack": n_attack,
                "n_messages_benign": n_benign,
                "n_predicted_attack": n_pred_attack,
                "n_predicted_benign": n_pred_benign,
                "explanation": "Binary methods output hard 0/1 predictions. Message-level metrics show raw prediction quality. Low precision typically means many benign messages flagged as anomalous (high false positive rate at message granularity)."
            }
        }
    return _evaluate_scores_generic(scores, labels, metrics_mod, target_fprs)



def _get_canonical_window_label(df_window: pd.DataFrame, window_id: str, log) -> int:
    """
    Compute CANONICAL window label using uniform "any attack" rule.
    
    Ensures ALL methods use the same window label for fair comparison.
    
    Rule: Window labeled "attack" (1) if ANY message has label==1, else "benign" (0).
    """
    if "label" not in df_window.columns:
        raise ValueError(f"Window {window_id}: missing 'label' column")
    if len(df_window) == 0:
        raise ValueError(f"Window {window_id}: empty window")
    label = int(df_window["label"].any())
    return label


def _normalize_attack_type(value, label: int) -> str:
    """Normalize attack_type values for consistent downstream metrics."""
    if value is None:
        return "benign" if label == 0 else "attack_unknown"
    try:
        if pd.isna(value):
            return "benign" if label == 0 else "attack_unknown"
    except Exception:
        pass

    if isinstance(value, str) and value.strip() == "":
        return "benign" if label == 0 else "attack_unknown"
    
    # If benign message has a non-null attack_type, force it to "benign"
    # This handles datasets where attack_type column is incorrectly populated
    if label == 0:
        return "benign"

    return str(value)



def evaluate_per_attack_type(
    scores,
    labels,
    attack_types,
    metrics_mod,
    target_fprs=(0.001, 0.01, 0.05),
    is_binary: bool = False,
    attack_types_normalized: bool = False,
):
    """
    Compute metrics separately for each attack_type while using all benign samples
    as the negative class. This reuses the standard message-level evaluation
    to keep thresholds and ROC logic consistent.
    
    Returns:
        Dictionary mapping attack_type -> metrics, or None if no attack_types provided.
        Includes comprehensive validation warnings in results.
    """
    log = logging.getLogger("unified_ids")
    
    if attack_types is None:
        return None

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    attack_types = np.asarray(attack_types, dtype=object)

    if scores.size == 0 or attack_types.size != scores.size:
        return None

    # Data quality check: warn if benign messages have non-null attack_types
    benign_mask = labels == 0
    benign_attack_types = attack_types[benign_mask]
    non_null_benign_attacks = sum(
        1 for at in benign_attack_types 
        if at is not None and not (isinstance(at, float) and pd.isna(at)) and str(at).strip() != ""
    )
    if non_null_benign_attacks > 0:
        log.warning(
            "Data quality issue: %d benign messages (%.1f%%) have non-null attack_type values. "
            "These will be normalized to 'benign' for per-attack-type evaluation.",
            non_null_benign_attacks,
            100.0 * non_null_benign_attacks / benign_mask.sum() if benign_mask.sum() > 0 else 0
        )

    if attack_types_normalized:
        norm_attack_types = list(attack_types)
    else:
        norm_attack_types = [
            _normalize_attack_type(at, int(lab)) for at, lab in zip(attack_types, labels)
        ]

    attack_values = sorted({atype for atype, lab in zip(norm_attack_types, labels) if lab == 1 and atype != "benign"})
    if not attack_values:
        log.warning("No valid attack types found in labels (all normalized to 'benign' or 'attack_unknown')")
        return {}

    results = {}
    total_benign = int(benign_mask.sum())

    for atype in attack_values:
        # pos_mask: messages with this attack_type (after normalization)
        pos_mask = np.asarray([a == atype for a in norm_attack_types], dtype=bool)
        
        # CRITICAL FIX: Count only attack messages (label=1) with this attack_type
        attack_mask = labels == 1
        n_attack_this_type = int((pos_mask & attack_mask).sum())
        
        if n_attack_this_type == 0:
            log.warning("Attack type '%s' has 0 attack messages after filtering - skipping", atype)
            continue
        
        # Validation: pos_mask should only be True for attack messages
        if (pos_mask & benign_mask).sum() > 0:
            log.warning(
                "Data inconsistency: %d benign messages have attack_type='%s' (should be impossible after normalization)",
                int((pos_mask & benign_mask).sum()),
                atype
            )
        
        # Create subset: this attack type + all benign
        subset_mask = pos_mask | benign_mask
        sub_scores = scores[subset_mask]
        sub_labels = np.where(pos_mask[subset_mask], 1, 0)
        
        # Validation: sub_labels should have both classes
        n_pos_subset = int(sub_labels.sum())
        n_neg_subset = int(len(sub_labels) - n_pos_subset)
        
        if n_pos_subset == 0 or n_neg_subset == 0:
            log.warning(
                "Attack type '%s': subset has only one class (pos=%d, neg=%d) - skipping",
                atype, n_pos_subset, n_neg_subset
            )
            continue
        
        # Validation: counts should match expectations
        if n_pos_subset != n_attack_this_type:
            log.error(
                "INTERNAL ERROR for attack type '%s': n_pos_subset=%d != n_attack_this_type=%d",
                atype, n_pos_subset, n_attack_this_type
            )

        try:
            eval_res = evaluate_scores_message(
                sub_scores,
                sub_labels,
                metrics_mod,
                target_fprs=target_fprs,
                is_binary=is_binary,
            )
        except Exception as e:
            lbl_vals, lbl_cnts = np.unique(sub_labels, return_counts=True)
            lbl_dist = {int(v): int(c) for v, c in zip(lbl_vals, lbl_cnts)}

            score_vals = np.unique(sub_scores)
            score_preview = score_vals[:10].tolist()

            raw_attack_preview = sorted({str(x) for x in attack_types[subset_mask]})[:10]

            log.error(
                "Per-attack eval failed for attack_type='%s': subset_n=%d, n_attack=%d, "
                "n_benign=%d, sub_label_dist=%s, score_unique=%d, score_preview=%s, "
                "raw_attack_values_preview=%s, is_binary=%s, error=%s",
                atype,
                int(subset_mask.sum()),
                n_attack_this_type,
                total_benign,
                lbl_dist,
                int(score_vals.size),
                score_preview,
                raw_attack_preview,
                bool(is_binary),
                e,
            )
            raise

        results[atype] = {
            "n_total": int(subset_mask.sum()),
            "n_attack": n_attack_this_type,  # FIXED: only count attack messages with this type
            "n_benign": total_benign,
            "note": "Metrics computed using attack messages of this type against all benign samples.",
            **eval_res,
        }

    return results


def save_roc_plot(
    method_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    best_f1_threshold: Optional[float] = None,
) -> None:
    """
    Save ROC curve with a marker at best-F1 threshold.
    
    Args:
        method_name: Name for plot title
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0=benign, 1=attack)
        out_path: Output file path for PNG
        best_f1_threshold: Pre-computed best F1 threshold (avoids expensive recomputation)
        
    Note:
        Downsamples ROC curve to max 1000 points for plotting efficiency.
        Skips plotting if labels contain only one class.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if np.unique(labels).size < 2:
        return

    fpr, tpr, thr = sk_roc_curve(labels, scores)
    rocauc = auc(fpr, tpr)

    # Downsample ROC curve for plotting if too large
    max_points = 1000
    n_points = len(fpr)
    if n_points > max_points:
        indices = np.linspace(0, n_points - 1, max_points, dtype=int)
        fpr_plot = fpr[indices]
        tpr_plot = tpr[indices]
    else:
        fpr_plot = fpr
        tpr_plot = tpr

    # Find best F1 point on ROC curve
    if best_f1_threshold is not None:
        # Use pre-computed threshold - find closest point on ROC curve
        best_idx = np.argmin(np.abs(thr - best_f1_threshold))
        best_f1 = -1.0  # Don't recompute
    else:
        # Fallback: compute best F1 on downsampled thresholds (to avoid O(n²) complexity)
        # For large datasets with millions of unique thresholds, checking all thresholds
        # takes prohibitively long. Instead, sample ~1000 equally-spaced thresholds.
        n_thr_to_check = min(1000, len(thr))
        if n_thr_to_check < len(thr):
            # Downsample thresholds
            thr_check_indices = np.linspace(0, len(thr) - 1, n_thr_to_check, dtype=int)
            thr_check = thr[thr_check_indices]
        else:
            thr_check_indices = np.arange(len(thr))
            thr_check = thr
        
        best_f1 = -1.0
        best_idx = 0
        for i, t in enumerate(thr_check):
            yhat = (scores >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                labels, yhat, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_idx = thr_check_indices[i]  # Map back to original threshold array index

    plt.figure()
    plt.plot(fpr_plot, tpr_plot, lw=2, label=f"ROC AUC = {rocauc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    if len(fpr) > 0:
        # Plot best F1 point if it's within the downsampled points
        plt.scatter([fpr[best_idx]], [tpr[best_idx]], marker="o", s=40, label="Best F1 point")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – {method_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------
# Core single-run logic
# --------------------------------------------------------------------


def _build_model(method: str, df_tr: pd.DataFrame, log) -> Any:
    """
    Construct an IDS model from the registry, with possible
    dataset-specific tweaks (e.g. CANet logging / column checks).
    
    Args:
        method: IDS method name (must be in METHOD_REGISTRY)
        df_tr: Training dataframe (used for method-specific checks)
        log: Logger instance
        
    Returns:
        Initialized IDS model instance (not yet fitted)
        
    Raises:
        ValueError: If method is not in METHOD_REGISTRY
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method {method}")

    factory = METHOD_REGISTRY[method]
    return factory()


def _run_single_on_loaded(
    method: str,
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    dataset_tag: str,
    out_dir_root: str,
    metrics_mod,
    paper_eval: bool,
    log,
    stage_cb=None,
) -> Dict[str, Any]:
    """
    Core logic of an experiment when df_tr / df_te are already in memory.
    """
    out_dir_path = prepare_out_dir(out_dir_root, method, dataset_tag)

    # ----------------------------------------------------------------
    # 1) Build and fit model
    # ----------------------------------------------------------------
    if stage_cb:
        stage_cb("fit")
    model = _build_model(method, df_tr, log)
    
    # Extract parameters/configuration from the model
    model_params = {}
    try:
        if hasattr(model, 'get_parameters'):
            model_params = model.get_parameters()
            log.info("Extracted model parameters: %s", list(model_params.keys()))
    except Exception as e:
        log.warning("Failed to extract model parameters: %s", e)

    log.info("Fitting %s", method)
    model.fit(df_tr)

    if hasattr(model, "_fit_history"):
        hist = getattr(model, "_fit_history")
        if isinstance(hist, (list, tuple)) and len(hist) > 0:
            log.info("fit history (first/last 10): %s ... %s", hist[:10], hist[-10:])
    
    # ----------------------------------------------------------------
    # 1b) Save trained model to disk (if supported)
    # ----------------------------------------------------------------
    if hasattr(model, 'save_model'):
        try:
            log.info("Saving trained model to disk...")
            model.save_model(str(out_dir_path), dataset_name=dataset_tag)
            log.info("Model saved successfully")
        except Exception as e:
            log.warning("Failed to save model: %s", e)

    # ----------------------------------------------------------------
    # 2) Scoring: windows
    # ----------------------------------------------------------------
    if stage_cb:
        stage_cb("score")
    caps = METHOD_CAPS.get(method)
    if caps is None:
        raise ValueError(f"METHOD_CAPS missing entry for method '{method}'")
    if "binary" not in caps:
        raise ValueError(f"METHOD_CAPS['{method}'] missing required 'binary' key")
    is_binary = bool(caps["binary"])
    win_scores: List[float] = []
    win_labels: List[int] = []
    win_attack_types: List[str] = []
    win_non_binary_seen = False
    n_win = n_win_attack = n_win_benign = 0

    if hasattr(model, "score_windows"):
        try:
            log.info("Scoring ALL windows on test set")
            for w, sc in model.score_windows(df_te):
                sc = float(sc)
                window_id = getattr(w, 'window_id', '?')
                
                # CRITICAL: Validate score is finite (not NaN/Inf)
                # Score range can be unbounded - ROC/threshold functions handle any range
                if np.isnan(sc) or np.isinf(sc):
                    raise AssertionError(
                        f"Method {method} window {window_id}: NaN/Inf score. "
                        "Methods MUST output valid finite scores."
                    )
                
                # Enforce binary outputs for declared-binary methods
                if is_binary and sc not in (0.0, 1.0):
                    raise AssertionError(
                        f"Binary method {method} window {window_id}: got {sc}, not in {{0.0, 1.0}}."
                    )
                if not is_binary and sc not in (0.0, 1.0):
                    win_non_binary_seen = True
                
                # *** USE CANONICAL LABEL, NOT METHOD'S LABEL ***
                # Ensures all methods fairly compared using uniform labeling
                sub = df_te.loc[w.idx_start : w.idx_end]
                lab_i = _get_canonical_window_label(sub, window_id, log)
                
                # *** USE CANONICAL LABEL, NOT METHOD'S LABEL ***
                # Ensures all methods fairly compared using uniform labeling
                sub = df_te.loc[w.idx_start : w.idx_end]
                lab_i = _get_canonical_window_label(sub, window_id, log)
                
                win_scores.append(sc)
                win_labels.append(lab_i)

                # Per-attack-type tracking using canonical window bounds
                if "attack_type" in df_te.columns:
                    atypes = sub[sub["label"] == 1]["attack_type"].dropna().unique()
                    if len(atypes) == 0:
                        # No attack_type in window -> normalize based on label
                        win_attack_types.append(_normalize_attack_type(None, lab_i))
                    elif len(atypes) == 1:
                        win_attack_types.append(_normalize_attack_type(atypes[0], lab_i))
                    else:
                        raise ValueError(
                            f"Window {window_id} has multiple attack types {list(atypes)}. "
                            f"This indicates a data quality issue."
                        )
                n_win += 1
                n_win_attack += lab_i
            n_win_benign = n_win - n_win_attack
            log.info(
                "Test windows: %d (benign: %d, attack: %d); stored samples: %d",
                n_win,
                n_win_benign,
                n_win_attack,
                len(win_scores),
            )
        except NotImplementedError:
            log.info("Model does not implement score_windows(); skipping window-level eval")
    else:
        log.info("Model has no score_windows(); skipping window-level eval")

    # ----------------------------------------------------------------
    # 3) Scoring: messages
    # ----------------------------------------------------------------
    msg_scores: List[float] = []
    msg_labels: List[int] = []
    msg_non_binary_seen = False
    msg_attack_types: List[Any] = []
    attack_type_lookup = None
    if "attack_type" in df_te.columns:
        attack_type_lookup = {str(idx): val for idx, val in df_te["attack_type"].items()}

    n_msg = n_msg_attack = n_msg_benign = 0

    if hasattr(model, "score_messages"):
        try:
            log.info("Model has score_messages(): performing message-level eval")
            for msg_id, sc, lab in model.score_messages(df_te):
                sc = float(sc)
                lab_i = int(lab)

                if lab_i not in (0, 1):
                    raise AssertionError(
                        f"Method {method} message {msg_id}: label={lab_i} is not binary {{0,1}}. "
                        "Unified IDS evaluation currently requires binary labels."
                    )
                
                # CRITICAL: Validate score is finite (not NaN/Inf)
                # Score range can be unbounded - ROC/threshold functions handle any range
                if np.isnan(sc) or np.isinf(sc):
                    raise AssertionError(
                        f"Method {method} message {msg_id}: NaN/Inf score. "
                        "Methods MUST output valid finite scores."
                    )
                
                # Enforce binary outputs for declared-binary methods
                if is_binary and sc not in (0.0, 1.0):
                    raise AssertionError(
                        f"Binary method {method} message {msg_id}: got {sc}, not in {{0.0, 1.0}}."
                    )
                if not is_binary and sc not in (0.0, 1.0):
                    msg_non_binary_seen = True
                
                msg_scores.append(sc)
                msg_labels.append(lab_i)
                if attack_type_lookup is not None:
                    msg_attack_types.append(
                        _normalize_attack_type(attack_type_lookup.get(str(msg_id)), lab_i)
                    )
                n_msg += 1
                n_msg_attack += lab_i
            n_msg_benign = n_msg - n_msg_attack
            log.info(
                "Test messages: %d (benign: %d, attack: %d); stored samples: %d",
                n_msg,
                n_msg_benign,
                n_msg_attack,
                len(msg_scores),
            )
        except NotImplementedError:
            log.info("Model's score_messages() not implemented; skipping message-level eval")
    else:
        log.info("Model has no score_messages(); skipping message-level eval")

    # Validate non-binary methods produced non-binary scores
    if not is_binary and n_win > 0 and len(win_scores) > 0 and not win_non_binary_seen:
        log.warning(
            "Non-binary method %s produced ONLY 0/1 window scores (no continuous values seen). "
            "Valid but suggests method may not utilize full capability.",
            method
        )
    if not is_binary and n_msg > 0 and len(msg_scores) > 0 and not msg_non_binary_seen:
        log.warning(
            "Non-binary method %s produced ONLY 0/1 message scores (no continuous values seen). "
            "Valid but suggests method may not utilize full capability.",
            method
        )

    # ----------------------------------------------------------------
    # 4) Meta
    # ----------------------------------------------------------------
    # caps / is_binary defined earlier before scoring
    meta: Dict[str, Any] = {
        "method": method,
        "dataset_tag": dataset_tag,
        "n_train_rows": int(len(df_tr)),
        "n_test_rows": int(len(df_te)),
        # NOTE: for now we keep simple hasattr() semantics here.
        "supports_message_eval": bool(hasattr(model, "score_messages")),
        "supports_window_eval": bool(hasattr(model, "score_windows")),
        "binary_method": bool(caps.get("binary", False)),
        "evaluation_strategy": "both_levels",  # Uniform evaluation: all methods at both window AND message levels
        "paper_dataset": caps.get("paper_dataset"),
        # Window configuration for transparency and fair comparison
        "window_config": caps.get("window_config"),
        "window_labeling_strategy": "any_attack",  # Canonical: window=1 if ANY message has label=1
        # Method parameters/configuration
        "method_parameters": model_params,
    }

    if n_win > 0:
        meta.update(
            {
                "n_test_windows": int(n_win),
                "n_test_windows_attack": int(n_win_attack),
                "n_test_windows_benign": int(n_win_benign),
                "score_unit_window": "window",
                "confusion_unit_window": "window",
                "label_semantics_window": {
                    "0": "benign window",
                    "1": "attack window",
                },
            }
        )

    if n_msg > 0:
        meta.update(
            {
                "n_test_messages": int(n_msg),
                "n_test_messages_attack": int(n_msg_attack),
                "n_test_messages_benign": int(n_msg_benign),
                "score_unit_message": "message",
                "confusion_unit_message": "message",
                "label_semantics_message": {
                    "0": "benign message",
                    "1": "attack message",
                },
            }
        )

    out: Dict[str, Any] = {"meta": meta}

    # ----------------------------------------------------------------
    # 5) Generic evaluation
    # ----------------------------------------------------------------
    if stage_cb:
        stage_cb("metrics")
    is_binary = bool(caps.get("binary", False))

    if n_win > 0 and len(win_scores) > 0:
        out["window_level"] = evaluate_scores_window(
            win_scores,
            win_labels,
            metrics_mod,
            is_binary=is_binary,
        )

        # Per-attack-type evaluation at window level (no re-windowing required)
        if win_attack_types and len(win_attack_types) == len(win_scores):
            try:
                per_attack = evaluate_per_attack_type(
                    scores=win_scores,
                    labels=win_labels,
                    attack_types=win_attack_types,
                    metrics_mod=metrics_mod,
                    is_binary=is_binary,
                    attack_types_normalized=True,
                )
                if per_attack:
                    out["window_level"]["per_attack_type"] = per_attack
            except Exception as e:
                log.warning(
                    "Failed to compute window-level per-attack-type metrics: %s",
                    e
                )

    if n_msg > 0 and len(msg_scores) > 0:
        out["message_level"] = evaluate_scores_message(
            msg_scores,
            msg_labels,
            metrics_mod,
            is_binary=is_binary,
        )

        if attack_type_lookup is not None:
            if len(msg_attack_types) == len(msg_scores):
                per_attack = evaluate_per_attack_type(
                    scores=msg_scores,
                    labels=msg_labels,
                    attack_types=msg_attack_types,
                    metrics_mod=metrics_mod,
                    is_binary=is_binary,
                    attack_types_normalized=True,
                )
                if per_attack:
                    out["message_level"]["per_attack_type"] = per_attack
                    
                    # Validate consistency between overall and per-attack metrics
                    _validate_per_attack_consistency(
                        per_attack, n_msg_attack, n_msg_benign, log
                    )
            else:
                log.warning(
                    "Attack type lookup size mismatch: attacks=%d scores=%d; skipping per-attack metrics",
                    len(msg_attack_types),
                    len(msg_scores),
                )

    # ----------------------------------------------------------------
    # 6) Optional per-method paper_eval hook
    # ----------------------------------------------------------------
    if paper_eval and hasattr(model, "paper_eval"):
        try:
            log.info("Calling model.paper_eval(.) for method %s", method)
            pe = model.paper_eval(df_tr, df_te, str(out_dir_path))
            if pe is not None:
                out.setdefault("paper_eval", {})
                if isinstance(pe, dict):
                    out["paper_eval"].update(pe)
                else:
                    out["paper_eval"][f"{method}_paper_eval"] = pe
        except NotImplementedError:
            log.info("model.paper_eval is not implemented for %s", method)
        except Exception as e:
            log.warning("model.paper_eval(%s) failed: %s", method, e)

    # ----------------------------------------------------------------
    # 7) Save to disk
    # ----------------------------------------------------------------
    if stage_cb:
        stage_cb("save")
    metrics_path = out_dir_path / "metrics.json"
    roc_win_path = out_dir_path / "roc_window.png"
    roc_msg_path = out_dir_path / "roc_message.png"

    # Extract best F1 thresholds for efficient ROC plotting
    win_best_f1_thr = None
    msg_best_f1_thr = None
    if "window_level" in out:
        win_best_f1_thr = out["window_level"].get("best_f1_threshold")
    if "message_level" in out:
        msg_best_f1_thr = out["message_level"].get("best_f1_threshold")

    log.info("Saving ROC plots...")
    if not is_binary:
        # Only generate ROC plots for scoring methods (not binary 0/1 methods)
        if len(win_scores) > 0 and len(np.unique(win_labels)) > 1:
            save_roc_plot(f"{method}_window", win_scores, win_labels, roc_win_path, win_best_f1_thr)
        if len(msg_scores) > 0 and len(np.unique(msg_labels)) > 1:
            save_roc_plot(f"{method}_message", msg_scores, msg_labels, roc_msg_path, msg_best_f1_thr)
        log.info("ROC plots saved")
    else:
        log.info("ROC plots skipped for binary method (no threshold tuning available)")

    log.info("Saving metrics to %s", metrics_path)
    with metrics_path.open("w") as f:
        json.dump(out, f, indent=2)

    log.info("[%s] %s: evaluation complete - %s", dataset_tag, method, metrics_path)
    print(f"[{dataset_tag}] {method}: saved {metrics_path}")
    return out


def run_single(
    method: str,
    train_glob: str,
    test_glob: str,
    out_dir: str,
    paper_eval: bool,
    log_level: str = "INFO",
    dataset_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public entrypoint for running one (method, dataset) experiment.

    Args:
        method: IDS method name
        train_glob: Glob pattern for benign-only training files
        test_glob: Glob pattern for test files
        out_dir: Output directory for results
        paper_eval: Whether to run paper-specific evaluation blocks
        log_level: Logging level (default INFO)
        dataset_tag: Optional tag for dataset (inferred if not provided)
    """
    from unified_ids.utils.logging import setup_logging
    from unified_ids.eval import metrics as metrics_mod

    log = setup_logging(log_level)

    df_tr, df_te = load_train_test(
        train_glob=train_glob,
        test_glob=test_glob,
        log=log,
    )
    if dataset_tag is None:
        dataset_tag = infer_dataset_tag_from_df(df_te)
        if dataset_tag == "unknown":
            dataset_tag = _infer_dataset_tag_from_glob(test_glob)
            log.info("Inferred dataset tag from glob: %s", dataset_tag)

    # After we know dataset_tag, attach a per-run file log in results dir
    # This keeps console logging as-is and adds a file handler.
    out_dir_path = prepare_out_dir(out_dir, method, dataset_tag)
    log = setup_logging(log_level, log_file=out_dir_path / "evaluation.log")

    return _run_single_on_loaded(
        method=method,
        df_tr=df_tr,
        df_te=df_te,
        dataset_tag=dataset_tag,
        out_dir_root=out_dir,
        metrics_mod=metrics_mod,
        paper_eval=paper_eval,
        log=log,
    )

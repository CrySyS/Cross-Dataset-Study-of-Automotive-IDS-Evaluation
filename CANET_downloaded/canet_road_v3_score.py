#!/usr/bin/env python3
"""Score ROAD v3 CANet outputs with explicit handling for one-class files."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = PROJECT_DIR.parent / "Results"


def evaluation(real_labels: np.ndarray, predicted_labels: np.ndarray):
    if len(real_labels) != len(predicted_labels):
        raise ValueError("Inputs must have same length.")

    cm_data = metrics.confusion_matrix(real_labels, predicted_labels, labels=[0, 1])
    tn, fp, fn, tp = cm_data.ravel()

    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    if (tp + fn) == 0:
        tpr = None
        f1 = None
    else:
        tpr = tp / (tp + fn)
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "TPR": tpr,
        "TNR": tnr,
        "F1": f1,
    }


def safe_mean(values):
    keep = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(keep)) if keep else None


def binarize_session(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        vals = numeric.astype(int).to_numpy()
        if set(np.unique(vals).tolist()).issubset({0, 1}):
            return vals
    s = series.astype(str).str.strip().str.lower()
    return np.where(s.isin({"0", "normal", "benign"}), 0, 1).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ROAD v3 CANet parquet outputs")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory with ROAD v3 CANet parquet outputs")
    parser.add_argument("--experiment-id", required=True, help="Timestamp used in RoadV3_CANet result filenames")
    parser.add_argument("--quantile", type=float, default=0.999, help="Quantile for threshold from valid set")
    parser.add_argument("--one-class-policy", choices=["include", "skip"], default="include", help="If include, keep one-class files in per-file table (AUC/TPR may be N/A); if skip, omit them entirely")
    parser.add_argument("--out-csv", default="", help="Optional output CSV path")
    parser.add_argument("--out-json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    valid_file = results_dir / f"RoadV3_CANet_{args.experiment_id}_valid.parquet"
    if not valid_file.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_file}")

    valid_df = pd.read_parquet(valid_file)
    if "MSE" not in valid_df.columns:
        raise KeyError(f"Column 'MSE' not found in {valid_file}")

    threshold = float(np.quantile(valid_df["MSE"].to_numpy(), args.quantile))
    print(f"[CANet ROAD v3] Quantile={args.quantile}, Threshold={threshold}")

    all_files = sorted(results_dir.glob(f"RoadV3_CANet_{args.experiment_id}_*.parquet"))
    attack_files = [p for p in all_files if p.name != valid_file.name]
    if not attack_files:
        raise RuntimeError("No ROAD v3 attack parquet files found to score.")

    rows = []
    skipped_one_class = []
    for in_file in attack_files:
        attack = in_file.stem.replace(f"RoadV3_CANet_{args.experiment_id}_", "")
        df = pd.read_parquet(in_file)
        for col in ["MSE", "Session"]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in {in_file}")

        y_true = binarize_session(df["Session"])
        classes = np.unique(y_true)
        one_class = len(classes) < 2

        if one_class and args.one_class_policy == "skip":
            skipped_one_class.append(attack)
            continue

        y_pred = (df["MSE"].to_numpy() >= threshold).astype(int)
        metric_row = evaluation(y_true, y_pred)

        try:
            auc = float(metrics.roc_auc_score(y_true, df["MSE"].to_numpy())) if not one_class else None
        except Exception:
            auc = None

        metric_row.update(
            {
                "attack": attack,
                "n_rows": int(len(df)),
                "threshold": threshold,
                "AUC": auc,
                "one_class_labels": bool(one_class),
            }
        )
        rows.append(metric_row)

        tpr_str = "N/A" if metric_row["TPR"] is None else f"{metric_row['TPR']:.4f}"
        tnr_str = "N/A" if metric_row["TNR"] is None else f"{metric_row['TNR']:.4f}"
        f1_str = "N/A" if metric_row["F1"] is None else f"{metric_row['F1']:.4f}"
        auc_str = "N/A" if auc is None else f"{auc:.4f}"
        flag = " [one-class]" if one_class else ""
        print(
            f"attack={attack}{flag} n={metric_row['n_rows']} "
            f"TPR={tpr_str} TNR={tnr_str} F1={f1_str} AUC={auc_str} "
            f"(tn={metric_row['tn']} fp={metric_row['fp']} fn={metric_row['fn']} tp={metric_row['tp']})"
        )

    if not rows:
        raise RuntimeError("No files left to score after applying one-class policy.")

    df_metrics = pd.DataFrame(rows)
    comparable = df_metrics[~df_metrics["one_class_labels"]].copy()

    avg_all = {
        "avg_TPR": safe_mean(df_metrics["TPR"].tolist()),
        "avg_TNR": safe_mean(df_metrics["TNR"].tolist()),
        "avg_F1": safe_mean(df_metrics["F1"].tolist()),
        "avg_AUC": safe_mean(df_metrics["AUC"].tolist()),
    }
    avg_comparable = {
        "avg_TPR": safe_mean(comparable["TPR"].tolist()) if len(comparable) else None,
        "avg_TNR": safe_mean(comparable["TNR"].tolist()) if len(comparable) else None,
        "avg_F1": safe_mean(comparable["F1"].tolist()) if len(comparable) else None,
        "avg_AUC": safe_mean(comparable["AUC"].tolist()) if len(comparable) else None,
    }

    print(
        "Summary(all): "
        f"avg_TPR={avg_all['avg_TPR'] if avg_all['avg_TPR'] is not None else 'N/A'} "
        f"avg_TNR={avg_all['avg_TNR'] if avg_all['avg_TNR'] is not None else 'N/A'} "
        f"avg_F1={avg_all['avg_F1'] if avg_all['avg_F1'] is not None else 'N/A'} "
        f"avg_AUC={avg_all['avg_AUC'] if avg_all['avg_AUC'] is not None else 'N/A'}"
    )
    print(
        "Summary(comparable two-class only): "
        f"avg_TPR={avg_comparable['avg_TPR'] if avg_comparable['avg_TPR'] is not None else 'N/A'} "
        f"avg_TNR={avg_comparable['avg_TNR'] if avg_comparable['avg_TNR'] is not None else 'N/A'} "
        f"avg_F1={avg_comparable['avg_F1'] if avg_comparable['avg_F1'] is not None else 'N/A'} "
        f"avg_AUC={avg_comparable['avg_AUC'] if avg_comparable['avg_AUC'] is not None else 'N/A'}"
    )

    out_csv = Path(args.out_csv) if args.out_csv else results_dir / f"RoadV3_CANet_{args.experiment_id}_metrics.csv"
    out_json = Path(args.out_json) if args.out_json else results_dir / f"RoadV3_CANet_{args.experiment_id}_metrics_summary.json"

    df_metrics.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": args.experiment_id,
                "quantile": args.quantile,
                "threshold": threshold,
                "one_class_policy": args.one_class_policy,
                "averages_all": avg_all,
                "averages_two_class_only": avg_comparable,
                "n_scored_files": int(len(df_metrics)),
                "n_two_class_files": int(len(comparable)),
                "n_one_class_files": int(len(df_metrics) - len(comparable)),
                "skipped_one_class_files": skipped_one_class,
            },
            f,
            indent=2,
        )

    print(f"Saved per-attack metrics: {out_csv}")
    print(f"Saved summary metrics: {out_json}")


if __name__ == "__main__":
    main()

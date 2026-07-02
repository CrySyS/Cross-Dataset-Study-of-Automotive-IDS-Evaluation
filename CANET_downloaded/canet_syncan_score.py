#!/usr/bin/env python3
"""Score SynCAN CANet outputs with quantile thresholding (from Plotting results.ipynb)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics


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
    if not keep:
        return None
    return float(np.mean(keep))


def binarize_session(series: pd.Series, attack: str) -> np.ndarray:
    # Handle numeric binary labels directly.
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        vals = numeric.astype(int).to_numpy()
        uniq = set(np.unique(vals).tolist())
        if len(uniq) == 1:
            return np.zeros(len(vals), dtype=int) if attack == "normal" else np.ones(len(vals), dtype=int)
        if uniq.issubset({0, 1}):
            return vals

    # Handle string labels (e.g., Normal, flooding, etc.).
    s = series.astype(str).str.strip().str.lower()
    normal_tokens = {"0", "normal", "benign"}
    return np.where(s.isin(normal_tokens), 0, 1).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SynCAN CANet parquet outputs")
    parser.add_argument("--results-dir", default="../../Results", help="Directory with CANet parquet outputs")
    parser.add_argument("--experiment-id", required=True, help="Timestamp used in result filenames")
    parser.add_argument("--quantile", type=float, default=0.999, help="Quantile for threshold from valid set")
    parser.add_argument(
        "--attacks",
        nargs="*",
        default=["normal", "flooding", "suppress", "plateau", "continuous", "playback"],
        help="Attack tags to score",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional output CSV path (default: <results-dir>/Syncan_CANet_<exp>_metrics.csv)",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional output JSON path (default: <results-dir>/Syncan_CANet_<exp>_metrics_summary.json)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    valid_file = results_dir / f"Syncan_CANet_{args.experiment_id}_valid.parquet"
    if not valid_file.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_file}")

    valid_df = pd.read_parquet(valid_file)
    if "MSE" not in valid_df.columns:
        raise KeyError(f"Column 'MSE' not found in {valid_file}")

    threshold = float(np.quantile(valid_df["MSE"].to_numpy(), args.quantile))
    print(f"[CANet] Quantile={args.quantile}, Threshold={threshold}")

    rows = []
    for attack in args.attacks:
        in_file = results_dir / f"Syncan_CANet_{args.experiment_id}_{attack}.parquet"
        if not in_file.exists():
            print(f"[WARN] Missing file, skipping: {in_file}")
            continue

        df = pd.read_parquet(in_file)
        for col in ["MSE", "Session"]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in {in_file}")

        y_true = binarize_session(df["Session"], attack=attack)
        y_pred = (df["MSE"].to_numpy() >= threshold).astype(int)

        metric_row = evaluation(y_true, y_pred)
        try:
            if len(np.unique(y_true)) < 2:
                auc = None
            else:
                auc = float(metrics.roc_auc_score(y_true, df["MSE"].to_numpy()))
        except Exception:
            auc = None

        metric_row.update(
            {
                "attack": attack,
                "n_rows": int(len(df)),
                "threshold": threshold,
                "AUC": auc,
            }
        )
        rows.append(metric_row)

        tpr_str = "N/A" if metric_row["TPR"] is None else f"{metric_row['TPR']:.4f}"
        tnr_str = "N/A" if metric_row["TNR"] is None else f"{metric_row['TNR']:.4f}"
        f1_str = "N/A" if metric_row["F1"] is None else f"{metric_row['F1']:.4f}"
        auc_str = "N/A" if auc is None else f"{auc:.4f}"
        print(
            f"attack={attack} n={metric_row['n_rows']} "
            f"TPR={tpr_str} TNR={tnr_str} F1={f1_str} AUC={auc_str} "
            f"(tn={metric_row['tn']} fp={metric_row['fp']} fn={metric_row['fn']} tp={metric_row['tp']})"
        )

    if not rows:
        raise RuntimeError("No attack files were scored.")

    df_metrics = pd.DataFrame(rows)
    avg = {
        "avg_TPR": safe_mean(df_metrics["TPR"].tolist()),
        "avg_TNR": safe_mean(df_metrics["TNR"].tolist()),
        "avg_F1": safe_mean(df_metrics["F1"].tolist()),
        "avg_AUC": safe_mean(df_metrics["AUC"].tolist()),
    }

    print(
        "Summary: "
        f"avg_TPR={avg['avg_TPR'] if avg['avg_TPR'] is not None else 'N/A'} "
        f"avg_TNR={avg['avg_TNR'] if avg['avg_TNR'] is not None else 'N/A'} "
        f"avg_F1={avg['avg_F1'] if avg['avg_F1'] is not None else 'N/A'} "
        f"avg_AUC={avg['avg_AUC'] if avg['avg_AUC'] is not None else 'N/A'}"
    )

    out_csv = Path(args.out_csv) if args.out_csv else results_dir / f"Syncan_CANet_{args.experiment_id}_metrics.csv"
    out_json = Path(args.out_json) if args.out_json else results_dir / f"Syncan_CANet_{args.experiment_id}_metrics_summary.json"

    df_metrics.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": args.experiment_id,
                "quantile": args.quantile,
                "threshold": threshold,
                "averages": avg,
                "n_scored_files": int(len(df_metrics)),
            },
            f,
            indent=2,
        )

    print(f"Saved per-attack metrics: {out_csv}")
    print(f"Saved summary metrics: {out_json}")


if __name__ == "__main__":
    main()

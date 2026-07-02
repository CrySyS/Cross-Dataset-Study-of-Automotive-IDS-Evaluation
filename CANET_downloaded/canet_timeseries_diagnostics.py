#!/usr/bin/env python3
"""Create per-file CANet time-series diagnostics from parquet outputs.

Each plot contains:
- MSE over time with threshold line
- shaded attack windows (Session == 1)
- shaded detection windows (MSE >= threshold)
- binary tracks for ground truth and detections
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def contiguous_intervals(time_values: np.ndarray, binary_values: np.ndarray):
    """Return contiguous [start, end] intervals where binary_values == 1."""
    if len(time_values) == 0:
        return []
    b = np.asarray(binary_values, dtype=int)
    t = np.asarray(time_values)
    starts = np.where((b == 1) & (np.r_[0, b[:-1]] == 0))[0]
    ends = np.where((b == 1) & (np.r_[b[1:], 0] == 0))[0]
    intervals = []
    for s, e in zip(starts, ends):
        intervals.append((float(t[s]), float(t[e])))
    return intervals


def maybe_downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CANet MSE/threshold time-series diagnostics")
    parser.add_argument("--results-dir", default="../../Results", help="Directory with parquet outputs")
    parser.add_argument("--prefix", required=True, help="Result file prefix: Road_CANet or Syncan_CANet")
    parser.add_argument("--experiment-id", required=True, help="Experiment timestamp in result filenames")
    parser.add_argument("--quantile", type=float, default=0.999, help="Threshold quantile from valid file")
    parser.add_argument("--out-dir", default="", help="Output directory for plots")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of attack files (0 = all)")
    parser.add_argument("--max-points", type=int, default=60000, help="Downsample per plot to this many points")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    default_out = results_dir / f"{args.prefix}_{args.experiment_id}_timeseries_q{str(args.quantile).replace('.', '')}"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_path = results_dir / f"{args.prefix}_{args.experiment_id}_valid.parquet"
    if not valid_path.exists():
        raise FileNotFoundError(f"Missing valid parquet: {valid_path}")

    valid_df = pd.read_parquet(valid_path)
    if "MSE" not in valid_df.columns:
        raise KeyError(f"Column 'MSE' not found in {valid_path}")
    threshold = float(np.quantile(valid_df["MSE"].to_numpy(), args.quantile))

    files = sorted(results_dir.glob(f"{args.prefix}_{args.experiment_id}_*.parquet"))
    attack_files = [p for p in files if p.name != valid_path.name]
    if args.max_files > 0:
        attack_files = attack_files[: args.max_files]

    rows = []
    for p in attack_files:
        attack = p.stem.replace(f"{args.prefix}_{args.experiment_id}_", "")
        df = pd.read_parquet(p)
        required = {"Time", "MSE", "Session"}
        missing = sorted(required.difference(df.columns))
        if missing:
            print(f"[WARN] Skip {p.name}: missing columns {missing}")
            continue

        df = df[["Time", "MSE", "Session"]].copy()
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df["MSE"] = pd.to_numeric(df["MSE"], errors="coerce")
        df["Session"] = pd.to_numeric(df["Session"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["Time", "MSE"]).sort_values("Time")
        if df.empty:
            continue

        df["Detect"] = (df["MSE"] >= threshold).astype(int)

        n = len(df)
        n_pos = int((df["Session"] == 1).sum())
        n_det = int((df["Detect"] == 1).sum())
        tp = int(((df["Session"] == 1) & (df["Detect"] == 1)).sum())
        fp = int(((df["Session"] == 0) & (df["Detect"] == 1)).sum())
        fn = int(((df["Session"] == 1) & (df["Detect"] == 0)).sum())

        plot_df = maybe_downsample(df, max_points=args.max_points)

        t = plot_df["Time"].to_numpy()
        mse = plot_df["MSE"].to_numpy()
        y_true = plot_df["Session"].to_numpy().astype(int)
        y_pred = plot_df["Detect"].to_numpy().astype(int)

        attack_intervals = contiguous_intervals(t, y_true)
        detect_intervals = contiguous_intervals(t, y_pred)

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        ax1 = axes[0]
        ax1.plot(t, mse, color="#1f77b4", linewidth=0.8, label="MSE")
        ax1.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.2, label=f"threshold={threshold:.6g}")

        for s, e in attack_intervals:
            ax1.axvspan(s, e, color="#d62728", alpha=0.12)
        for s, e in detect_intervals:
            ax1.axvspan(s, e, color="#2ca02c", alpha=0.10)

        ax1.set_ylabel("MSE")
        ax1.set_title(f"{attack} | n={n:,} pos={n_pos:,} det={n_det:,} tp={tp:,} fp={fp:,} fn={fn:,}")
        ax1.legend(loc="upper right", fontsize=8)

        ax2 = axes[1]
        ax2.step(t, y_true, where="post", color="#d62728", linewidth=1.1, label="Attack (Session)")
        ax2.step(t, y_pred, where="post", color="#2ca02c", linewidth=1.1, label="Detection")
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_xlabel("Time")
        ax2.set_ylabel("0/1")
        ax2.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"timeseries_{attack}.png", dpi=150)
        plt.close(fig)

        rows.append(
            {
                "attack": attack,
                "n_rows": n,
                "n_attack": n_pos,
                "n_detect": n_det,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "threshold": threshold,
                "quantile": args.quantile,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "timeseries_detection_summary.csv", index=False)
    print(f"Threshold ({args.quantile}) = {threshold}")
    print(f"Saved plots and summary to: {out_dir}")


if __name__ == "__main__":
    main()

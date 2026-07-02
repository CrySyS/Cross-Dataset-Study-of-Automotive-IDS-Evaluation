#!/usr/bin/env python3
"""
Window-size count analysis (no model training).

Generates a plot similar to legacy "*_window_sizes.png" outputs, showing:
- Number of attack windows (log y)
- Number of benign windows (log y)
- Average messages per window (log y)

All window labels use the canonical "any attack -> attack window" rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unified_ids.dataio.loaders import read_parquet_glob
from unified_ids.dataio.windowing import windows_fixed_time


def _fmt_num(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def analyze_window_counts(df_test: pd.DataFrame, window_size: float) -> dict:
    windows = list(windows_fixed_time(df_test, span_seconds=window_size, stride_seconds=window_size))
    if not windows:
        return {
            "window_size": float(window_size),
            "n_windows": 0,
            "n_attack": 0,
            "n_benign": 0,
            "avg_msgs_per_window": 0.0,
        }

    labels = np.array([int(w.label_window) for w in windows], dtype=int)
    msg_counts = np.array([(int(w.idx_end) - int(w.idx_start) + 1) for w in windows], dtype=float)

    return {
        "window_size": float(window_size),
        "n_windows": int(len(windows)),
        "n_attack": int((labels == 1).sum()),
        "n_benign": int((labels == 0).sum()),
        "avg_msgs_per_window": float(msg_counts.mean()),
    }


def plot_window_counts(df: pd.DataFrame, dataset_name: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Number of Attack and Benign Windows vs. Window Size", fontsize=12)

    x = df["window_size"].values
    y_attack = df["n_attack"].values
    y_benign = df["n_benign"].values
    y_avg = df["avg_msgs_per_window"].values

    ax.plot(x, y_attack, "o-", color="#1f77b4", label="Attack Windows")
    ax.plot(x, y_benign, "o-", color="#ff7f0e", label="Benign Windows")
    ax.plot(x, y_avg, "o--", color="#2ca02c", label="Avg Messages per Window")

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_fmt_num(float(v)) for v in x])
    ax.set_yscale("log")
    ax.set_xlabel("Window Size (seconds, values shown; log-spaced axis)")
    ax.set_ylabel("Number of Windows (log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    # Numeric point labels, matching the style of the provided example.
    for xv, yv in zip(x, y_attack):
        if yv > 0:
            ax.annotate(_fmt_num(float(yv)), (xv, yv), color="#0000ff", fontsize=12, ha="center", xytext=(0, 8), textcoords="offset points")
    for xv, yv in zip(x, y_benign):
        if yv > 0:
            ax.annotate(_fmt_num(float(yv)), (xv, yv), color="#ff9900", fontsize=12, ha="center", xytext=(0, 0), textcoords="offset points")
    for xv, yv in zip(x, y_avg):
        if yv > 0:
            ax.annotate(_fmt_num(float(yv)), (xv, yv), color="#008800", fontsize=12, ha="center", xytext=(0, 8), textcoords="offset points")

    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Window count analysis by window size (no training)")
    parser.add_argument("--dataset_name", required=True, help="Name used in plot title and output file names")
    parser.add_argument("--test_glob", required=True, help="Glob pattern for test parquet files")
    parser.add_argument(
        "--window_sizes",
        type=str,
        default="0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10",
        help="Comma-separated window sizes in seconds",
    )
    parser.add_argument(
        "--output_dir",
        default="window_size_hyperparameter_module",
        help="Output directory for PNG/SVG/PDF/CSV/JSON",
    )
    args = parser.parse_args()

    window_sizes = [float(x.strip()) for x in args.window_sizes.split(",") if x.strip()]

    print("=" * 80)
    print("WINDOW SIZE COUNT ANALYSIS (NO TRAINING)")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Test glob: {args.test_glob}")
    print(f"Window sizes: {window_sizes}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    df_test = read_parquet_glob(args.test_glob)
    print(f"Loaded test rows: {len(df_test):,}")

    rows = []
    for ws in window_sizes:
        print(f"Analyzing window size: {ws}s")
        rows.append(analyze_window_counts(df_test, ws))

    df_results = pd.DataFrame(rows).sort_values("window_size").reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{args.dataset_name}_window_sizes_counts.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved counts CSV: {csv_path}")

    fig = plot_window_counts(df_results, args.dataset_name)
    png_path = out_dir / f"{args.dataset_name}_window_sizes.png"
    svg_path = out_dir / f"{args.dataset_name}_window_sizes.svg"
    pdf_path = out_dir / f"{args.dataset_name}_window_sizes.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {svg_path}")
    print(f"Saved plot: {pdf_path}")

    summary = {
        "dataset": args.dataset_name,
        "window_sizes": window_sizes,
        "total_rows_test": int(len(df_test)),
        "results": df_results.to_dict(orient="records"),
    }
    json_path = out_dir / f"{args.dataset_name}_window_sizes_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Standalone CrySyS training-signal overview plotter.

Purpose:
- Read all training CSV files from a directory.
- Extract one signal column (default: Sig_7_of_ID_897).
- Downsample values to keep plotting lightweight.
- Plot where the signal typically lives across files.
- Optional raw mode (no normalization).

Outputs:
- artifacts/signal_overview/<signal>_train_overview.png
- artifacts/signal_overview/<signal>_train_overview_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _downsample_array(values: np.ndarray, sample_every: int, max_points_per_file: int) -> np.ndarray:
    """Downsample with stride first, then cap to max points for stability."""
    if values.size == 0:
        return values

    step = max(1, int(sample_every))
    sampled = values[::step]
    if sampled.size <= max_points_per_file:
        return sampled

    idx = np.linspace(0, sampled.size - 1, num=max_points_per_file, dtype=int)
    return sampled[idx]


def _load_minmax_for_signal(scaler_csv: Path, signal: str) -> tuple[float, float] | None:
    """Load MinMax bounds matching CANShield scaler convention."""
    if not scaler_csv.exists():
        return None

    df = pd.read_csv(scaler_csv, index_col=0)
    if signal not in df.columns:
        return None

    vals = pd.to_numeric(df[signal], errors="coerce").dropna().values.astype(np.float64)
    if vals.size == 0:
        return None
    return float(np.min(vals)), float(np.max(vals))


def _scale(values: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    den = max(max_v - min_v, 1e-12)
    return (values - min_v) / den


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one signal across all CrySyS train files.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("../data/crysys_extracted/ambient"),
        help="Directory containing training CSV files.",
    )
    parser.add_argument(
        "--signal",
        type=str,
        default="Sig_7_of_ID_897",
        help="Signal column to analyze.",
    )
    parser.add_argument(
        "--scaler-csv",
        type=Path,
        default=Path("../scaler/min_max_values_crysys.csv"),
        help="Scaler CSV used to compute scaled values (optional).",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable MinMax normalization and plot raw signal values.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Keep every Nth non-NaN signal sample in each file.",
    )
    parser.add_argument(
        "--max-points-per-file",
        type=int,
        default=5000,
        help="Hard cap of plotted points per file after stride sampling.",
    )
    parser.add_argument(
        "--global-max-points",
        type=int,
        default=120000,
        help="Global cap for combined histogram points.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../artifacts/signal_overview"),
        help="Output directory for plot and summary.",
    )
    args = parser.parse_args()

    train_dir = args.train_dir.resolve()
    scaler_csv = args.scaler_csv.resolve()
    output_dir = args.output_dir.resolve()

    csv_paths = sorted(train_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {train_dir}")

    per_file_stats = []
    per_file_samples_raw = []
    file_names = []

    for csv_path in csv_paths:
        file_name = csv_path.stem
        try:
            df = pd.read_csv(csv_path, usecols=[args.signal])
        except ValueError:
            # Missing signal in this file; skip explicitly.
            continue

        signal_series = pd.to_numeric(df[args.signal], errors="coerce").dropna()
        raw_values = signal_series.to_numpy(dtype=np.float64)
        if raw_values.size == 0:
            continue

        sampled_raw = _downsample_array(
            raw_values,
            sample_every=args.sample_every,
            max_points_per_file=args.max_points_per_file,
        )

        file_names.append(file_name)
        per_file_samples_raw.append(sampled_raw)
        per_file_stats.append(
            {
                "file": file_name,
                "rows_non_na": int(raw_values.size),
                "rows_sampled": int(sampled_raw.size),
                "raw_mean": float(np.mean(raw_values)),
                "raw_std": float(np.std(raw_values)),
                "raw_q10": float(np.quantile(raw_values, 0.10)),
                "raw_q50": float(np.quantile(raw_values, 0.50)),
                "raw_q90": float(np.quantile(raw_values, 0.90)),
            }
        )

    if not per_file_samples_raw:
        raise ValueError(
            f"Signal '{args.signal}' was not found with usable values in {train_dir}"
        )

    scaler_bounds = None if args.no_scale else _load_minmax_for_signal(scaler_csv, args.signal)
    has_scaler = scaler_bounds is not None

    if has_scaler:
        min_v, max_v = scaler_bounds
        per_file_samples_plot = [_scale(v, min_v, max_v) for v in per_file_samples_raw]
        y_label = f"{args.signal} (scaled)"
    else:
        per_file_samples_plot = per_file_samples_raw
        y_label = f"{args.signal} (raw)"

    all_points = np.concatenate(per_file_samples_plot)
    if all_points.size > args.global_max_points:
        idx = np.linspace(0, all_points.size - 1, num=args.global_max_points, dtype=int)
        all_points = all_points[idx]

    x_idx = np.arange(len(file_names))
    q10 = np.array([np.quantile(v, 0.10) for v in per_file_samples_plot])
    q50 = np.array([np.quantile(v, 0.50) for v in per_file_samples_plot])
    q90 = np.array([np.quantile(v, 0.90) for v in per_file_samples_plot])

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), constrained_layout=True)

    # Panel 1: file-level quantile bands
    ax0 = axes[0]
    ax0.fill_between(x_idx, q10, q90, alpha=0.25, color="tab:blue", label="q10-q90 band")
    ax0.plot(x_idx, q50, color="tab:orange", linewidth=1.6, label="median")
    ax0.scatter(x_idx, q50, s=16, color="tab:orange", alpha=0.85)
    ax0.set_title(f"Per-file distribution of {args.signal} in train data")
    ax0.set_ylabel(y_label)
    ax0.set_xticks(x_idx)
    ax0.set_xticklabels(file_names, rotation=75, ha="right", fontsize=8)
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best")

    # Panel 2: global histogram of sampled points
    ax1 = axes[1]
    ax1.hist(all_points, bins=120, color="tab:green", alpha=0.8)
    ax1.set_title("Global sampled distribution across all train files")
    ax1.set_xlabel(y_label)
    ax1.set_ylabel("Count")
    ax1.grid(alpha=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_signal = args.signal.replace("/", "_").replace(" ", "_")
    mode_suffix = "scaled" if has_scaler else "raw"
    out_plot = output_dir / f"{safe_signal}_train_overview_{mode_suffix}.png"
    out_summary = output_dir / f"{safe_signal}_train_overview_{mode_suffix}_summary.json"
    fig.savefig(out_plot, dpi=140, bbox_inches="tight")
    plt.close(fig)

    global_summary = {
        "signal": args.signal,
        "value_mode": mode_suffix,
        "train_dir": str(train_dir),
        "num_files_used": len(file_names),
        "sample_every": int(args.sample_every),
        "max_points_per_file": int(args.max_points_per_file),
        "global_max_points": int(args.global_max_points),
        "has_scaler": bool(has_scaler),
        "scaler_csv": str(scaler_csv),
        "scaler_min": None if not has_scaler else float(scaler_bounds[0]),
        "scaler_max": None if not has_scaler else float(scaler_bounds[1]),
        "global_plot_points": int(all_points.size),
        "global_plot_mean": float(np.mean(all_points)),
        "global_plot_std": float(np.std(all_points)),
        "global_plot_q10": float(np.quantile(all_points, 0.10)),
        "global_plot_q50": float(np.quantile(all_points, 0.50)),
        "global_plot_q90": float(np.quantile(all_points, 0.90)),
        "files": per_file_stats,
        "output_plot": str(out_plot),
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    print("[DONE] Signal overview plot:", out_plot)
    print("[DONE] Summary JSON:", out_summary)
    print(
        "[INFO] Files used:",
        len(file_names),
        "| Total sampled points:",
        int(sum(len(v) for v in per_file_samples_plot)),
        "| Histogram points:",
        int(all_points.size),
    )


if __name__ == "__main__":
    main()
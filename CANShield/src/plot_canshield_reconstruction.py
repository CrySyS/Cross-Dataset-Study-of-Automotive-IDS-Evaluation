#!/usr/bin/env python3
"""
Plot CANShield original-vs-reconstructed signals for one file.

This mirrors the spirit of CANET diagnostic signal plots, but uses CANShield's
own autoencoder reconstruction output.

Outputs are written to:
  artifacts/reconstruction_plots/<dataset>/ts<T>_sp<S>/<stem>/
    - reconstruction_signals.png
    - reconstruction_summary.json

Usage examples (run from CANShield-main/src):

  python plot_canshield_reconstruction.py \
      --config road \
      --file correlated_signal_attack_1_masquerade

  python plot_canshield_reconstruction.py \
      --config road \
      --file /abs/path/to/file.csv \
      --sampling-period 1 \
      --time-step 50 \
      --n-signals 6 \
      --max-points 4000
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from omegaconf import OmegaConf, open_dict

# Ensure local imports work when called from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset.load_dataset import (  # noqa: E402
    load_data,
    scale_dataset,
    create_x_sequences,
    create_y_sequences,
)
from training.backend_factory import get_model_backend  # noqa: E402


def _find_csv(stem_or_path: str, config) -> Path:
    """Resolve a CSV path from absolute path or stem name."""
    p = Path(stem_or_path)
    if p.exists():
        return p

    if p.suffix != ".csv":
        p_csv = p.with_suffix(".csv")
        if p_csv.exists():
            return p_csv

    stem = p.stem if p.suffix else p.name
    for data_dir in [config.get("test_data_dir"), config.get("train_data_dir")]:
        if not data_dir:
            continue
        candidate = Path(data_dir) / f"{stem}.csv"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find CSV for '{stem_or_path}'. "
        f"Searched in {config.get('test_data_dir')} and {config.get('train_data_dir')}."
    )


def _contiguous_intervals(mask: np.ndarray):
    """Return contiguous True intervals as (start, end) over index axis."""
    intervals = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(mask)))
    return intervals


def _shade_intervals(ax, x, mask, color, alpha, label=None):
    for start, end in _contiguous_intervals(mask):
        x0 = x[start]
        x1 = x[min(end - 1, len(x) - 1)]
        ax.axvspan(x0, x1, color=color, alpha=alpha, label=label)
        label = None


def _extract_attacked_ids_from_filename(file_name: str):
    """Extract attacked CAN IDs (decimal) from a CrySyS-style filename."""
    attacked_ids = []
    for hex_id in re.findall(r"0x([0-9a-fA-F]+)", str(file_name)):
        attacked_ids.append(int(hex_id, 16))
    return sorted(set(attacked_ids))


def _feature_can_id(feature_name: str):
    """Extract decimal CAN ID from feature names like Sig_5_of_ID_768."""
    match = re.search(r"_of_ID_(\d+)$", str(feature_name))
    if not match:
        return None
    return int(match.group(1))


def _select_top_signals(x_seq_last, x_recon_last, y_seq, n_signals):
    """Rank signals by attack-vs-benign MAE gap (fallback to global MAE)."""
    err = np.abs(x_recon_last - x_seq_last)
    attack_mask = y_seq.astype(int) == 1
    benign_mask = y_seq.astype(int) == 0

    if attack_mask.any() and benign_mask.any():
        attack_err = err[attack_mask].mean(axis=0)
        benign_err = err[benign_mask].mean(axis=0)
        score = attack_err - benign_err
    else:
        score = err.mean(axis=0)

    idx = np.argsort(score)[::-1]
    return idx[: max(1, min(n_signals, x_seq_last.shape[1]))]


def _load_loss_threshold(args):
    """Try to load loss threshold for this setting. Returns float or None."""
    th_path = Path(
        f"{args.root_dir}/../data/thresholds/{args.dataset_name}/"
        f"thresholds_loss_{args.dataset_name}_{args.num_signals}_{args.time_step}_{args.sampling_period}.csv"
    )
    if not th_path.exists():
        return None

    df = pd.read_csv(th_path)
    target_factor = float(args.loss_factor)
    row = df.loc[np.isclose(df["loss_factor"].astype(float), target_factor)]
    if row.empty:
        return None
    return float(row.iloc[0]["th"])


def _make_plot(
    out_path: Path,
    file_name: str,
    features,
    x_seq,
    x_recon,
    y_seq,
    top_idx,
    threshold,
    attacked_ids,
    max_points,
    plot_start,
    plot_end,
):
    """Save a multi-panel plot with overall loss and top-signal overlays."""
    # Use the last timestep of each window to build timeline-like traces.
    x_last = x_seq[:, -1, :, 0]
    r_last = x_recon[:, -1, :, 0]

    n = x_last.shape[0]
    start = max(0, int(plot_start))
    end = n if plot_end is None else min(n, int(plot_end))
    if end <= start:
        raise ValueError(
            f"Invalid plot range: start={start}, end={end}, total_windows={n}. "
            "Expected end > start."
        )

    x_last = x_last[start:end]
    r_last = r_last[start:end]
    y_plot = y_seq[start:end].astype(int)

    n_plot = x_last.shape[0]
    if n_plot > max_points:
        step = max(1, n_plot // max_points)
        sl = slice(0, n_plot, step)
    else:
        sl = slice(None)

    x_axis = np.arange(start, end)[sl]
    x_last = x_last[sl]
    r_last = r_last[sl]
    y_plot = y_plot[sl]

    err_all = np.abs(r_last - x_last)
    score = err_all.mean(axis=1)
    attack_mask = y_plot == 1
    detect_mask = score >= threshold if threshold is not None else np.zeros_like(score, dtype=bool)

    n_panels = 1 + len(top_idx)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Panel 0: mean absolute reconstruction error
    ax0 = axes[0]
    _shade_intervals(ax0, x_axis, attack_mask, color="salmon", alpha=0.28, label="Attack")
    if threshold is not None:
        _shade_intervals(ax0, x_axis, detect_mask, color="khaki", alpha=0.35, label="Detected")

    ax0.plot(x_axis, score, color="steelblue", linewidth=0.9, label="Mean abs recon error")
    if threshold is not None:
        ax0.axhline(threshold, color="red", linestyle="--", linewidth=1.0, label=f"Loss threshold ({threshold:.4g})")

    ax0.set_yscale("log")
    ax0.set_ylabel("Error (log)")
    ax0.set_title(f"{file_name} - CANShield reconstruction diagnostics")
    handles = [mpatches.Patch(color="salmon", alpha=0.5, label="Attack window")]
    if threshold is not None:
        handles.append(mpatches.Patch(color="khaki", alpha=0.6, label="Detected (score >= threshold)"))
    ax0.legend(handles=handles + ax0.lines, loc="upper right", fontsize=8, ncol=2)

    # Lower panels: selected signals
    for i, sig_idx in enumerate(top_idx):
        sig_idx = int(sig_idx)
        ax = axes[i + 1]
        signal_name = str(features[sig_idx])
        signal_id = _feature_can_id(signal_name)

        # For CrySyS attacked files, only shade panels belonging to attacked CAN IDs.
        # For other datasets or unparseable names, keep legacy behavior.
        if attacked_ids and signal_id is not None:
            panel_attack_mask = attack_mask if signal_id in attacked_ids else np.zeros_like(attack_mask, dtype=bool)
        else:
            panel_attack_mask = attack_mask

        _shade_intervals(ax, x_axis, panel_attack_mask, color="salmon", alpha=0.20)
        ax.plot(x_axis, x_last[:, sig_idx], color="steelblue", linewidth=0.8, label="Original (scaled)")
        ax.plot(
            x_axis,
            r_last[:, sig_idx],
            color="darkorange",
            linewidth=0.8,
            linestyle="--",
            alpha=0.9,
            label="Reconstructed",
        )
        ax.set_ylabel(features[sig_idx], fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Window index")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot CANShield original vs reconstructed signals for one file.")
    parser.add_argument("--config", required=True, choices=["syncan", "road", "crysys"], help="Config name in ../config.")
    parser.add_argument("--file", required=True, help="CSV path or bare stem (searched in test/train dirs).")
    parser.add_argument("--time-step", type=int, default=None, help="Override time_step.")
    parser.add_argument("--sampling-period", type=int, default=None, help="Override sampling_period.")
    parser.add_argument("--window-step", type=int, default=None, help="Override window step used for sequence creation.")
    parser.add_argument(
        "--output-dataset-name",
        type=str,
        default=None,
        help="Optional model namespace (artifacts/models/<name>) to load for inference.",
    )
    parser.add_argument("--per-of-samples", type=float, default=1.0, help="Fraction of rows to load from the file.")
    parser.add_argument("--n-signals", type=int, default=6, help="Number of top signals to plot.")
    parser.add_argument("--max-points", type=int, default=4000, help="Max points on x-axis (downsampled if larger).")
    parser.add_argument("--plot-start", type=int, default=0, help="Window index to start plotting from (inclusive).")
    parser.add_argument("--plot-end", type=int, default=None, help="Window index to stop plotting at (exclusive).")
    parser.add_argument("--loss-factor", type=float, default=None, help="Loss factor to pick threshold row (default from config).")
    parser.add_argument("--no-threshold", action="store_true", help="Do not overlay threshold/detection shading.")
    args_cli = parser.parse_args()

    cfg_path = Path(__file__).resolve().parent / ".." / "config" / f"{args_cli.config}.yaml"
    args = OmegaConf.load(cfg_path)

    root_dir = Path(__file__).resolve().parent
    with open_dict(args):
        args.root_dir = root_dir
        args.data_type = "testing"
        args.time_step = int(args_cli.time_step or args.time_steps[0])
        args.sampling_period = int(args_cli.sampling_period or args.sampling_periods[0])
        args.window_step = int(args_cli.window_step or args.window_step_test)
        args.per_of_samples = float(args_cli.per_of_samples)
        args.loss_factor = float(args_cli.loss_factor if args_cli.loss_factor is not None else args.loss_factor)
        if args_cli.output_dataset_name:
            args.output_dataset_name = str(args_cli.output_dataset_name)

    csv_path = _find_csv(args_cli.file, args)
    file_name = csv_path.stem

    print(f"[INFO] File: {csv_path}")
    print(
        f"[INFO] Settings: dataset={args.dataset_name}, backend={args.backend}, "
        f"time_step={args.time_step}, sampling_period={args.sampling_period}, "
        f"window_step={args.window_step}, per_of_samples={args.per_of_samples}, "
        f"plot_start={args_cli.plot_start}, plot_end={args_cli.plot_end}"
    )

    min_required_rows = args.sampling_period * args.time_step + 1
    X, y, can_ids = load_data(
        args.dataset_name,
        file_name,
        str(csv_path),
        args.features,
        args.org_columns,
        args.per_of_samples,
        data_type=args.data_type,
        train_only_benign=False,
        min_required_rows=min_required_rows,
    )

    X_scaled = scale_dataset(X, args.dataset_name, args.features, args.scaler_dir)
    x_seq = create_x_sequences(X_scaled, args.time_step, args.window_step, args.num_signals, args.sampling_period)
    y_seq = create_y_sequences(y, args.time_step, args.window_step, args.sampling_period,
                              can_ids=can_ids, file_name=file_name, dataset_name=args.dataset_name)

    backend = get_model_backend(args)
    model = backend.load_model_for_inference(args)
    x_recon = backend.predict_reconstruction(model, x_seq)

    # Release GPU memory aggressively for torch backend after inference.
    if str(getattr(args, "backend", "")).lower() == "torch":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    x_last = x_seq[:, -1, :, 0]
    r_last = x_recon[:, -1, :, 0]
    top_idx = _select_top_signals(x_last, r_last, y_seq, args_cli.n_signals)
    attacked_ids = _extract_attacked_ids_from_filename(file_name)

    if attacked_ids:
        print(f"[INFO] Attacked CAN IDs inferred from filename: {attacked_ids}")
    else:
        print("[INFO] No attacked CAN ID parsed from filename; using global window shading behavior.")

    threshold = None
    if not args_cli.no_threshold:
        threshold = _load_loss_threshold(args)
        if threshold is None:
            print("[WARN] Loss threshold not found for this setting. Plot will omit threshold overlay.")

    output_dataset_name = str(getattr(args, "output_dataset_name", args.dataset_name))

    out_dir = (
        root_dir
        / ".."
        / "artifacts"
        / "reconstruction_plots"
        / output_dataset_name
        / f"ts{args.time_step}_sp{args.sampling_period}"
        / file_name
    )
    out_png = out_dir / "reconstruction_signals.png"

    _make_plot(
        out_path=out_png,
        file_name=file_name,
        features=args.features,
        x_seq=x_seq,
        x_recon=x_recon,
        y_seq=y_seq,
        top_idx=top_idx,
        threshold=threshold,
        attacked_ids=attacked_ids,
        max_points=max(200, int(args_cli.max_points)),
        plot_start=max(0, int(args_cli.plot_start)),
        plot_end=None if args_cli.plot_end is None else int(args_cli.plot_end),
    )

    summary = {
        "file_name": file_name,
        "csv_path": str(csv_path),
        "dataset_name": args.dataset_name,
        "backend": str(args.backend),
        "time_step": int(args.time_step),
        "sampling_period": int(args.sampling_period),
        "window_step": int(args.window_step),
        "per_of_samples": float(args.per_of_samples),
        "plot_start": int(max(0, int(args_cli.plot_start))),
        "plot_end": None if args_cli.plot_end is None else int(args_cli.plot_end),
        "x_seq_shape": list(x_seq.shape),
        "y_seq_shape": list(y_seq.shape),
        "attack_windows": int((y_seq.astype(int) == 1).sum()),
        "benign_windows": int((y_seq.astype(int) == 0).sum()),
        "mean_abs_error": float(np.mean(np.abs(x_recon - x_seq))),
        "median_abs_error": float(np.median(np.abs(x_recon - x_seq))),
        "threshold_used": None if threshold is None else float(threshold),
        "top_signal_indices": [int(i) for i in top_idx.tolist()],
        "top_signal_names": [str(args.features[int(i)]) for i in top_idx.tolist()],
        "attacked_ids_from_filename": [int(v) for v in attacked_ids],
        "output_plot": str(out_png),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "reconstruction_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[DONE] Plot saved to: {out_png}")
    print(f"[DONE] Summary saved to: {out_dir / 'reconstruction_summary.json'}")


if __name__ == "__main__":
    # Avoid TensorFlow spam when keras backend is selected.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()

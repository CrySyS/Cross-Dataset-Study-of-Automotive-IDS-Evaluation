#!/usr/bin/env python3
"""
canet_syncan_signal_plots.py – Per-signal actual-vs-predicted diagnostic plots
for the CANet SynCAN model.

Re-runs inference on each test file (GPU) and generates a multi-panel PNG:
  • Top panel  : overall MSE over time with threshold line, attack window
                 (red), and detection window (yellow) shading.
  • Lower panels: actual vs. predicted for the top-N signals sorted by
                 mean absolute reconstruction error during attack vs. normal.

Usage
-----
    python canet_syncan_signal_plots.py \\
        --experiment-id 2026-03-09_22-58-13 \\
        --epoch 10 \\
        --quantile 0.999 \\
        --n-signals 6

All paths default to the same layout used by canet_syncan_test.py.
"""

import sys
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torch.utils.data import DataLoader

# ── import helpers from canet_syncan_test (same directory) ───────────────────
_CANET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CANET_DIR))
from canet_syncan_test import (  # noqa: E402
    CANetTorch,
    DictTensorDataset,
    FIXED_IDS,
    ID_NSIG,
    load_inputs,
    to_device_batch,
    session_to_int,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_signals(
    model: "CANetTorch",
    x_dict: dict,
    y_true: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred, y_true_aligned) both shape (N, n_sigs)."""
    loader = DataLoader(
        DictTensorDataset(x_dict, y_true),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = to_device_batch(xb, yb, device)
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def build_signal_names() -> list[tuple[str, str]]:
    """Ordered list of (can_id, signal_name) matching y_true column index.
    Follows the same concatenation order as load_inputs: sorted FIXED_IDS."""
    names = []
    for can_id in FIXED_IDS:
        n = ID_NSIG[can_id]
        for i in range(1, n + 1):
            names.append((can_id, f"Signal{i}"))
    return names


def select_top_signals(
    pred: np.ndarray,
    y_true: np.ndarray,
    session_labels: np.ndarray,
    n_top: int,
) -> np.ndarray:
    """
    Return indices of the top-n signals ranked by:
      mean |error| during attack  –  mean |error| during normal.
    Falls back to overall mean |error| if no attack rows exist.
    """
    is_attack = session_labels == 1
    is_normal = session_labels == 0
    err = np.abs(pred - y_true)

    if is_attack.sum() == 0 or is_normal.sum() == 0:
        return np.argsort(err.mean(axis=0))[::-1][:n_top]

    diff = err[is_attack].mean(axis=0) - err[is_normal].mean(axis=0)
    return np.argsort(diff)[::-1][:n_top]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals = []
    in_seg = False
    for i, v in enumerate(mask):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            intervals.append((start, i))
            in_seg = False
    if in_seg:
        intervals.append((start, len(mask)))
    return intervals


def shade_intervals(ax, times, mask, color, alpha, label=None):
    for s, e in contiguous_intervals(mask):
        t0 = times[s]
        t1 = times[min(e, len(times) - 1)]
        ax.axvspan(t0, t1, color=color, alpha=alpha, label=label)
        label = None


def make_signal_plot(
    times: np.ndarray,
    mse: np.ndarray,
    session: np.ndarray,
    pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    signal_names: list[tuple[str, str]],
    top_idx: np.ndarray,
    attack_label: str,
    out_path: Path,
    time_start: float | None = None,
    time_end: float | None = None,
    relative_time: bool = False,
    max_points: int = 60_000,
) -> None:
    if relative_time:
        times = times - float(times[0])

    if time_start is not None or time_end is not None:
        tmask = np.ones(len(times), dtype=bool)
        if time_start is not None:
            tmask &= times >= time_start
        if time_end is not None:
            tmask &= times <= time_end
        if tmask.any():
            times = times[tmask]
            mse = mse[tmask]
            session = session[tmask]
            pred = pred[tmask]
            y_true = y_true[tmask]
        else:
            raise ValueError(
                f"No samples in requested time range: start={time_start}, end={time_end}"
            )

    N = len(times)
    if N > max_points:
        step = max(1, N // max_points)
        sl = slice(0, N, step)
    else:
        sl = slice(None)

    t   = times[sl]
    m   = mse[sl]
    s   = session[sl]
    pr  = pred[sl]
    yt  = y_true[sl]

    attack_m = s == 1
    detect_m = m >= threshold

    n_panels = 1 + len(top_idx)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(16, 3 * n_panels), sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    # ── Panel 0: MSE ─────────────────────────────────────────────────────────
    ax = axes[0]
    shade_intervals(ax, t, attack_m, "salmon", 0.30)
    shade_intervals(ax, t, detect_m, "khaki",  0.50)
    ax.plot(t, m, lw=0.6, color="steelblue", label="MSE")
    ax.axhline(
        threshold, color="red", lw=1.2, ls="--",
        label=f"Threshold ({threshold:.3e})"
    )
    ax.set_yscale("log")
    ax.set_ylabel("MSE (log scale)", fontsize=9)
    ax.set_title(f"{attack_label} – overall MSE", fontsize=10)
    legend_patches = [
        mpatches.Patch(color="salmon", alpha=0.5, label="Attack window"),
        mpatches.Patch(color="khaki",  alpha=0.7, label="Detected (MSE ≥ thr)"),
    ]
    ax.legend(
        handles=legend_patches + ax.lines,
        fontsize=8, loc="upper right", ncol=2,
    )

    # ── Signal panels ─────────────────────────────────────────────────────────
    for pi, sig_idx in enumerate(top_idx):
        ax = axes[1 + pi]
        can_id, col = signal_names[sig_idx]
        shade_intervals(ax, t, attack_m, "salmon", 0.20)
        ax.plot(t, yt[:, sig_idx], lw=0.6, color="steelblue",    label="Actual")
        ax.plot(t, pr[:, sig_idx], lw=0.6, color="darkorange",
                ls="--", alpha=0.85, label="Predicted")
        ax.set_ylabel(f"{can_id}\n{col}", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(attack_label, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-signal actual-vs-predicted plots for CANet SynCAN"
    )
    parser.add_argument("--dataset-dir",   default="../data_raw/01_SynCAN")
    parser.add_argument("--model-dir",     default="../../models/CANET")
    parser.add_argument("--results-dir",   default="../../Results")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--epoch",         type=int, required=True)
    parser.add_argument("--hidden-size",   type=int, default=5)
    parser.add_argument("--quantile",      type=float, default=0.999)
    parser.add_argument("--n-signals",     type=int, default=6,
        help="Number of top signals to plot per file")
    parser.add_argument("--batch-size",    type=int, default=512)
    parser.add_argument("--window-size",   type=int, default=1)
    parser.add_argument("--eval-stride",   type=int, default=1,
        help="Use every k-th sample for inference/plotting (k>1 speeds up runs)")
    parser.add_argument("--max-points",    type=int, default=60_000)
    parser.add_argument("--max-files",     type=int, default=0,
        help="Process only first N matching test files (0 = all)")
    parser.add_argument("--trace",         default=None,
        help="Optional test filename stem to process only one trace (e.g. test_flooding)")
    parser.add_argument("--time-start",    type=float, default=None,
        help="Optional lower Time(s) bound for plotting")
    parser.add_argument("--time-end",      type=float, default=None,
        help="Optional upper Time(s) bound for plotting")
    parser.add_argument("--relative-time", action="store_true",
        help="Plot with trace-relative time axis (first sample at t=0)")
    parser.add_argument("--out-dir",       default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir   = Path(args.model_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    exp_id = args.experiment_id
    q_str  = f"{args.quantile:.4f}".replace(".", "")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = results_dir / f"Syncan_CANet_{exp_id}_signals_q{q_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Load model ───────────────────────────────────────────────────────────
    weight_file = model_dir / f"Syncan_{exp_id}_epoch{args.epoch:02d}"
    checkpoint  = torch.load(weight_file, map_location=device)

    model = CANetTorch(args.hidden_size).to(device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model: {weight_file.name}")
    print(f"Fixed IDs: {FIXED_IDS}")

    signal_names = build_signal_names()
    print(f"Total signals in model output: {len(signal_names)}")

    # ── Compute threshold from valid parquet ─────────────────────────────────
    valid_parquet = results_dir / f"Syncan_CANet_{exp_id}_valid.parquet"
    if not valid_parquet.exists():
        raise FileNotFoundError(
            f"Valid parquet not found: {valid_parquet}\n"
            "Run canet_syncan_test.py first to generate it."
        )
    valid_df  = pd.read_parquet(valid_parquet)
    threshold = float(np.quantile(valid_df["MSE"].to_numpy(), args.quantile))
    print(f"Threshold (q={args.quantile}): {threshold:.6e}")

    # ── Test files ────────────────────────────────────────────────────────────
    test_files = [
        dataset_dir / "test_normal.csv",
        dataset_dir / "test_flooding.csv",
        dataset_dir / "test_plateau.csv",
        dataset_dir / "test_continuous.csv",
        dataset_dir / "test_playback.csv",
        dataset_dir / "test_suppress.csv",
    ]
    test_files = [p for p in test_files if p.exists()]
    if args.trace:
        test_files = [p for p in test_files if p.stem == args.trace]
    if args.max_files > 0:
        test_files = test_files[: args.max_files]
    if not test_files:
        raise FileNotFoundError("No SynCAN test files matched --trace/--max-files filters")
    print(f"Found {len(test_files)} test files")

    time_cutoff = args.window_size + 1

    for tf in test_files:
        attack_label = tf.stem  # e.g. "test_flooding"
        out_path = out_dir / f"signals_{attack_label}.png"
        if out_path.exists():
            print(f"  Skipping (exists): {out_path.name}")
            continue

        print(f"\nProcessing: {attack_label}")
        try:
            x_dict, y_raw, time_label = load_inputs(tf, time_cutoff=time_cutoff, shuffle=False)
        except Exception as exc:
            print(f"  ERROR loading {tf.stem}: {exc}")
            continue

        # Optional inference downsampling for very large SynCAN files.
        if args.eval_stride > 1:
            stride = args.eval_stride
            x_dict = {k: v[::stride] for k, v in x_dict.items()}
            y_raw = y_raw[::stride]
            time_label = time_label[::stride]
            print(f"  Applied eval stride: 1/{stride} (rows now: {len(y_raw):,})")

        pred, y_true = predict_with_signals(
            model, x_dict, y_raw, args.batch_size, device
        )

        times   = np.array(time_label[:, 0], dtype=float)
        session = session_to_int(time_label[:, 1])
        mse     = np.mean((pred - y_true) ** 2, axis=1)
        t_for_print = times - float(times[0]) if args.relative_time else times
        print(f"  Time span: [{t_for_print.min():.3f}, {t_for_print.max():.3f}] s")

        top_idx = select_top_signals(pred, y_true, session, n_top=args.n_signals)
        print(f"  Top {args.n_signals} signal indices: {top_idx.tolist()}")
        for i in top_idx:
            can_id, col = signal_names[i]
            print(f"    [{i}] {can_id} – {col}")

        make_signal_plot(
            times=times,
            mse=mse,
            session=session,
            pred=pred,
            y_true=y_true,
            threshold=threshold,
            signal_names=signal_names,
            top_idx=top_idx,
            attack_label=attack_label,
            out_path=out_path,
            time_start=args.time_start,
            time_end=args.time_end,
            relative_time=args.relative_time,
            max_points=args.max_points,
        )

        del x_dict, y_raw, pred, y_true
        gc.collect()

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

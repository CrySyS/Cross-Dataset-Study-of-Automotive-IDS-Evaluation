#!/usr/bin/env python3
"""
canet_crysys_signal_plots.py - Per-signal actual-vs-predicted diagnostic plots
for the CANet CrySyS model.

Re-runs inference on selected CrySyS attack logs and generates a multi-panel PNG:
  - Top panel: overall MSE over time with threshold line, attack-window,
    and detection-window shading.
  - Lower panels: actual vs. predicted for the top-N signals sorted by
    mean absolute reconstruction error during attack vs. normal.
"""

import argparse
import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_CANET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CANET_DIR))
from canet_crysys_test import (  # noqa: E402
    CANetTorch,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SIGNAL_MASK_PATH,
    DictTensorDataset,
    _signal_indices_from_columns,
    crysys_files,
    load_inputs,
    load_signal_mask_map,
    prepare_parsed_logs_train,
    to_device_batch,
)


def predict_with_signals(
    model: "CANetTorch",
    x_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        DictTensorDataset(x_dict, y_true),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = to_device_batch(x_batch, y_batch, device)
            pred = model(x_batch)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y_batch.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def build_signal_names(
    fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]]
) -> list[tuple[str, str]]:
    return [(can_id, col) for can_id in fixed_ids for col in id_to_signal_cols[can_id]]


def select_top_signals(
    pred: np.ndarray,
    y_true: np.ndarray,
    session_labels: np.ndarray,
    n_top: int,
) -> np.ndarray:
    is_attack = session_labels == 1
    is_normal = session_labels == 0
    err = np.abs(pred - y_true)

    if is_attack.sum() == 0 or is_normal.sum() == 0:
        return np.argsort(err.mean(axis=0))[::-1][:n_top]

    diff = err[is_attack].mean(axis=0) - err[is_normal].mean(axis=0)
    return np.argsort(diff)[::-1][:n_top]


def contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals = []
    in_seg = False
    for idx, value in enumerate(mask):
        if value and not in_seg:
            start = idx
            in_seg = True
        elif not value and in_seg:
            intervals.append((start, idx))
            in_seg = False
    if in_seg:
        intervals.append((start, len(mask)))
    return intervals


def shade_intervals(ax, times, mask, color, alpha, label=None):
    for start, end in contiguous_intervals(mask):
        t0 = times[start]
        t1 = times[min(end, len(times) - 1)]
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
    trace_label: str,
    out_path: Path,
    max_points: int = 60_000,
) -> None:
    if len(times) > max_points:
        step = max(1, len(times) // max_points)
        sl = slice(0, len(times), step)
    else:
        sl = slice(None)

    t = times[sl]
    m = mse[sl]
    s = session[sl]
    pr = pred[sl]
    yt = y_true[sl]

    attack_m = s == 1
    detect_m = m >= threshold

    n_panels = 1 + len(top_idx)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    shade_intervals(ax, t, attack_m, "salmon", 0.30)
    shade_intervals(ax, t, detect_m, "khaki", 0.50)
    ax.plot(t, m, lw=0.6, color="steelblue", label="MSE")
    ax.axhline(threshold, color="red", lw=1.2, ls="--", label=f"Threshold ({threshold:.3e})")
    ax.set_yscale("log")
    ax.set_ylabel("MSE (log scale)", fontsize=9)
    ax.set_title(f"{trace_label} - overall MSE", fontsize=10)
    legend_patches = [
        mpatches.Patch(color="salmon", alpha=0.5, label="Attack window"),
        mpatches.Patch(color="khaki", alpha=0.7, label="Detected (MSE >= thr)"),
    ]
    ax.legend(handles=legend_patches + ax.lines, fontsize=8, loc="upper right", ncol=2)

    for panel_idx, sig_idx in enumerate(top_idx):
        ax = axes[1 + panel_idx]
        can_id, col = signal_names[sig_idx]
        shade_intervals(ax, t, attack_m, "salmon", 0.20)

        # Show only change-points to avoid thick horizontal bands from repeated per-ID values.
        val_changes = np.concatenate([[True], yt[1:, sig_idx] != yt[:-1, sig_idx]])
        t_id = t[val_changes]
        yt_id = yt[val_changes, sig_idx]
        pr_id = pr[val_changes, sig_idx]

        ax.plot(t_id, yt_id, lw=0.8, color="steelblue", label="Actual")
        ax.plot(t_id, pr_id, lw=0.8, color="darkorange", ls="--", alpha=0.85, label="Predicted")
        ax.set_ylabel(f"ID {can_id}\n{col}", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(trace_label, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-signal actual-vs-predicted plots for CANet CrySyS")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="CrySyS dataset root")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory where model weights are stored")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory with CrySyS CANet parquet outputs")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory to cache parsed CrySyS logs")
    parser.add_argument("--signal-mask-path", default=str(DEFAULT_SIGNAL_MASK_PATH), help="HDF5 signal mask path used for bit-level extraction")
    parser.add_argument("--experiment-id", required=True, help="Timestamp part of trained model filename")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of weight file to load")
    parser.add_argument("--quantile", type=float, default=0.999, help="Quantile for threshold from valid set")
    parser.add_argument("--n-signals", type=int, default=6, help="Number of top signals to plot per file")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--trace", default=None, help="Optional substring filter for trace filename")
    parser.add_argument("--max-files", type=int, default=0, help="Process only first N attack files (0 = all)")
    parser.add_argument("--max-points", type=int, default=60000, help="Max data points per plot (downsampled if larger)")
    parser.add_argument("--out-dir", default=None, help="Output directory; default: <results-dir>/CrySyS_CANet_<exp>_signals_q<q>")
    parser.add_argument("--force", action="store_true", help="Re-generate plots even if output PNG already exists")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    signal_mask_path = Path(args.signal_mask_path).resolve()

    q_str = f"{args.quantile:.4f}".replace(".", "")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_dir / f"CrySyS_CANet_{args.experiment_id}_signals_q{q_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output dir: {out_dir}")

    weight_file = model_dir / f"CrySyS_{args.experiment_id}_epoch{args.epoch:02d}"
    checkpoint = torch.load(weight_file, map_location=device)

    fixed_ids = checkpoint["fixed_ids"]
    id_to_signal_cols = checkpoint["id_to_signal_cols"]
    id_nsig = checkpoint["id_nsig"]
    id_mps = checkpoint["id_mps"]
    hidden_size = int(checkpoint.get("hidden_size", 5))

    id_to_signal_indices = _signal_indices_from_columns(id_to_signal_cols)
    max_signal_index = max(idx for idxs in id_to_signal_indices.values() for idx in idxs)
    signal_mask_map = load_signal_mask_map(signal_mask_path, id_to_signal_indices)

    model = CANetTorch(hidden_size, fixed_ids=fixed_ids, id_nsig=id_nsig).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Loaded model: {weight_file.name}")

    valid_parquet = results_dir / f"CrySyS_CANet_{args.experiment_id}_valid.parquet"
    if not valid_parquet.exists():
        raise FileNotFoundError(
            f"Valid parquet not found: {valid_parquet}\n"
            "Run canet_crysys_test.py first to generate it."
        )
    valid_df = pd.read_parquet(valid_parquet)
    if "MSE" not in valid_df.columns:
        raise KeyError(f"Column 'MSE' not found in {valid_parquet}")
    threshold = float(np.quantile(pd.to_numeric(valid_df["MSE"], errors="coerce").dropna().to_numpy(), args.quantile))
    print(f"Threshold (q={args.quantile}): {threshold:.6e}")

    _, attack_logs = crysys_files(dataset_dir)
    if args.trace:
        attack_logs = [p for p in attack_logs if args.trace in p.stem]
    if args.max_files > 0:
        attack_logs = attack_logs[: args.max_files]
    if not attack_logs:
        raise FileNotFoundError("No CrySyS attack files matched --trace/--max-files filters")
    print(f"Attack files selected: {len(attack_logs)}")

    attack_csvs = prepare_parsed_logs_train(
        attack_logs,
        selected_ids=set(fixed_ids),
        cache_root=cache_dir,
        id_to_signal_indices=id_to_signal_indices,
        signal_mask_map=signal_mask_map,
        max_signal_index=max_signal_index,
    )

    signal_names = build_signal_names(fixed_ids, id_to_signal_cols)
    time_cutoff = args.window_size + 1

    for raw_log, csv_path in zip(attack_logs, attack_csvs):
        out_path = out_dir / f"signals_{raw_log.stem}.png"
        if out_path.exists() and not args.force:
            print(f"  Skipping (exists): {out_path.name}")
            continue

        print(f"\nProcessing: {raw_log.stem}")
        x_dict, y_raw, time_label = load_inputs(
            csv_path,
            time_cutoff,
            fixed_ids,
            id_to_signal_cols,
            id_mps,
        )
        pred, y_true = predict_with_signals(model, x_dict, y_raw, args.batch_size, device)

        times = np.array(pd.to_numeric(time_label[:, 0]), dtype=float)
        session = pd.to_numeric(pd.Series(time_label[:, 1]), errors="coerce").fillna(0).astype(int).to_numpy()
        mse = np.mean((pred - y_true) ** 2, axis=1)

        top_idx = select_top_signals(pred, y_true, session, n_top=args.n_signals)
        print(f"  Top signal indices: {top_idx.tolist()}")

        make_signal_plot(
            times=times,
            mse=mse,
            session=session,
            pred=pred,
            y_true=y_true,
            threshold=threshold,
            signal_names=signal_names,
            top_idx=top_idx,
            trace_label=raw_log.stem,
            out_path=out_path,
            max_points=args.max_points,
        )

        del x_dict, y_raw, pred, y_true
        gc.collect()

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

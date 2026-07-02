#!/usr/bin/env python3
"""Train CANet on ROAD extracted signals with reproducible ROAD-specific config."""

import argparse
import datetime
import gc
import json
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

np.set_printoptions(precision=4, suppress=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "data_raw" / "02_Road_cleaned" / "signal_extractions"
DEFAULT_MODEL_DIR = PROJECT_DIR.parent / "models" / "CANET_ROAD_V3"
DEFAULT_SELECTION_CONFIG = SCRIPT_DIR / "road_signal_selection.json"
DEFAULT_ID_CONFIG = SCRIPT_DIR / "road_id_config_v3.json"


class CANetTorch(nn.Module):
    def __init__(self, hidden_scale: int, fixed_ids: list[str], id_nsig: dict[str, int]):
        super().__init__()
        self.fixed_ids = fixed_ids
        self.n_sigs = sum(id_nsig.values())
        self.lstm_blocks = nn.ModuleDict(
            {
                can_id: nn.LSTM(
                    input_size=id_nsig[can_id],
                    hidden_size=hidden_scale * id_nsig[can_id],
                    batch_first=True,
                )
                for can_id in fixed_ids
            }
        )
        self.fc1 = nn.Linear(hidden_scale * self.n_sigs, (hidden_scale * self.n_sigs) // 2)
        self.fc2 = nn.Linear((hidden_scale * self.n_sigs) // 2, self.n_sigs - 1)
        self.fc3 = nn.Linear(self.n_sigs - 1, self.n_sigs)
        self.act = nn.ELU()

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x_id = []
        for can_id in self.fixed_ids:
            _, (h_n, _) = self.lstm_blocks[can_id](x_dict[can_id])
            x_id.append(h_n[-1])
        x = torch.cat(x_id, dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.act(self.fc3(x))


class DictTensorDataset(Dataset):
    def __init__(self, x_dict: dict[str, np.ndarray], y: np.ndarray):
        self.x_dict = {k: torch.from_numpy(v).float() for k, v in x_dict.items()}
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.x_dict.items()}, self.y[idx]


def parse_selected_signals(config_path: Path) -> dict[str, list[str]]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    selected = data.get("selected_signals", [])
    pat = re.compile(r"^Sig_(\d+)_of_ID_(\d+)$")
    id_to_cols: dict[str, set[str]] = defaultdict(set)
    for item in selected:
        m = pat.match(item)
        if not m:
            raise ValueError(f"Invalid selected signal format: {item}")
        sig_idx, can_id = m.group(1), m.group(2)
        id_to_cols[can_id].add(f"Signal{sig_idx}")
    return {k: sorted(v, key=lambda x: int(x.replace("Signal", ""))) for k, v in id_to_cols.items()}


def road_files(dataset_dir: Path) -> tuple[list[Path], list[Path]]:
    ambient = sorted((dataset_dir / "ambient").glob("*.csv"))
    attacks = sorted((dataset_dir / "attacks").glob("*.csv"))
    ambient = [p for p in ambient if "generated" not in p.parts]
    attacks = [p for p in attacks if "generated" not in p.parts]
    if not ambient:
        raise FileNotFoundError(f"No ambient CSV files found under {dataset_dir / 'ambient'}")
    if not attacks:
        raise FileNotFoundError(f"No attack CSV files found under {dataset_dir / 'attacks'}")
    return ambient, attacks


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {f"Signal_{i}_of_ID": f"Signal{i}" for i in range(1, 23)}
    present = {k: v for k, v in rename_map.items() if k in df.columns}
    if present:
        df = df.rename(columns=present)
    if "Label" in df.columns:
        df = df.rename(columns={"Label": "Session"})
    return df


def compute_normalization_stats(x_dict: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Compute mean and std for each signal in each CAN ID from the data dict.
    
    Args:
        x_dict: dict mapping can_id -> array of shape (n_seq, window_len, n_signals)
    
    Returns:
        dict mapping can_id -> {'mean': ndarray, 'std': ndarray} per signal
    """
    stats = {}
    for can_id, x_arr in x_dict.items():
        # Flatten to (total_points, n_signals) across all sequences and timesteps
        x_flat = x_arr.reshape(-1, x_arr.shape[-1])
        # Compute per-signal statistics
        mean_val = np.mean(x_flat, axis=0)
        std_val = np.std(x_flat, axis=0)
        # Prevent division by zero: min std = 1e-8
        std_val = np.maximum(std_val, 1e-8)
        stats[can_id] = {'mean': mean_val.astype(np.float32), 'std': std_val.astype(np.float32)}
    return stats


def apply_normalization(x_dict: dict[str, np.ndarray], norm_stats: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Apply z-score normalization to data using provided statistics.
    
    Args:
        x_dict: dict mapping can_id -> array of shape (n_seq, window_len, n_signals)
        norm_stats: dict mapping can_id -> {'mean': ndarray, 'std': ndarray}
    
    Returns:
        Normalized data dict with same structure as input
    """
    x_norm = {}
    for can_id, x_arr in x_dict.items():
        if can_id not in norm_stats:
            raise ValueError(f"No normalization stats for CAN ID {can_id}")
        mean_val = norm_stats[can_id]['mean']
        std_val = norm_stats[can_id]['std']
        x_norm[can_id] = ((x_arr - mean_val) / std_val).astype(np.float32)
    return x_norm


def _count_csv_rows(csv_path: Path) -> int:
    n_rows = 0
    with csv_path.open("r", encoding="utf-8") as f:
        _ = f.readline()
        for _ in f:
            n_rows += 1
    return n_rows


def split_train_valid_file(csv_path: Path, out_dir: Path, valid_prop: float = 0.05, chunk_rows: int = 250_000) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{csv_path.stem}_train.csv"
    valid_path = out_dir / f"{csv_path.stem}_valid.csv"
    if train_path.exists():
        train_path.unlink()
    if valid_path.exists():
        valid_path.unlink()

    total_rows = _count_csv_rows(csv_path)
    n_valid = int(total_rows * valid_prop)
    train_cut = max(0, total_rows - n_valid)

    rows_seen = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):
        chunk_len = len(chunk)
        chunk_start = rows_seen
        chunk_end = rows_seen + chunk_len

        train_stop = min(chunk_end, train_cut)
        n_train_chunk = max(0, train_stop - chunk_start)
        n_valid_chunk = chunk_len - n_train_chunk

        if n_train_chunk > 0:
            chunk.iloc[:n_train_chunk].to_csv(train_path, mode="a", index=False, header=not train_path.exists())
        if n_valid_chunk > 0:
            chunk.iloc[n_train_chunk:].to_csv(valid_path, mode="a", index=False, header=not valid_path.exists())

        rows_seen = chunk_end

    return train_path, valid_path


def build_ambient_splits(ambient_files: list[Path], cache_dir: Path, valid_prop: float = 0.05) -> tuple[list[Path], list[Path]]:
    train_files, valid_files = [], []
    for p in ambient_files:
        t, v = split_train_valid_file(p, out_dir=cache_dir, valid_prop=valid_prop)
        train_files.append(t)
        valid_files.append(v)
    return train_files, valid_files


def process_cache_dir(base_dir: Path, prefix: str) -> Path:
    cache_dir = base_dir / f"{prefix}_pid{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def estimate_id_mps(train_files: list[Path], fixed_ids: list[str], sample_rows: int = 300_000) -> dict[str, int]:
    counts = {can_id: 0 for can_id in fixed_ids}
    durations = {can_id: 0.0 for can_id in fixed_ids}

    for file_path in train_files:
        df = pd.read_csv(file_path, nrows=sample_rows)
        df = normalize_columns(df)
        df["ID"] = df["ID"].astype(str)
        for can_id in fixed_ids:
            dfi = df[df["ID"] == can_id]
            if len(dfi) > 2:
                counts[can_id] += len(dfi)
                durations[can_id] += float(dfi["Time"].iloc[-1] - dfi["Time"].iloc[0])

    out = {}
    for can_id in fixed_ids:
        if counts[can_id] > 2 and durations[can_id] > 0:
            mps = counts[can_id] / max(durations[can_id], 1e-6)
            out[can_id] = int(np.clip(round(mps), 2, 200))
        else:
            out[can_id] = 5
    return out


def resolve_id_mps(
    id_config_path: Path,
    fixed_ids: list[str],
    train_files: list[Path],
    id_mps_source: str,
    write_id_config: bool,
) -> dict[str, int]:
    config = {}
    if id_config_path.exists():
        config = json.loads(id_config_path.read_text(encoding="utf-8"))

    configured = config.get("id_mps", {}) if isinstance(config, dict) else {}
    configured = {str(k): int(v) for k, v in configured.items()}

    if id_mps_source == "fixed":
        missing = [can_id for can_id in fixed_ids if can_id not in configured]
        if missing:
            raise ValueError(
                "Missing ID_MPS values in id-config for IDs: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
                "Run once with --id-mps-source estimate and --write-id-config."
            )
        return {can_id: configured[can_id] for can_id in fixed_ids}

    estimated = estimate_id_mps(train_files, fixed_ids=fixed_ids)
    if write_id_config:
        new_config = config if isinstance(config, dict) else {}
        new_config["id_mps"] = estimated
        id_config_path.write_text(json.dumps(new_config, indent=2), encoding="utf-8")
        print(f"Wrote estimated ID_MPS to {id_config_path}")
    return estimated


def load_arrange_data(file_path: Path, selected_ids: set[str]) -> pd.DataFrame:
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    df = normalize_columns(df)
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"].isin(selected_ids)].copy()
    return df


def get_repeated_sequences(data: pd.DataFrame, can_id: str, sig_columns: list[str], n_step: int) -> np.ndarray:
    df_id = data.loc[data["ID"] == can_id, ["Idx"] + sig_columns]
    if df_id.empty:
        raise RuntimeError(f"ID {can_id} has no rows in this slice")

    np_sig = df_id[sig_columns].fillna(0).to_numpy()
    if len(np_sig) < n_step:
        raise RuntimeError(f"ID {can_id} has {len(np_sig)} rows, needs at least n_step={n_step}")

    np_seq = np.lib.stride_tricks.sliding_window_view(np_sig, window_shape=n_step, axis=0)
    np_seq = np_seq.swapaxes(1, 2)
    n_seq = np_seq.shape[0]
    # Use an exclusive end index so an ID appearing on the final row still gets one repeat.
    end_idx_exclusive = int(data["Idx"].iloc[-1]) + 1
    n_repeats = np.diff(df_id["Idx"].to_list() + [end_idx_exclusive])[-n_seq:]
    if (n_repeats <= 0).any():
        raise RuntimeError(f"Non-positive repeat count found for ID {can_id}")
    return np.repeat(np_seq, n_repeats, axis=0)


def prepare_dataset(file_path: Path, time_cutoff: float, fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]], id_mps: dict[str, int], norm_stats: dict[str, dict[str, np.ndarray]] = None) -> dict[str, np.ndarray]:
    data = load_arrange_data(file_path, selected_ids=set(fixed_ids)).reset_index(drop=True)
    if data.empty:
        raise RuntimeError(f"No selected IDs found in {file_path}")

    # Build a contiguous local index for repeat alignment after ID filtering.
    data["Idx"] = np.arange(len(data), dtype=np.int64)

    time_start = float(data["Time"].iloc[0])
    n_rows_to_use = data.loc[data["Time"] > time_start + time_cutoff, "Time"].shape[0]
    if n_rows_to_use <= 0:
        raise RuntimeError(f"No rows left after time cutoff for {file_path}")

    data_dict: dict[str, np.ndarray] = {}
    for can_id in fixed_ids:
        seq_data = get_repeated_sequences(data, can_id, id_to_signal_cols[can_id], id_mps[can_id])
        data_dict[can_id] = seq_data[-n_rows_to_use:].copy()

    # Align all IDs to the same sample count so downstream shuffling/indexing is safe.
    # Different per-ID windows (id_mps) can produce different sequence counts.
    min_len = min(v.shape[0] for v in data_dict.values())
    if min_len <= 0:
        raise RuntimeError(f"No usable rows after preprocessing for {file_path}")
    if len({v.shape[0] for v in data_dict.values()}) > 1:
        data_dict = {k: v[-min_len:].copy() for k, v in data_dict.items()}

    # Apply z-score normalization if statistics are provided
    if norm_stats is not None:
        data_dict = apply_normalization(data_dict, norm_stats)

    return data_dict


def load_inputs(data_path: Path, time_cutoff: float, fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]], id_mps: dict[str, int], shuffle: bool = True, seed: int = 0, norm_stats: dict[str, dict[str, np.ndarray]] = None):
    x_dict = prepare_dataset(data_path, time_cutoff, fixed_ids, id_to_signal_cols, id_mps, norm_stats=norm_stats)
    if shuffle:
        np.random.seed(seed)
        n_samples = len(next(iter(x_dict.values())))
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        x_dict = {k: v[idx] for k, v in x_dict.items()}
    y = np.concatenate([x_dict[can_id][:, -1, :] for can_id in fixed_ids], axis=1)
    return x_dict, y


def _signal_offsets(fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]]) -> dict[str, tuple[int, int]]:
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for can_id in fixed_ids:
        width = len(id_to_signal_cols[can_id])
        offsets[can_id] = (start, start + width)
        start += width
    return offsets


def validate_y_alignment(
    x_dict: dict[str, np.ndarray],
    y: np.ndarray,
    fixed_ids: list[str],
    id_to_signal_cols: dict[str, list[str]],
) -> dict[str, float]:
    y_from_x = np.concatenate([x_dict[can_id][:, -1, :] for can_id in fixed_ids], axis=1)
    abs_diff = np.abs(y_from_x - y)
    return {
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "n_elements": int(abs_diff.size),
    }


def dump_pre_model_debug(
    debug_dir: Path,
    phase: str,
    epoch_idx: int,
    source_name: str,
    x_dict: dict[str, np.ndarray],
    y: np.ndarray,
    fixed_ids: list[str],
    id_to_signal_cols: dict[str, list[str]],
    signal_limit: int,
    plot_points: int,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
    stem = f"{phase}_epoch{epoch_idx + 1:02d}_{safe_source}"

    align = validate_y_alignment(x_dict, y, fixed_ids, id_to_signal_cols)
    offsets = _signal_offsets(fixed_ids, id_to_signal_cols)

    summary = {
        "phase": phase,
        "epoch": int(epoch_idx + 1),
        "source": source_name,
        "n_samples": int(y.shape[0]),
        "y_dim": int(y.shape[1]),
        "alignment": align,
        "ids": fixed_ids,
        "id_shapes": {can_id: list(x_dict[can_id].shape) for can_id in fixed_ids},
    }
    (debug_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for can_id in fixed_ids:
        arr = x_dict[can_id]
        rows.append(
            {
                "ID": can_id,
                "n_samples": int(arr.shape[0]),
                "window": int(arr.shape[1]),
                "n_signals": int(arr.shape[2]),
                "x_min": float(np.min(arr)),
                "x_max": float(np.max(arr)),
                "x_mean": float(np.mean(arr)),
                "x_std": float(np.std(arr)),
                "x_nan_count": int(np.isnan(arr).sum()),
                "y_block_min": float(np.min(y[:, offsets[can_id][0] : offsets[can_id][1]])),
                "y_block_max": float(np.max(y[:, offsets[can_id][0] : offsets[can_id][1]])),
            }
        )
    pd.DataFrame(rows).to_csv(debug_dir / f"{stem}_id_stats.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        n_plot = min(plot_points, y.shape[0])
        ids_to_plot = fixed_ids[: min(2, len(fixed_ids))]
        for can_id in ids_to_plot:
            arr = x_dict[can_id]
            sig_names = id_to_signal_cols[can_id]
            sig_count = min(signal_limit, len(sig_names))
            if sig_count <= 0 or n_plot <= 0:
                continue

            fig, axes = plt.subplots(sig_count, 1, figsize=(11, 3 * sig_count), squeeze=False)
            y_a, y_b = offsets[can_id]
            y_block = y[:, y_a:y_b]

            for sig_i in range(sig_count):
                ax = axes[sig_i, 0]
                ax.plot(arr[:n_plot, -1, sig_i], label="x_last_timestep", linewidth=1.2)
                ax.plot(y_block[:n_plot, sig_i], label="y_target", linewidth=1.0, alpha=0.75)
                ax.set_title(f"ID {can_id} - {sig_names[sig_i]} (first {n_plot} samples)")
                ax.grid(alpha=0.3)
                ax.legend(loc="best")

            fig.tight_layout()
            fig.savefig(debug_dir / f"{stem}_id{can_id}_target_alignment.png", dpi=140)
            plt.close(fig)

            sample_idx = min(max(0, n_plot // 2), arr.shape[0] - 1)
            fig2, axes2 = plt.subplots(sig_count, 1, figsize=(11, 3 * sig_count), squeeze=False)
            for sig_i in range(sig_count):
                ax2 = axes2[sig_i, 0]
                ax2.plot(arr[sample_idx, :, sig_i], marker="o", linewidth=1.2)
                ax2.set_title(
                    f"ID {can_id} - {sig_names[sig_i]} window profile at sample {sample_idx}"
                )
                ax2.grid(alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(debug_dir / f"{stem}_id{can_id}_window_profile.png", dpi=140)
            plt.close(fig2)
    except Exception as exc:
        print(f"[debug] plotting skipped ({type(exc).__name__}: {exc})")


def slice_data(file_path: Path, n_sliced: int, max_rows_per_slice: int = 120_000) -> list[Path]:
    if file_path.suffix != ".csv":
        raise ValueError("Memory-safe slicing currently supports CSV inputs only.")

    total_rows = _count_csv_rows(file_path)
    target_rows = max(1, math.ceil(total_rows / max(1, n_sliced)))
    target_rows = min(target_rows, max_rows_per_slice)

    cache_dir = process_cache_dir(file_path.parent, prefix="cache")
    cache_dir.mkdir(exist_ok=True)

    paths: list[Path] = []
    chunk_index = 1
    for chunk_df in pd.read_csv(file_path, chunksize=target_rows):
        save_path = cache_dir / f"{file_path.stem}_{chunk_index}.parquet"
        temp_path = save_path.with_suffix(f"{save_path.suffix}.tmp.{os.getpid()}")
        chunk_df.to_parquet(temp_path, index=False)
        temp_path.replace(save_path)
        paths.append(save_path)
        chunk_index += 1
    return paths


def to_device_batch(x_batch: dict[str, torch.Tensor], y_batch: torch.Tensor, device: torch.device):
    x_batch = {k: v.to(device, non_blocking=True) for k, v in x_batch.items()}
    y_batch = y_batch.to(device, non_blocking=True)
    return x_batch, y_batch


def run_epoch(model: CANetTorch, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = to_device_batch(x_batch, y_batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        bs = y_batch.size(0)
        total_loss += float(loss.item()) * bs
        n_samples += bs
    return total_loss / max(1, n_samples)


def eval_epoch(model: CANetTorch, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = to_device_batch(x_batch, y_batch, device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            bs = y_batch.size(0)
            total_loss += float(loss.item()) * bs
            n_samples += bs
    return total_loss / max(1, n_samples)


def compute_train_normalization_stats(train_files: list[Path], time_cutoff: float, fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]], id_mps: dict[str, int], max_files: int = None) -> dict[str, dict[str, np.ndarray]]:
    """Compute normalization statistics (mean, std) from training files only.
    
    This computes once at the start of training and is used for all subsequent
    train/valid/test data to ensure anomalies remain detectable.
    
    Args:
        train_files: List of training CSV/parquet files
        time_cutoff: Time cutoff for each file
        fixed_ids: List of CAN IDs to include
        id_to_signal_cols: Mapping of CAN ID to signal column names
        id_mps: Messages per second for each CAN ID
        max_files: Optional limit on number of files to process
    
    Returns:
        dict: {can_id: {'mean': ndarray, 'std': ndarray}}
    """
    print("Computing normalization statistics from training data...")
    all_stats = {}
    
    for file_idx, train_file in enumerate(train_files[:max_files] if max_files else train_files):
        print(f"  Processing file {file_idx + 1}: {train_file.name}")
        try:
            x_dict = prepare_dataset(train_file, time_cutoff, fixed_ids, id_to_signal_cols, id_mps, norm_stats=None)
            file_stats = compute_normalization_stats(x_dict)
            
            # Accumulate stats (running mean/std using Welford's algorithm would be more precise,
            # but for practical purposes we'll update with weighted average)
            for can_id, stats in file_stats.items():
                if can_id not in all_stats:
                    all_stats[can_id] = {'mean': stats['mean'].copy(), 'std': stats['std'].copy(), 'count': 1}
                else:
                    # Simple averaging for mean; for std, use combined statistics
                    old_count = all_stats[can_id]['count']
                    new_count = old_count + 1
                    all_stats[can_id]['mean'] = (all_stats[can_id]['mean'] * old_count + stats['mean']) / new_count
                    # For std, use the max to be conservative (use larger std to avoid cutting off anomalies)
                    all_stats[can_id]['std'] = np.maximum(all_stats[can_id]['std'], stats['std'])
                    all_stats[can_id]['count'] = new_count
        except Exception as e:
            print(f"    Warning: failed to process {train_file.name}: {e}")
            continue
    
    # Clean up temporary count field
    result = {}
    for can_id, stats in all_stats.items():
        result[can_id] = {'mean': stats['mean'], 'std': stats['std']}
    
    print(f"  Computed stats for {len(result)} CAN IDs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CANet ROAD v3 with reproducible ID config")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="ROAD signal extraction root")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory to save model weights")
    parser.add_argument("--selection-config", default=str(DEFAULT_SELECTION_CONFIG), help="JSON with selected signal list")
    parser.add_argument("--id-config", default=str(DEFAULT_ID_CONFIG), help="JSON with frozen id_mps values")
    parser.add_argument("--id-mps-source", choices=["fixed", "estimate"], default="fixed", help="Use frozen id_mps or estimate from ambient train files")
    parser.add_argument("--write-id-config", action="store_true", help="Write estimated id_mps back to --id-config")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--hidden-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--train-slices", type=int, default=10)
    parser.add_argument("--valid-slices", type=int, default=1)
    parser.add_argument("--valid-prop", type=float, default=0.05)
    parser.add_argument("--max-rows-per-slice", type=int, default=120_000)
    parser.add_argument("--debug-preprocess", action="store_true", help="Dump pre-model tensor checks and plots for a few slices")
    parser.add_argument("--debug-dir", default="", help="Optional debug output dir (default: <model-dir>/debug_preprocess)")
    parser.add_argument("--debug-max-slices", type=int, default=2, help="How many train/valid slices per epoch to dump")
    parser.add_argument("--debug-signal-limit", type=int, default=3, help="Max signals per ID to plot")
    parser.add_argument("--debug-plot-points", type=int, default=500, help="Number of sequence points for alignment plots")
    parser.add_argument("--strict-target-check", action="store_true", help="Fail if y is not exactly x[:, -1, :] after preprocessing")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    id_to_signal_cols = parse_selected_signals(Path(args.selection_config).resolve())
    fixed_ids = sorted(id_to_signal_cols.keys(), key=int)
    id_nsig = OrderedDict((i, len(id_to_signal_cols[i])) for i in fixed_ids)

    print("python", sys.executable)
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible to PyTorch.")
    device = torch.device("cuda")
    print("gpu_name", torch.cuda.get_device_name(device))

    ambient_files, _ = road_files(dataset_dir)
    split_cache = process_cache_dir(dataset_dir / "ambient", prefix="canet_cache_splits_v3")
    train_files, valid_files = build_ambient_splits(ambient_files, cache_dir=split_cache, valid_prop=args.valid_prop)

    id_mps = resolve_id_mps(
        id_config_path=Path(args.id_config).resolve(),
        fixed_ids=fixed_ids,
        train_files=train_files,
        id_mps_source=args.id_mps_source,
        write_id_config=args.write_id_config,
    )

    print(f"selected_ids={len(fixed_ids)} total_selected_signals={sum(id_nsig.values())}")
    print("id_mps", id_mps)

    model = CANetTorch(args.hidden_size, fixed_ids=fixed_ids, id_nsig=id_nsig).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss(reduction="mean")

    # Compute normalization statistics from training data only
    norm_stats = compute_train_normalization_stats(train_files, args.window_size + 1, fixed_ids, id_to_signal_cols, id_mps)

    train_start = args.window_size + 1
    starttime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    debug_dir = Path(args.debug_dir).resolve() if args.debug_dir else model_dir / "debug_preprocess"

    for epoch in range(args.epochs):
        print(f"********** Epoch {epoch + 1} **********")
        debug_train_done = 0
        debug_valid_done = 0

        for data_file in train_files:
            sliced_files = slice_data(data_file, args.train_slices, max_rows_per_slice=args.max_rows_per_slice)
            for sliced_file in sliced_files:
                x_train, y_train = load_inputs(sliced_file, train_start, fixed_ids, id_to_signal_cols, id_mps, shuffle=True, seed=epoch, norm_stats=norm_stats)
                if args.debug_preprocess and debug_train_done < args.debug_max_slices:
                    dump_pre_model_debug(
                        debug_dir=debug_dir,
                        phase="train",
                        epoch_idx=epoch,
                        source_name=sliced_file.name,
                        x_dict=x_train,
                        y=y_train,
                        fixed_ids=fixed_ids,
                        id_to_signal_cols=id_to_signal_cols,
                        signal_limit=args.debug_signal_limit,
                        plot_points=args.debug_plot_points,
                    )
                    align = validate_y_alignment(x_train, y_train, fixed_ids, id_to_signal_cols)
                    print(
                        "[debug] train alignment "
                        f"slice={sliced_file.name} max_abs_diff={align['max_abs_diff']:.3e} "
                        f"mean_abs_diff={align['mean_abs_diff']:.3e}"
                    )
                    if args.strict_target_check and align["max_abs_diff"] > 0:
                        raise RuntimeError(
                            f"Strict target check failed on train slice {sliced_file.name}: "
                            f"max_abs_diff={align['max_abs_diff']}"
                        )
                    debug_train_done += 1
                print(f"Training with {sliced_file.name} {y_train.shape}")
                train_loader = DataLoader(
                    DictTensorDataset(x_train, y_train),
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=True,
                )
                train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
                print(f"train_loss={train_loss:.6f}")
                del x_train, y_train, train_loader
                gc.collect()

        val_loss = 0.0
        for data_file in valid_files:
            sliced_files = slice_data(data_file, args.valid_slices, max_rows_per_slice=args.max_rows_per_slice)
            for sliced_file in sliced_files:
                x_valid, y_valid = load_inputs(sliced_file, train_start, fixed_ids, id_to_signal_cols, id_mps, shuffle=False, norm_stats=norm_stats)
                if args.debug_preprocess and debug_valid_done < args.debug_max_slices:
                    dump_pre_model_debug(
                        debug_dir=debug_dir,
                        phase="valid",
                        epoch_idx=epoch,
                        source_name=sliced_file.name,
                        x_dict=x_valid,
                        y=y_valid,
                        fixed_ids=fixed_ids,
                        id_to_signal_cols=id_to_signal_cols,
                        signal_limit=args.debug_signal_limit,
                        plot_points=args.debug_plot_points,
                    )
                    align = validate_y_alignment(x_valid, y_valid, fixed_ids, id_to_signal_cols)
                    print(
                        "[debug] valid alignment "
                        f"slice={sliced_file.name} max_abs_diff={align['max_abs_diff']:.3e} "
                        f"mean_abs_diff={align['mean_abs_diff']:.3e}"
                    )
                    if args.strict_target_check and align["max_abs_diff"] > 0:
                        raise RuntimeError(
                            f"Strict target check failed on valid slice {sliced_file.name}: "
                            f"max_abs_diff={align['max_abs_diff']}"
                        )
                    debug_valid_done += 1
                print(f"Validating with {sliced_file.name} {y_valid.shape}")
                valid_loader = DataLoader(
                    DictTensorDataset(x_valid, y_valid),
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=True,
                )
                val_loss += eval_epoch(model, valid_loader, loss_fn, device)
                del x_valid, y_valid, valid_loader
                gc.collect()

        print(f"Epoch {epoch + 1} validation loss sum: {val_loss:.6f}")
        weight_name = model_dir / f"RoadV3_{starttime}_epoch{epoch + 1:02d}"
        # Convert norm_stats for pickling (numpy arrays are fine with torch.save)
        norm_stats_serializable = {
            can_id: {
                'mean': stats['mean'] if isinstance(stats['mean'], np.ndarray) else np.array(stats['mean']),
                'std': stats['std'] if isinstance(stats['std'], np.ndarray) else np.array(stats['std'])
            }
            for can_id, stats in norm_stats.items()
        }
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden_size": args.hidden_size,
                "fixed_ids": fixed_ids,
                "id_nsig": dict(id_nsig),
                "id_mps": id_mps,
                "id_to_signal_cols": id_to_signal_cols,
                "valid_prop": args.valid_prop,
                "norm_stats": norm_stats_serializable,
            },
            weight_name,
        )
        print(f"Saved weights: {weight_name}")


if __name__ == "__main__":
    main()

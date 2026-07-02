#!/usr/bin/env python3
"""Run CANet v3 inference on ROAD extracted signals using frozen ROAD config."""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from pympler import asizeof
except ImportError:
    asizeof = None

np.set_printoptions(precision=4, suppress=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "data_raw" / "02_Road_cleaned" / "signal_extractions"
DEFAULT_MODEL_DIR = PROJECT_DIR.parent / "models" / "CANET_ROAD_V3"
DEFAULT_RESULTS_DIR = PROJECT_DIR.parent / "Results"


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


def road_files(dataset_dir: Path) -> tuple[list[Path], list[Path]]:
    ambient = sorted((dataset_dir / "ambient").glob("*.csv"))
    attacks = sorted((dataset_dir / "attacks").glob("*.csv"))
    ambient = [p for p in ambient if "generated" not in p.parts]
    attacks = [p for p in attacks if "generated" not in p.parts]
    if not ambient or not attacks:
        raise FileNotFoundError("ROAD ambient/attack files not found.")
    return ambient, attacks


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {f"Signal_{i}_of_ID": f"Signal{i}" for i in range(1, 23)}
    present = {k: v for k, v in rename_map.items() if k in df.columns}
    if present:
        df = df.rename(columns=present)
    if "Label" in df.columns:
        df = df.rename(columns={"Label": "Session"})
    return df


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


def build_valid_file(ambient_files: list[Path], cache_dir: Path, valid_prop: float = 0.05) -> Path:
    val_parts = []
    for p in ambient_files:
        _, v = split_train_valid_file(p, out_dir=cache_dir, valid_prop=valid_prop)
        val_parts.append(v)
    df = pd.concat([pd.read_csv(p) for p in val_parts], axis=0, ignore_index=True)
    out_valid = cache_dir / "road_valid_merged_v3.csv"
    df.to_csv(out_valid, index=False)
    return out_valid


def process_cache_dir(base_dir: Path, prefix: str) -> Path:
    cache_dir = base_dir / f"{prefix}_pid{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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
    df_id = data.loc[data["ID"] == can_id, ["Idx", "Session"] + sig_columns]
    if df_id.empty:
        raise RuntimeError(f"ID {can_id} has no rows in this slice")

    np_sig = df_id[["Session"] + sig_columns].fillna(0).to_numpy()
    np_sig = np_sig[:, 1:]
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


def prepare_dataset(file_path: Path, time_cutoff: float, fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]], id_mps: dict[str, int], norm_stats: dict[str, dict[str, np.ndarray]] = None):
    data = load_arrange_data(file_path, selected_ids=set(fixed_ids)).reset_index(drop=True)
    if data.empty:
        raise RuntimeError(f"No selected IDs found in {file_path}")

    # Build a contiguous local index for repeat alignment after ID filtering.
    data["Idx"] = np.arange(len(data), dtype=np.int64)

    time_start = float(data["Time"].iloc[0])
    mask = data["Time"] > time_start + time_cutoff
    n_rows_to_use = data.loc[mask, "Time"].shape[0]
    if n_rows_to_use <= 0:
        raise RuntimeError(f"No rows left after time cutoff for {file_path}")

    time_and_labels = data.loc[mask, ["Time", "Session"]].to_numpy()

    data_dict = {}
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
    
    # Align time_and_labels to match the aligned sequence lengths
    time_and_labels = time_and_labels[-min_len:]

    # Apply z-score normalization if statistics are provided
    if norm_stats is not None:
        data_dict = apply_normalization(data_dict, norm_stats)

    return data_dict, time_and_labels


def load_inputs(data_path: Path, time_cutoff: float, fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]], id_mps: dict[str, int], norm_stats: dict[str, dict[str, np.ndarray]] = None):
    x_dict, x_time_label = prepare_dataset(data_path, time_cutoff, fixed_ids, id_to_signal_cols, id_mps, norm_stats=norm_stats)
    y = np.concatenate([x_dict[can_id][:, -1, :] for can_id in fixed_ids], axis=1)
    return x_dict, y, x_time_label


def session_to_int(labels: np.ndarray) -> np.ndarray:
    s = pd.Series(labels)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        vals = numeric.astype(int).to_numpy()
        if set(np.unique(vals).tolist()).issubset({0, 1}):
            return vals
    text = s.astype(str).str.strip().str.lower()
    normal_tokens = {"normal", "benign", "0"}
    return np.where(text.isin(normal_tokens), 0, 1).astype(int)


def to_device_batch(x_batch: dict[str, torch.Tensor], y_batch: torch.Tensor, device: torch.device):
    x_batch = {k: v.to(device, non_blocking=True) for k, v in x_batch.items()}
    y_batch = y_batch.to(device, non_blocking=True)
    return x_batch, y_batch


def predict_mse(model: CANetTorch, x_dict: dict[str, np.ndarray], y_true: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(DictTensorDataset(x_dict, y_true), batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    all_mse = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = to_device_batch(x_batch, y_batch, device)
            pred = model(x_batch)
            mse_batch = torch.mean((pred - y_batch) ** 2, dim=1)
            all_mse.append(mse_batch.detach().cpu().numpy())
    return np.concatenate(all_mse, axis=0)


def save_prediction_file(model: CANetTorch, input_file: Path, save_path: Path, batch_size: int, time_cutoff: float, fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]], id_mps: dict[str, int], device: torch.device, norm_stats: dict[str, dict[str, np.ndarray]] = None) -> int:
    x_dict, y_true, time_label = load_inputs(input_file, time_cutoff, fixed_ids, id_to_signal_cols, id_mps, norm_stats=norm_stats)
    mse_values = predict_mse(model, x_dict, y_true, batch_size=batch_size, device=device)

    results = pd.DataFrame(
        {
            "Time": time_label[:, 0],
            "MSE": mse_values,
            "Session": session_to_int(time_label[:, 1]),
        }
    )
    results["Time"] = pd.to_numeric(results["Time"]).round(7)
    results["Session"] = results["Session"].astype(int)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(save_path, index=False)

    n_rows = len(results)
    del x_dict, y_true, mse_values, results
    gc.collect()
    return n_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANet v3 inference on ROAD files")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="ROAD signal extraction root")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory where model weights are stored")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory to write result parquet files")
    parser.add_argument("--experiment-id", required=True, help="Timestamp part of trained model filename")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of weight file to load")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--max-test-files", type=int, default=0, help="Process only first N attack files (0 means all)")
    parser.add_argument("--valid-prop", type=float, default=0.05)
    args = parser.parse_args()

    print("python", sys.executable)
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible to PyTorch.")
    device = torch.device("cuda")
    print("gpu_name", torch.cuda.get_device_name(device))

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    weight_file = model_dir / f"RoadV3_{args.experiment_id}_epoch{args.epoch:02d}"
    checkpoint = torch.load(weight_file, map_location=device)
    fixed_ids = checkpoint["fixed_ids"]
    id_to_signal_cols = checkpoint["id_to_signal_cols"]
    id_nsig = checkpoint["id_nsig"]
    id_mps = checkpoint["id_mps"]
    hidden_size = int(checkpoint.get("hidden_size", 5))
    # Load normalization statistics (computed on training data only)
    norm_stats = checkpoint.get("norm_stats", None)
    if norm_stats is not None:
        print("Loaded normalization statistics from checkpoint")
    else:
        print("Warning: No normalization statistics in checkpoint - using unnormalized data")

    model = CANetTorch(hidden_size, fixed_ids=fixed_ids, id_nsig=id_nsig).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if asizeof is not None:
        print(f"Model size: {asizeof.asizeof(model) / 1024:.2f} KB")
    print(f"# parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loaded weights: {weight_file}")

    ambient_files, attack_files = road_files(dataset_dir)
    if args.max_test_files > 0:
        attack_files = attack_files[: args.max_test_files]

    valid_cache = process_cache_dir(dataset_dir / "ambient", prefix="canet_cache_splits_v3")
    valid_file = build_valid_file(ambient_files, cache_dir=valid_cache, valid_prop=args.valid_prop)

    time_cutoff = args.window_size + 1

    first_file = attack_files[0]
    print(f"Measuring inference speed on {first_file.name}")
    x_test_dict, y_test, _ = load_inputs(first_file, time_cutoff=time_cutoff, fixed_ids=fixed_ids, id_to_signal_cols=id_to_signal_cols, id_mps=id_mps, norm_stats=norm_stats)
    start_time = time.perf_counter()
    _ = predict_mse(model, x_test_dict, y_test, batch_size=args.batch_size, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    speed = len(y_test) / elapsed if elapsed > 0 else float("inf")
    print("-----------------------------------------")
    print(f"GPU execution time: {elapsed:,} seconds")
    print(f"Inference speed: {speed:.2f} messages per second")
    print("-----------------------------------------")
    del x_test_dict, y_test
    gc.collect()

    try:
        for af in tqdm(attack_files, desc="Scoring ROAD v3 attack files"):
            out_path = results_dir / f"RoadV3_CANet_{args.experiment_id}_{af.stem}.parquet"
            n_rows = save_prediction_file(
                model=model,
                input_file=af,
                save_path=out_path,
                batch_size=args.batch_size,
                time_cutoff=time_cutoff,
                fixed_ids=fixed_ids,
                id_to_signal_cols=id_to_signal_cols,
                id_mps=id_mps,
                device=device,
                norm_stats=norm_stats,
            )
            print(f"Saved {n_rows:,} rows: {out_path}")

        valid_out = results_dir / f"RoadV3_CANet_{args.experiment_id}_valid.parquet"
        n_rows = save_prediction_file(
            model=model,
            input_file=valid_file,
            save_path=valid_out,
            batch_size=args.batch_size,
            time_cutoff=time_cutoff,
            fixed_ids=fixed_ids,
            id_to_signal_cols=id_to_signal_cols,
            id_mps=id_mps,
            device=device,
            norm_stats=norm_stats,
        )
        print(f"Saved {n_rows:,} rows: {valid_out}")
    except KeyboardInterrupt:
        print("Interrupted by user. Partial results already written are kept.")
        return


if __name__ == "__main__":
    main()

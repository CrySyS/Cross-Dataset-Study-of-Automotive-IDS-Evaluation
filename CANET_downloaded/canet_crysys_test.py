#!/usr/bin/env python3
"""Run CANet inference on CrySyS logs with metadata-aware, ID-specific labels."""

import argparse
import gc
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

from canet_crysys_train import (
    DEFAULT_SIGNAL_MASK_PATH,
    assert_only_benign_labels,
    crysys_files,
    load_signal_mask_map,
    parse_attack_intervals,
    plot_trace_csv,
    prepare_parsed_logs as prepare_parsed_logs_train,
    resolve_metadata_json,
)

np.set_printoptions(precision=4, suppress=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "data_raw" / "06_CrySyS_dataset"
DEFAULT_MODEL_DIR = PROJECT_DIR.parent / "models" / "CANET_CRYSYS"
DEFAULT_RESULTS_DIR = PROJECT_DIR.parent / "Results"
DEFAULT_CACHE_DIR = DEFAULT_DATASET_DIR / "_canet_parsed_cache"

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


def _signal_indices_from_columns(id_to_signal_cols: dict[str, list[str]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for can_id, cols in id_to_signal_cols.items():
        indices = []
        for col in cols:
            if not str(col).startswith("Signal"):
                raise ValueError(f"Unexpected signal column name for {can_id}: {col}")
            idx = int(str(col)[6:]) - 1
            if idx < 0:
                raise ValueError(f"Invalid signal index in column name for {can_id}: {col}")
            indices.append(idx)
        out[can_id] = sorted(set(indices))
    return out


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


def build_valid_file(benign_csvs: list[Path], cache_dir: Path, valid_prop: float = 0.05) -> Path:
    merged_csv = cache_dir / "ambient_train.csv"
    merged_csv.parent.mkdir(parents=True, exist_ok=True)

    with merged_csv.open("w", encoding="utf-8") as out_f:
        wrote_header = False
        for p in benign_csvs:
            with p.open("r", encoding="utf-8") as in_f:
                header = in_f.readline()
                if not wrote_header:
                    out_f.write(header)
                    wrote_header = True
                for line in in_f:
                    out_f.write(line)

    _, valid_file = split_train_valid_file(merged_csv, out_dir=cache_dir, valid_prop=valid_prop)
    return valid_file


def load_arrange_data(file_path: Path, selected_ids: set[str]) -> pd.DataFrame:
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    df["ID"] = df["ID"].astype(str).str.lower().str.strip()
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
    end_idx_exclusive = int(data["Idx"].iloc[-1]) + 1
    n_repeats = np.diff(df_id["Idx"].to_list() + [end_idx_exclusive])[-n_seq:]
    if (n_repeats <= 0).any():
        raise RuntimeError(f"Non-positive repeat count found for ID {can_id}")
    return np.repeat(np_seq, n_repeats, axis=0)


def prepare_dataset(
    file_path: Path,
    time_cutoff: float,
    fixed_ids: list[str],
    id_to_signal_cols: dict[str, list[str]],
    id_mps: dict[str, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    data = load_arrange_data(file_path, selected_ids=set(fixed_ids)).reset_index(names="Idx")
    if data.empty:
        raise RuntimeError(f"No selected IDs found in {file_path}")

    time_start = float(data["Time"].iloc[0])
    mask = data["Time"] > time_start + time_cutoff
    n_rows_to_use = int(mask.sum())
    if n_rows_to_use <= 0:
        raise RuntimeError(f"No rows left after time cutoff for {file_path}")

    time_and_labels = data.loc[mask, ["Time", "Label"]].to_numpy()

    data_dict: dict[str, np.ndarray] = {}
    for can_id in fixed_ids:
        seq_data = get_repeated_sequences(data, can_id, id_to_signal_cols[can_id], id_mps[can_id])
        data_dict[can_id] = seq_data[-n_rows_to_use:].copy()

    min_len = min(v.shape[0] for v in data_dict.values())
    if min_len <= 0:
        raise RuntimeError(f"No usable rows after preprocessing for {file_path}")
    if len({v.shape[0] for v in data_dict.values()}) > 1:
        data_dict = {k: v[-min_len:].copy() for k, v in data_dict.items()}
    time_and_labels = time_and_labels[-min_len:]

    return data_dict, time_and_labels


def load_inputs(
    data_path: Path,
    time_cutoff: float,
    fixed_ids: list[str],
    id_to_signal_cols: dict[str, list[str]],
    id_mps: dict[str, int],
):
    x_dict, time_label = prepare_dataset(data_path, time_cutoff, fixed_ids, id_to_signal_cols, id_mps)
    y = np.concatenate([x_dict[can_id][:, -1, :] for can_id in fixed_ids], axis=1)
    return x_dict, y, time_label


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


def save_prediction_file(
    model: CANetTorch,
    input_file: Path,
    save_path: Path,
    batch_size: int,
    time_cutoff: float,
    fixed_ids: list[str],
    id_to_signal_cols: dict[str, list[str]],
    id_mps: dict[str, int],
    device: torch.device,
) -> int:
    x_dict, y_true, time_label = load_inputs(input_file, time_cutoff, fixed_ids, id_to_signal_cols, id_mps)
    mse_values = predict_mse(model, x_dict, y_true, batch_size=batch_size, device=device)

    results = pd.DataFrame(
        {
            "Time": pd.to_numeric(time_label[:, 0]).round(7),
            "MSE": mse_values,
            "Session": pd.to_numeric(pd.Series(time_label[:, 1]), errors="coerce").fillna(0).astype(int).to_numpy(),
        }
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(save_path, index=False)
    n_rows = len(results)

    del x_dict, y_true, mse_values, results
    gc.collect()
    return n_rows


def _attack_intervals_for_cached_csv(csv_path: Path, dataset_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """Best-effort metadata interval lookup for a cached CSV path."""
    scenario = csv_path.parent.name
    raw_log = dataset_dir / scenario / f"{csv_path.stem}.log"
    if not raw_log.exists():
        return {}
    metadata_json = resolve_metadata_json(raw_log)
    default_label = 1 if "-malicious-" in raw_log.name else 0
    return parse_attack_intervals(metadata_json, default_label=default_label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANet inference on CrySyS files")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="CrySyS dataset root")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory where model weights are stored")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory to write result parquet files")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory to cache parsed CrySyS logs")
    parser.add_argument("--signal-mask-path", default=str(DEFAULT_SIGNAL_MASK_PATH), help="HDF5 signal mask path used for bit-level extraction")
    parser.add_argument("--experiment-id", required=True, help="Timestamp part of trained model filename")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of weight file to load")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--max-test-files", type=int, default=0, help="Process only first N attack files (0 means all)")
    parser.add_argument("--valid-prop", type=float, default=0.05)
    parser.add_argument(
        "--debug-plot",
        metavar="CSV_FILE",
        default=None,
        help=(
            "Path to a cached CAN signal CSV file to plot for visual inspection before inference. "
            "Saves a PNG next to the CSV and then exits. "
            "Use 'first_attack' to auto-select the first attack CSV in the cache."
        ),
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()

    # Handle debug-plot early (no GPU needed)
    if args.debug_plot is not None:
        cache_dir = Path(args.cache_dir).resolve()
        if args.debug_plot == "first_attack":
            # Find the first attack CSV in the cache
            import glob
            attack_csvs = sorted(glob.glob(str(cache_dir / "*" / "*-malicious*.csv")))
            if not attack_csvs:
                raise FileNotFoundError(f"No malicious CSV files found in {cache_dir}")
            target_csv = Path(attack_csvs[0])
        else:
            target_csv = Path(args.debug_plot).resolve()
            if not target_csv.exists():
                raise FileNotFoundError(f"--debug-plot CSV not found: {target_csv}")

        plot_output = target_csv.with_suffix(".debug_plot.png")
        attack_intervals = _attack_intervals_for_cached_csv(target_csv, dataset_dir)
        print(f"[debug-plot] csv      : {target_csv.name}")
        print(f"[debug-plot] output   : {plot_output.name}")
        print(f"[debug-plot] intervals: {attack_intervals}")
        plot_trace_csv(target_csv, attack_intervals, plot_output)
        sys.exit(0)

    print("python", sys.executable)
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible to PyTorch.")
    device = torch.device("cuda")
    print("gpu_name", torch.cuda.get_device_name(device))

    model_dir = Path(args.model_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    signal_mask_path = Path(args.signal_mask_path).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_file = model_dir / f"CrySyS_{args.experiment_id}_epoch{args.epoch:02d}"
    checkpoint = torch.load(weight_file, map_location=device)
    fixed_ids = checkpoint["fixed_ids"]
    id_to_signal_cols = checkpoint["id_to_signal_cols"]
    id_nsig = checkpoint["id_nsig"]
    id_mps = checkpoint["id_mps"]
    id_to_signal_indices = _signal_indices_from_columns(id_to_signal_cols)
    max_signal_index = max(idx for idxs in id_to_signal_indices.values() for idx in idxs)
    signal_mask_map = load_signal_mask_map(signal_mask_path, id_to_signal_indices)
    hidden_size = int(checkpoint.get("hidden_size", 5))

    model = CANetTorch(hidden_size, fixed_ids=fixed_ids, id_nsig=id_nsig).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if asizeof is not None:
        print(f"Model size: {asizeof.asizeof(model) / 1024:.2f} KB")
    print(f"# parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loaded weights: {weight_file}")

    benign_logs, attack_logs = crysys_files(dataset_dir)
    if args.max_test_files > 0:
        attack_logs = attack_logs[: args.max_test_files]

    benign_csvs = prepare_parsed_logs_train(
        benign_logs,
        selected_ids=set(fixed_ids),
        cache_root=cache_dir,
        id_to_signal_indices=id_to_signal_indices,
        signal_mask_map=signal_mask_map,
        max_signal_index=max_signal_index,
    )
    assert_only_benign_labels(benign_csvs)

    attack_csvs = prepare_parsed_logs_train(
        attack_logs,
        selected_ids=set(fixed_ids),
        cache_root=cache_dir,
        id_to_signal_indices=id_to_signal_indices,
        signal_mask_map=signal_mask_map,
        max_signal_index=max_signal_index,
    )

    valid_cache = cache_dir / "canet_cache_splits"
    valid_file = build_valid_file(benign_csvs, cache_dir=valid_cache, valid_prop=args.valid_prop)

    time_cutoff = args.window_size + 1

    speed_probe = None
    for p in attack_csvs:
        try:
            x_probe, y_probe, _ = load_inputs(p, time_cutoff, fixed_ids, id_to_signal_cols, id_mps)
            speed_probe = (p, x_probe, y_probe)
            break
        except RuntimeError as exc:
            print(f"Skipping speed probe file {p.name}: {exc}")

    if speed_probe is not None:
        p, x_probe, y_probe = speed_probe
        print(f"Measuring inference speed on {p.name}")
        start_time = time.perf_counter()
        _ = predict_mse(model, x_probe, y_probe, batch_size=args.batch_size, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        speed = len(y_probe) / elapsed if elapsed > 0 else float("inf")
        print("-----------------------------------------")
        print(f"GPU execution time: {elapsed:,} seconds")
        print(f"Inference speed: {speed:.2f} messages per second")
        print("-----------------------------------------")
        del x_probe, y_probe
        gc.collect()

    processed = 0
    skipped = 0
    for af in tqdm(attack_csvs, desc="Scoring CrySyS attack files"):
        out_path = results_dir / f"CrySyS_CANet_{args.experiment_id}_{af.stem}.parquet"
        try:
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
            )
        except RuntimeError as exc:
            print(f"Skipping {af.name}: {exc}")
            skipped += 1
            continue

        print(f"Saved {n_rows:,} rows: {out_path}")
        processed += 1

    valid_out = results_dir / f"CrySyS_CANet_{args.experiment_id}_valid.parquet"
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
    )
    print(f"Saved {n_rows:,} rows: {valid_out}")
    print(f"Finished CrySyS test run. processed_attack_files={processed}, skipped_attack_files={skipped}")


if __name__ == "__main__":
    main()

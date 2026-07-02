#!/usr/bin/env python3
"""Run CANet inference on SynCAN data using PyTorch."""

import argparse
import gc
import sys
import time
from collections import OrderedDict
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

# SynCAN constants from notebook
ID_MPS = {
    "id1": 67,
    "id10": 22,
    "id2": 33,
    "id3": 67,
    "id4": 22,
    "id5": 67,
    "id6": 33,
    "id7": 67,
    "id8": 67,
    "id9": 33,
}
ID_NSIG = OrderedDict(
    [
        ("id1", 2),
        ("id10", 4),
        ("id2", 3),
        ("id3", 2),
        ("id4", 1),
        ("id5", 2),
        ("id6", 2),
        ("id7", 2),
        ("id8", 1),
        ("id9", 1),
    ]
)
FIXED_IDS = sorted(ID_NSIG.keys())
N_SIGS = sum(ID_NSIG.values())


class CANetTorch(nn.Module):
    def __init__(self, hidden_scale: int):
        super().__init__()
        self.lstm_blocks = nn.ModuleDict(
            {
                can_id: nn.LSTM(
                    input_size=ID_NSIG[can_id],
                    hidden_size=hidden_scale * ID_NSIG[can_id],
                    batch_first=True,
                )
                for can_id in FIXED_IDS
            }
        )
        self.fc1 = nn.Linear((hidden_scale * N_SIGS), (hidden_scale * N_SIGS) // 2)
        self.fc2 = nn.Linear((hidden_scale * N_SIGS) // 2, N_SIGS - 1)
        self.fc3 = nn.Linear(N_SIGS - 1, N_SIGS)
        self.act = nn.ELU()

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x_id = []
        for can_id in FIXED_IDS:
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
        x_item = {k: v[idx] for k, v in self.x_dict.items()}
        return x_item, self.y[idx]


def load_arrange_data(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path, delimiter=",")
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    # SynCAN test files often use Signal*_of_ID naming.
    rename_map = {
        "Signal1_of_ID": "Signal1",
        "Signal2_of_ID": "Signal2",
        "Signal3_of_ID": "Signal3",
        "Signal4_of_ID": "Signal4",
    }
    if "Signal1_of_ID" in df.columns:
        df.rename(columns=rename_map, inplace=True)

    df["Time"] = round(df["Time"] / 1000, 7)
    df.rename(columns={"Label": "Session"}, inplace=True)
    return df


def get_repeated_sequences(data: pd.DataFrame, can_id: str, n_sig: int, n_step: int) -> np.ndarray:
    sig_columns = [f"Signal{i}" for i in range(1, n_sig + 1)]
    df_id = data.loc[data["ID"] == can_id, ["Idx", "Session"] + sig_columns]
    np_sig = df_id[["Session"] + sig_columns].to_numpy()
    np_seq = np.lib.stride_tricks.sliding_window_view(np_sig, window_shape=n_step, axis=0)
    np_seq = np_seq.swapaxes(1, 2)
    np_seq = np_seq[:, :, 1:]
    n_seq = np_seq.shape[0]
    end_idx = data["Idx"].iloc[-1]
    n_repeats = np.diff(df_id["Idx"].to_list() + [end_idx])[-n_seq:]
    return np.repeat(np_seq, n_repeats, axis=0)


def prepare_dataset(file_path: Path, time_cutoff: float):
    data = load_arrange_data(file_path).reset_index(names="Idx")
    time_start = data["Time"].iloc[0]
    n_rows_to_use = data.loc[data["Time"] > time_start + time_cutoff, "Time"].shape[0]
    time_and_labels = data.loc[data["Time"] > time_start + time_cutoff, ["Time", "Session"]].to_numpy()

    data_dict = {}
    for can_id, nsig in ID_NSIG.items():
        seq_data = get_repeated_sequences(data, can_id, nsig, ID_MPS[can_id])
        data_dict[can_id] = seq_data[-n_rows_to_use:].copy()

    return data_dict, time_and_labels


def load_inputs(data_path: Path, time_cutoff: float, shuffle: bool = False, seed: int = 0):
    x_dict, x_time_label = prepare_dataset(data_path, time_cutoff=time_cutoff)
    if shuffle:
        np.random.seed(seed)
        n_samples = len(next(iter(x_dict.values())))
        shuffled_idx = np.arange(n_samples)
        np.random.shuffle(shuffled_idx)
        x_dict = {can_id: seqs[shuffled_idx] for can_id, seqs in x_dict.items()}
        x_time_label = x_time_label[shuffled_idx]

    y = np.concatenate([x_dict[can_id][:, -1, :] for can_id in FIXED_IDS], axis=1)
    return x_dict, y, x_time_label


def clean_syncan_test_columns(file_paths: list[Path]) -> None:
    columns_to_rename = {
        "Signal1_of_ID": "Signal1",
        "Signal2_of_ID": "Signal2",
        "Signal3_of_ID": "Signal3",
        "Signal4_of_ID": "Signal4",
    }
    for file_path in tqdm(file_paths, desc="Arranging test dataset"):
        df = pd.read_csv(file_path)
        if "Signal1_of_ID" in df.columns:
            df.rename(columns=columns_to_rename, errors="ignore", inplace=True)
            df.to_csv(file_path, index=False)


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


def predict_mse(
    model: CANetTorch,
    x_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds = DictTensorDataset(x_dict, y_true)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    all_mse = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = to_device_batch(x_batch, y_batch, device)
            pred = model(x_batch)
            mse_batch = torch.mean((pred - y_batch) ** 2, dim=1)
            all_mse.append(mse_batch.detach().cpu().numpy())
    del ds, loader
    return np.concatenate(all_mse, axis=0)


def save_prediction_file(
    model: CANetTorch,
    input_file: Path,
    save_path: Path,
    batch_size: int,
    time_cutoff: float,
    device: torch.device,
) -> int:
    x_dict, y_true, time_label = load_inputs(input_file, time_cutoff=time_cutoff, shuffle=False)
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
    parser = argparse.ArgumentParser(description="Run CANet inference on SynCAN test files (PyTorch)")
    parser.add_argument("--dataset-dir", default="../data_raw/01_SynCAN", help="Path to SynCAN dataset directory")
    parser.add_argument("--model-dir", default="../../models/CANET", help="Directory where model weights are stored")
    parser.add_argument("--results-dir", default="../../Results", help="Directory to write result parquet files")
    parser.add_argument("--experiment-id", required=True, help="Timestamp part of trained model filename")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of weight file to load")
    parser.add_argument("--hidden-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--clean-test-columns", action="store_true")
    parser.add_argument(
        "--max-test-files",
        type=int,
        default=0,
        help="Process only first N test files (0 means all)",
    )
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

    data_files = {
        "valid": [dataset_dir / "train_valid.csv"],
        "test": [
            dataset_dir / "test_normal.csv",
            dataset_dir / "test_flooding.csv",
            dataset_dir / "test_plateau.csv",
            dataset_dir / "test_continuous.csv",
            dataset_dir / "test_playback.csv",
            dataset_dir / "test_suppress.csv",
        ],
    }

    for required in data_files["valid"] + data_files["test"]:
        if not required.exists():
            raise FileNotFoundError(f"Missing expected SynCAN file: {required}")

    if args.clean_test_columns:
        clean_syncan_test_columns(data_files["test"])

    model = CANetTorch(args.hidden_size).to(device)
    if asizeof is not None:
        print(f"Model size: {asizeof.asizeof(model) / 1024:.2f} KB")
    else:
        print("Model size: unavailable (install 'pympler' to enable)")
    print(f"# parameters: {sum(p.numel() for p in model.parameters()):,}")

    weight_file = model_dir / f"Syncan_{args.experiment_id}_epoch{args.epoch:02d}"
    checkpoint = torch.load(weight_file, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded weights: {weight_file}")

    time_cutoff = args.window_size + 1

    first_file = data_files["test"][0]
    print(f"Measuring inference speed on {first_file.name}")
    x_test_dict, y_test, _ = load_inputs(first_file, time_cutoff=time_cutoff, shuffle=False)
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

    test_files = data_files["test"]
    if args.max_test_files > 0:
        test_files = test_files[: args.max_test_files]

    try:
        for test_file in tqdm(test_files, desc="Scoring test files"):
            attack = test_file.stem.split("_")[-1]
            out_path = results_dir / f"Syncan_CANet_{args.experiment_id}_{attack}.parquet"
            n_rows = save_prediction_file(
                model=model,
                input_file=test_file,
                save_path=out_path,
                batch_size=args.batch_size,
                time_cutoff=time_cutoff,
                device=device,
            )
            print(f"Saved {n_rows:,} rows: {out_path}")

        for val_file in tqdm(data_files["valid"], desc="Scoring valid files"):
            attack = val_file.stem.split("_")[-1]
            out_path = results_dir / f"Syncan_CANet_{args.experiment_id}_{attack}.parquet"
            n_rows = save_prediction_file(
                model=model,
                input_file=val_file,
                save_path=out_path,
                batch_size=args.batch_size,
                time_cutoff=time_cutoff,
                device=device,
            )
            print(f"Saved {n_rows:,} rows: {out_path}")
    except KeyboardInterrupt:
        print("Interrupted by user. Partial results already written are kept.")
        return


if __name__ == "__main__":
    main()

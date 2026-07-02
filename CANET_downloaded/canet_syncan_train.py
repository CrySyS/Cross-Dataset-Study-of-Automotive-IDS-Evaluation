#!/usr/bin/env python3
"""Train CANet on SynCAN data using PyTorch."""

import argparse
import datetime
import gc
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


def _count_csv_rows(csv_path: Path) -> int:
    n_rows = 0
    with csv_path.open("r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for _ in f:
            n_rows += 1
    return n_rows


def split_train_valid(
    train_csv: Path,
    valid_prop: float = 0.05,
    chunk_rows: int = 250_000,
) -> tuple[Path, Path]:
    total_rows = _count_csv_rows(train_csv)
    n_valid = int(total_rows * valid_prop)
    train_cut = max(0, total_rows - n_valid)

    train_path = train_csv.parent / "train_train.csv"
    valid_path = train_csv.parent / "train_valid.csv"

    if train_path.exists():
        train_path.unlink()
    if valid_path.exists():
        valid_path.unlink()

    rows_seen = 0
    for chunk in pd.read_csv(train_csv, chunksize=chunk_rows):
        chunk_len = len(chunk)
        chunk_start = rows_seen
        chunk_end = rows_seen + chunk_len

        train_stop = min(chunk_end, train_cut)
        n_train_chunk = max(0, train_stop - chunk_start)
        n_valid_chunk = chunk_len - n_train_chunk

        if n_train_chunk > 0:
            chunk.iloc[:n_train_chunk].to_csv(
                train_path,
                mode="a",
                index=False,
                header=not train_path.exists(),
            )
        if n_valid_chunk > 0:
            chunk.iloc[n_train_chunk:].to_csv(
                valid_path,
                mode="a",
                index=False,
                header=not valid_path.exists(),
            )

        rows_seen = chunk_end

    print(f"Split {train_csv.name} -> {train_path.name}, {valid_path.name}")
    return train_path, valid_path


def merge_raw_train_files(dataset_dir: Path) -> tuple[Path, Path]:
    raw_files = sorted(dataset_dir.glob("train_[0-9].csv"))
    if len(raw_files) != 4:
        raise FileNotFoundError(
            "Expected SynCAN train_1.csv..train_4.csv when train.csv is unavailable."
        )

    columns = ["Label", "Time", "ID", "Signal1", "Signal2", "Signal3", "Signal4"]
    merged_path = dataset_dir / "train.csv"
    with merged_path.open("w", encoding="utf-8") as out_f:
        out_f.write(",".join(columns) + "\n")
        for file_path in raw_files:
            with file_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    parts = line.strip().split(",")
                    if not parts or parts[0] == "Label":
                        continue
                    if len(parts) < len(columns):
                        parts += ["nan"] * (len(columns) - len(parts))
                    out_f.write(",".join(parts[: len(columns)]) + "\n")

    print(f"Created merged training file: {merged_path}")
    return split_train_valid(merged_path, valid_prop=0.05)


def ensure_train_valid_files(dataset_dir: Path) -> tuple[Path, Path]:
    train_train = dataset_dir / "train_train.csv"
    train_valid = dataset_dir / "train_valid.csv"
    if train_train.exists() and train_valid.exists():
        return train_train, train_valid

    train_csv = dataset_dir / "train.csv"
    if train_csv.exists():
        return split_train_valid(train_csv, valid_prop=0.05)

    return merge_raw_train_files(dataset_dir)


def load_arrange_data(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    df["Time"] = round(df["Time"] / 1000, 7)
    df.rename(columns={"Label": "Session"}, inplace=True)
    return df


def get_repeated_sequences(data: pd.DataFrame, can_id: str, n_sig: int, n_step: int) -> np.ndarray:
    sig_columns = [f"Signal{i}" for i in range(1, n_sig + 1)]
    df_id = data.loc[data["ID"] == can_id, ["Idx"] + sig_columns]
    np_sig = df_id[sig_columns].to_numpy()
    np_seq = np.lib.stride_tricks.sliding_window_view(np_sig, window_shape=n_step, axis=0)
    np_seq = np_seq.swapaxes(1, 2)
    n_seq = np_seq.shape[0]
    end_idx = data["Idx"].iloc[-1]
    n_repeats = np.diff(df_id["Idx"].to_list() + [end_idx])[-n_seq:]
    return np.repeat(np_seq, n_repeats, axis=0)


def prepare_dataset(file_path: Path, time_cutoff: float) -> dict[str, np.ndarray]:
    data = load_arrange_data(file_path).reset_index(names="Idx")
    time_start = data["Time"].iloc[0]
    n_rows_to_use = data.loc[data["Time"] > time_start + time_cutoff, "Time"].shape[0]

    data_dict: dict[str, np.ndarray] = {}
    for can_id, nsig in ID_NSIG.items():
        seq_data = get_repeated_sequences(data, can_id, nsig, ID_MPS[can_id])
        data_dict[can_id] = seq_data[-n_rows_to_use:].copy()

    return data_dict


def load_inputs(data_path: Path, time_cutoff: float, shuffle: bool = True, seed: int = 0):
    x_dict = prepare_dataset(data_path, time_cutoff=time_cutoff)
    if shuffle:
        np.random.seed(seed)
        n_samples = len(next(iter(x_dict.values())))
        shuffled_idx = np.arange(n_samples)
        np.random.shuffle(shuffled_idx)
        x_dict = {can_id: seqs[shuffled_idx] for can_id, seqs in x_dict.items()}

    y = np.concatenate([x_dict[can_id][:, -1, :] for can_id in FIXED_IDS], axis=1)
    return x_dict, y


def slice_data(file_path: Path, n_sliced: int, max_rows_per_slice: int = 0) -> list[Path]:
    if file_path.suffix != ".csv":
        raise ValueError("Memory-safe slicing currently supports CSV inputs only.")

    total_rows = _count_csv_rows(file_path)
    target_rows = max(1, math.ceil(total_rows / max(1, n_sliced)))
    if max_rows_per_slice > 0:
        target_rows = min(target_rows, max_rows_per_slice)

    cache_dir = file_path.parent / "cache"
    cache_dir.mkdir(exist_ok=True)

    paths: list[Path] = []
    chunk_index = 1
    for chunk_df in pd.read_csv(file_path, chunksize=target_rows):
        save_path = cache_dir / f"{file_path.stem}_{chunk_index}.parquet"
        chunk_df.to_parquet(save_path, index=False)
        paths.append(save_path)
        chunk_index += 1

    return paths


def to_device_batch(x_batch: dict[str, torch.Tensor], y_batch: torch.Tensor, device: torch.device):
    x_batch = {k: v.to(device, non_blocking=True) for k, v in x_batch.items()}
    y_batch = y_batch.to(device, non_blocking=True)
    return x_batch, y_batch


def run_epoch(
    model: CANetTorch,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CANet with SynCAN data only (PyTorch)")
    parser.add_argument("--dataset-dir", default="../data_raw/01_SynCAN", help="Path to SynCAN dataset directory")
    parser.add_argument("--model-dir", default="../../models/CANET", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--hidden-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--train-slices", type=int, default=10)
    parser.add_argument("--valid-slices", type=int, default=1)
    parser.add_argument(
        "--max-rows-per-slice",
        type=int,
        default=120_000,
        help="Hard cap of rows in each temporary slice (reduces RAM pressure)",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    print("python", sys.executable)
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible to PyTorch.")
    device = torch.device("cuda")
    print("gpu_name", torch.cuda.get_device_name(device))
    pin_memory = device.type == "cuda"

    train_file, valid_file = ensure_train_valid_files(dataset_dir)
    data_files = {"train": [train_file], "valid": [valid_file]}

    model = CANetTorch(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss(reduction="mean")

    train_start = args.window_size + 1
    starttime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for epoch in range(args.epochs):
        print(f"********** Epoch {epoch + 1} **********")

        for data_file in data_files["train"]:
            sliced_files = slice_data(
                data_file,
                args.train_slices,
                max_rows_per_slice=args.max_rows_per_slice,
            )
            for sliced_file in sliced_files:
                x_train_dict, y_train = load_inputs(sliced_file, time_cutoff=train_start, shuffle=True, seed=epoch)
                print(f"Training with {sliced_file.name} {y_train.shape}")
                train_ds = DictTensorDataset(x_train_dict, y_train)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=pin_memory,
                    num_workers=args.num_workers,
                )
                train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
                print(f"train_loss={train_loss:.6f}")
                del x_train_dict, y_train, train_ds, train_loader
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        val_loss = 0.0
        for data_file in data_files["valid"]:
            sliced_files = slice_data(
                data_file,
                args.valid_slices,
                max_rows_per_slice=args.max_rows_per_slice,
            )
            for sliced_file in sliced_files:
                x_valid_dict, y_valid = load_inputs(sliced_file, time_cutoff=train_start, shuffle=False)
                print(f"Validating with {sliced_file.name} {y_valid.shape}")
                valid_ds = DictTensorDataset(x_valid_dict, y_valid)
                valid_loader = DataLoader(
                    valid_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=pin_memory,
                    num_workers=args.num_workers,
                )
                val_loss += eval_epoch(model, valid_loader, loss_fn, device)
                del x_valid_dict, y_valid, valid_ds, valid_loader
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1} validation loss sum: {val_loss:.6f}")
        weight_name = model_dir / f"Syncan_{starttime}_epoch{epoch + 1:02d}"
        torch.save({"state_dict": model.state_dict(), "hidden_size": args.hidden_size}, weight_name)
        print(f"Saved weights: {weight_name}")


if __name__ == "__main__":
    main()

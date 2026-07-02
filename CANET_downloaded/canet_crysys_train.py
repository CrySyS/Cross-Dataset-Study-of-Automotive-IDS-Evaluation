#!/usr/bin/env python3
"""Train CANet on CrySyS raw CAN logs using metadata-derived labels."""

import argparse
import datetime
import gc
import json
import math
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
DEFAULT_DATASET_DIR = PROJECT_DIR / "data_raw" / "06_CrySyS_dataset"
DEFAULT_MODEL_DIR = PROJECT_DIR.parent / "models" / "CANET_CRYSYS"
DEFAULT_SELECTION_CONFIG = SCRIPT_DIR / "crysys_signal_selection.json"
DEFAULT_CACHE_DIR = DEFAULT_DATASET_DIR / "_canet_parsed_cache"
DEFAULT_ID_CONFIG = SCRIPT_DIR / "crysys_id_config.json"
DEFAULT_SIGNAL_MASK_PATH = (
    PROJECT_DIR / "CTCN" / "CAN-Message-Modification-Detection-main" / "src" / "signal_extraction" / "signal_mask_reduced.h5"
)

LOG_LINE_RE = re.compile(r"^\(([-+0-9.eE]+)\)\s+\S+\s+([0-9A-Fa-f]{3})#([0-9A-Fa-f]+)")


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


def parse_selected_signals(config_path: Path) -> dict[str, list[int]]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    selected = data.get("selected_signals", [])
    id_to_indices: dict[str, set[int]] = defaultdict(set)
    pat = re.compile(r"^([0-9A-Fa-f]{3,4})_(\d+)$")
    for item in selected:
        m = pat.match(item)
        if not m:
            raise ValueError(f"Invalid selected signal format: {item}")
        # Normalize IDs to 3-digit lowercase hex to match parsed log IDs like "110".
        can_id = f"{int(m.group(1), 16):03x}"
        sig_idx = int(m.group(2))
        id_to_indices[can_id].add(sig_idx)
    return {k: sorted(v) for k, v in id_to_indices.items()}


def signal_columns_from_indices(id_to_signal_indices: dict[str, list[int]]) -> dict[str, list[str]]:
    return {can_id: [f"Signal{idx + 1}" for idx in indices] for can_id, indices in id_to_signal_indices.items()}


def load_signal_mask_map(signal_mask_path: Path, id_to_signal_indices: dict[str, list[int]]) -> dict[str, dict[int, tuple[int, int]]]:
    if not signal_mask_path.exists():
        raise FileNotFoundError(f"Signal mask not found: {signal_mask_path}")

    mask = pd.read_hdf(signal_mask_path, key="signal_mask")
    required_cols = {"id", "index", "start", "stop"}
    if not required_cols.issubset(mask.columns):
        raise ValueError(f"Signal mask must have columns {required_cols}, got {set(mask.columns)}")

    mask = mask.copy()
    mask["id_norm"] = mask["id"].astype(str).map(lambda x: f"{int(x, 16):03x}")

    out: dict[str, dict[int, tuple[int, int]]] = {}
    missing: list[str] = []
    for can_id, signal_indices in id_to_signal_indices.items():
        out[can_id] = {}
        for sig_idx in signal_indices:
            rows = mask[(mask["id_norm"] == can_id) & (mask["index"].astype(int) == int(sig_idx))]
            if rows.empty:
                missing.append(f"{can_id}_{sig_idx}")
                continue
            row = rows.iloc[0]
            start = int(row["start"])
            stop = int(row["stop"])
            if not (0 <= start < stop <= 64):
                raise ValueError(f"Invalid mask bit range for {can_id}_{sig_idx}: start={start}, stop={stop}")
            out[can_id][sig_idx] = (start, stop)

    if missing:
        raise ValueError(f"Missing signal mask definitions for selected signals: {missing}")
    return out


def _extract_mask_signal(payload_bytes: list[int], start: int, stop: int, scale_to_unit: bool = True) -> float:
    value_u64 = int.from_bytes(bytes(payload_bytes[:8]), byteorder="big", signed=False)
    n_bits = stop - start
    shift = 64 - stop
    raw = (value_u64 >> shift) & ((1 << n_bits) - 1)
    if scale_to_unit:
        return float(raw) / float(1 << n_bits)
    return float(raw)


def crysys_files(dataset_dir: Path) -> tuple[list[Path], list[Path]]:
    benign, malicious = [], []
    for scenario_dir in sorted(dataset_dir.glob("*")):
        if not scenario_dir.is_dir():
            continue
        if scenario_dir.name.startswith("_"):
            continue
        benign.extend(sorted(scenario_dir.glob("*-benign.log")))
        malicious.extend(
            p for p in sorted(scenario_dir.glob("*-malicious-*.log"))
            if "-inj-messages" not in p.name
        )

    if not benign:
        raise FileNotFoundError(f"No benign .log files found under {dataset_dir}")
    if not malicious:
        raise FileNotFoundError(f"No malicious .log files found under {dataset_dir}")
    return benign, malicious


def parse_attack_intervals(metadata_json: Path, default_label: int) -> dict[str, list[tuple[float, float]]]:
    """Return {can_id: [(start, end), ...]} with normalized 3-digit lowercase hex keys.

    Uses the ``packet_ID`` field in each marker so that only the specific attacked
    CAN ID is labeled — other IDs within the same time window remain benign.
    Falls back to the wildcard key ``"*"`` when no packet_ID information is present.
    """
    if not metadata_json.exists():
        if default_label == 0:
            return {}
        return {"*": [(float("-inf"), float("inf"))]}

    meta = json.loads(metadata_json.read_text(encoding="utf-8"))
    label = str(meta.get("label", "")).strip().lower()
    if label == "benign":
        return {}

    markers = meta.get("markers", []) or []
    starts_by_id: dict[str, list[float]] = defaultdict(list)
    ends_by_id: dict[str, list[float]] = defaultdict(list)

    for m in markers:
        desc = str(m.get("description", "")).lower()
        t = float(m.get("time", 0.0))
        raw_id = str(m.get("packet_ID", "")).strip()
        if raw_id:
            try:
                can_id = f"{int(raw_id, 16):03x}"
            except ValueError:
                can_id = raw_id.lower()
        else:
            can_id = "*"
        if "start" in desc:
            starts_by_id[can_id].append(t)
        elif "end" in desc:
            ends_by_id[can_id].append(t)

    if not starts_by_id and not ends_by_id:
        return {"*": [(float("-inf"), float("inf"))]}

    result: dict[str, list[tuple[float, float]]] = {}
    for can_id in set(starts_by_id) | set(ends_by_id):
        starts = sorted(starts_by_id.get(can_id, []))
        ends = sorted(ends_by_id.get(can_id, []))
        intervals: list[tuple[float, float]] = []
        for i, s in enumerate(starts):
            e = ends[i] if i < len(ends) else float("inf")
            if e < s:
                s, e = e, s
            intervals.append((s, e))
        if not intervals and ends:
            intervals.append((float("-inf"), ends[0]))
        result[can_id] = intervals
    return result


def resolve_metadata_json(raw_log: Path) -> Path:
    """Resolve the metadata JSON sidecar for a CrySyS log path.

    Most logs use ``<stem>.json``. Some injected-message artifacts are named
    ``...-inj-messages.log`` while the metadata is stored as ``... .json``
    (without the ``-inj-messages`` suffix).
    """
    primary = raw_log.with_suffix(".json")
    if primary.exists():
        return primary

    stem = raw_log.stem
    if stem.endswith("-inj-messages"):
        alt = raw_log.with_name(stem[: -len("-inj-messages")] + ".json")
        if alt.exists():
            return alt

    return primary


def is_attacked_time(t: float, can_id: str, intervals_by_id: dict[str, list[tuple[float, float]]]) -> int:
    """Return 1 if timestamp ``t`` for ``can_id`` falls within any attack interval for that ID.

    Also checks the ``"*"`` wildcard key for files whose metadata lacks packet_ID.
    """
    for key in (can_id, "*"):
        for a, b in intervals_by_id.get(key, []):
            if a <= t <= b:
                return 1
    return 0


def parse_crysys_log(
    raw_log: Path,
    selected_ids: set[str],
    cache_csv: Path,
    id_to_signal_indices: dict[str, list[int]],
    signal_mask_map: dict[str, dict[int, tuple[int, int]]],
    max_signal_index: int,
) -> Path:
    metadata_json = resolve_metadata_json(raw_log)
    default_label = 1 if "-malicious-" in raw_log.name else 0
    intervals = parse_attack_intervals(metadata_json, default_label=default_label)

    rows = []
    with raw_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_LINE_RE.match(line.strip())
            if not m:
                continue
            ts = float(m.group(1))
            can_id = m.group(2).lower()
            payload_hex = m.group(3).strip().lower()
            if can_id not in selected_ids:
                continue

            payload_hex = payload_hex[:16]
            if len(payload_hex) % 2 == 1:
                payload_hex = payload_hex + "0"

            payload_bytes: list[int] = []
            for i in range(0, min(len(payload_hex), 16), 2):
                payload_bytes.append(int(payload_hex[i : i + 2], 16))
            if len(payload_bytes) < 8:
                payload_bytes.extend([0] * (8 - len(payload_bytes)))

            # Store selected mask-signals in Signal{idx+1} slots; unselected slots remain 0.
            sig_vals = [0.0] * (max_signal_index + 1)
            for sig_idx in id_to_signal_indices.get(can_id, []):
                start, stop = signal_mask_map[can_id][sig_idx]
                sig_vals[sig_idx] = _extract_mask_signal(payload_bytes, start, stop, scale_to_unit=True)

            label = is_attacked_time(ts, can_id, intervals) if default_label == 1 else 0
            rows.append([label, ts, can_id] + sig_vals)

    if not rows:
        raise RuntimeError(f"No usable messages parsed from {raw_log}")

    df = pd.DataFrame(rows, columns=["Label", "Time", "ID"] + [f"Signal{i}" for i in range(1, max_signal_index + 2)])
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return cache_csv


def plot_trace_csv(
    csv_path: Path,
    attack_intervals_by_id: dict[str, list[tuple[float, float]]],
    output_path: Path,
) -> None:
    """Plot one subplot per (CAN ID, Signal) for clearer visual inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    df = pd.read_csv(csv_path)
    df["ID"] = df["ID"].astype(str).str.lower()
    signal_cols = [c for c in df.columns if c.startswith("Signal")]
    unique_ids = sorted(df["ID"].unique(), key=lambda x: int(x, 16))

    panel_order: list[tuple[str, str]] = []
    for can_id in unique_ids:
        df_id = df[df["ID"] == can_id]
        active_cols: list[str] = []
        for sig_col in signal_cols:
            vals = df_id[sig_col].fillna(0).to_numpy()
            if np.any(np.abs(vals) > 1e-12):
                active_cols.append(sig_col)
        if not active_cols:
            active_cols = signal_cols[:1]
        panel_order.extend((can_id, sig_col) for sig_col in active_cols)

    n_panels = len(panel_order)
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, max(3, 2.2 * n_panels)), sharex=True, squeeze=False)

    for row_idx, (can_id, sig_col) in enumerate(panel_order):
        ax = axes[row_idx][0]
        df_id = df[df["ID"] == can_id]
        times = df_id["Time"].to_numpy()
        vals = df_id[sig_col].fillna(0).to_numpy()
        labels = df_id["Label"].to_numpy(dtype=int)

        benign_mask = labels == 0
        attacked_mask = labels == 1
        benign_vals = np.where(benign_mask, vals, np.nan)
        attacked_vals = np.where(attacked_mask, vals, np.nan)
        ax.plot(times, benign_vals, color="steelblue", linewidth=0.8, alpha=0.95)
        ax.plot(times, attacked_vals, color="crimson", linewidth=0.8, alpha=0.95)

        for key in (can_id, "*"):
            for a, b in attack_intervals_by_id.get(key, []):
                ax.axvspan(
                    a if a != float("-inf") else times[0],
                    b if b != float("inf") else times[-1],
                    alpha=0.12,
                    color="red",
                    zorder=0,
                )

        n_attacked = int((df_id["Label"] == 1).sum())
        ax.set_ylabel(sig_col, fontsize=8)
        ax.set_title(f"ID {can_id} - {sig_col} ({n_attacked} attacked msgs)", fontsize=9)
        ax.tick_params(labelsize=7)

        legend_handles = [
            Line2D([0], [0], color="steelblue", lw=1.4, label="benign"),
            Line2D([0], [0], color="crimson", lw=1.4, label="attacked"),
            Patch(facecolor="red", alpha=0.12, label="attack window (metadata)"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.85)

    axes[-1][0].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(
        f"CrySyS trace: {csv_path.name}\n"
        "one panel per (CAN ID, Signal)",
        fontsize=10,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[debug-plot] saved → {output_path}")


def _cache_looks_like_mask_signals(csv_path: Path, expected_signal_cols: set[str], sample_rows: int = 2000) -> bool:
    if not csv_path.exists():
        return False
    try:
        sample = pd.read_csv(csv_path, nrows=sample_rows)
    except Exception:
        return False
    if sample.empty:
        return False
    if not expected_signal_cols.issubset(sample.columns):
        return False

    # Mask-extracted and scaled signals should stay in [0, 1].
    vals = sample[list(expected_signal_cols)].to_numpy(dtype=float)
    if np.isnan(vals).all():
        return False
    return float(np.nanmax(vals)) <= 1.000001 and float(np.nanmin(vals)) >= -1e-9


def prepare_parsed_logs(
    raw_logs: list[Path],
    selected_ids: set[str],
    cache_root: Path,
    id_to_signal_indices: dict[str, list[int]],
    signal_mask_map: dict[str, dict[int, tuple[int, int]]],
    max_signal_index: int,
) -> list[Path]:
    out = []
    expected_signal_cols = {f"Signal{idx + 1}" for idxs in id_to_signal_indices.values() for idx in idxs}
    for raw_log in raw_logs:
        scenario = raw_log.parent.name
        cache_csv = cache_root / scenario / f"{raw_log.stem}.csv"
        if not _cache_looks_like_mask_signals(cache_csv, expected_signal_cols):
            parse_crysys_log(
                raw_log=raw_log,
                selected_ids=selected_ids,
                cache_csv=cache_csv,
                id_to_signal_indices=id_to_signal_indices,
                signal_mask_map=signal_mask_map,
                max_signal_index=max_signal_index,
            )
        out.append(cache_csv)
    return out


def assert_only_benign_labels(csv_paths: list[Path], chunk_rows: int = 250_000) -> None:
    for csv_path in csv_paths:
        for chunk in pd.read_csv(csv_path, usecols=["Label"], chunksize=chunk_rows):
            bad_labels = sorted(set(int(v) for v in chunk["Label"].dropna().unique() if int(v) != 0))
            if bad_labels:
                raise AssertionError(
                    f"Benign training input contains non-zero labels in {csv_path}: {bad_labels}"
                )


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


def build_ambient_splits(benign_csvs: list[Path], cache_dir: Path, valid_prop: float = 0.05) -> tuple[list[Path], list[Path]]:
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

    train_file, valid_file = split_train_valid_file(merged_csv, out_dir=cache_dir, valid_prop=valid_prop)
    return [train_file], [valid_file]


def estimate_id_mps(train_files: list[Path], fixed_ids: list[str], sample_rows: int = 300_000) -> dict[str, int]:
    counts = {can_id: 0 for can_id in fixed_ids}
    durations = {can_id: 0.0 for can_id in fixed_ids}

    for file_path in train_files:
        df = pd.read_csv(file_path, nrows=sample_rows)
        df["ID"] = df["ID"].astype(str).str.lower()
        for can_id in fixed_ids:
            dfi = df[df["ID"] == can_id]
            if len(dfi) > 2:
                counts[can_id] += len(dfi)
                durations[can_id] += float(dfi["Time"].iloc[-1] - dfi["Time"].iloc[0])

    out = {}
    for can_id in fixed_ids:
        if counts[can_id] > 2 and durations[can_id] > 0:
            mps = counts[can_id] / max(durations[can_id], 1e-6)
            out[can_id] = int(np.clip(round(mps), 2, 400))
        else:
            out[can_id] = 5
    return out


def resolve_id_mps(
    train_files: list[Path],
    fixed_ids: list[str],
    id_mps_source: str,
    id_config: Path,
    write_id_config: bool,
) -> dict[str, int]:
    if id_mps_source == "fixed":
        if not id_config.exists():
            raise FileNotFoundError(
                f"ID config not found: {id_config}. Use --id-mps-source estimate --write-id-config once."
            )
        cfg = json.loads(id_config.read_text(encoding="utf-8"))
        raw_id_mps = cfg.get("id_mps", {})
        id_mps: dict[str, int] = {}
        missing: list[str] = []
        for can_id in fixed_ids:
            if can_id not in raw_id_mps:
                missing.append(can_id)
                continue
            id_mps[can_id] = int(raw_id_mps[can_id])
        if missing:
            raise ValueError(f"ID config missing IDs: {missing}")
        return id_mps

    id_mps = estimate_id_mps(train_files, fixed_ids=fixed_ids)
    if write_id_config:
        payload = {
            "notes": "Frozen id_mps for CANet CrySyS training.",
            "id_mps": {k: int(v) for k, v in id_mps.items()},
        }
        id_config.parent.mkdir(parents=True, exist_ok=True)
        id_config.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote ID config: {id_config}")
    return id_mps


def load_arrange_data(file_path: Path, selected_ids: set[str]) -> pd.DataFrame:
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    df["ID"] = df["ID"].astype(str).str.lower()
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


def prepare_dataset(file_path: Path, time_cutoff: float, fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]], id_mps: dict[str, int]) -> dict[str, np.ndarray]:
    data = load_arrange_data(file_path, selected_ids=set(fixed_ids)).reset_index(names="Idx")
    if data.empty:
        raise RuntimeError(f"No selected IDs found in {file_path}")

    time_start = float(data["Time"].iloc[0])
    n_rows_to_use = data.loc[data["Time"] > time_start + time_cutoff, "Time"].shape[0]
    if n_rows_to_use <= 0:
        raise RuntimeError(f"No rows beyond time_cutoff for {file_path}")

    data_dict: dict[str, np.ndarray] = {}
    for can_id in fixed_ids:
        seq_data = get_repeated_sequences(data, can_id, id_to_signal_cols[can_id], id_mps[can_id])
        data_dict[can_id] = seq_data[-n_rows_to_use:].copy()

    # Align all IDs to the same sample count so downstream indexing is safe.
    min_len = min(v.shape[0] for v in data_dict.values())
    if min_len <= 0:
        raise RuntimeError(f"No usable rows after preprocessing for {file_path}")
    if len({v.shape[0] for v in data_dict.values()}) > 1:
        data_dict = {k: v[-min_len:].copy() for k, v in data_dict.items()}

    return data_dict


def load_inputs(data_path: Path, time_cutoff: float, fixed_ids: list[str], id_to_signal_cols: dict[str, list[str]], id_mps: dict[str, int], shuffle: bool = True, seed: int = 0):
    x_dict = prepare_dataset(data_path, time_cutoff, fixed_ids, id_to_signal_cols, id_mps)
    if shuffle:
        np.random.seed(seed)
        n_samples = len(next(iter(x_dict.values())))
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        x_dict = {k: v[idx] for k, v in x_dict.items()}

    y = np.concatenate([x_dict[can_id][:, -1, :] for can_id in fixed_ids], axis=1)
    return x_dict, y


def slice_data(file_path: Path, n_sliced: int, max_rows_per_slice: int = 120_000) -> list[Path]:
    if file_path.suffix != ".csv":
        raise ValueError("Memory-safe slicing currently supports CSV inputs only.")

    total_rows = _count_csv_rows(file_path)
    target_rows = max(1, math.ceil(total_rows / max(1, n_sliced)))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CANet on CrySyS raw logs")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="CrySyS dataset root")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory to save model weights")
    parser.add_argument("--selection-config", default=str(DEFAULT_SELECTION_CONFIG), help="JSON with selected signal list")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory to cache parsed CrySyS logs")
    parser.add_argument("--signal-mask-path", default=str(DEFAULT_SIGNAL_MASK_PATH), help="HDF5 signal mask path used for bit-level extraction")
    parser.add_argument("--id-mps-source", choices=["fixed", "estimate"], default="fixed")
    parser.add_argument("--id-config", default=str(DEFAULT_ID_CONFIG), help="Path to frozen id_mps JSON config")
    parser.add_argument("--write-id-config", action="store_true", help="Write estimated id_mps to --id-config")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--hidden-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--train-slices", type=int, default=10)
    parser.add_argument("--valid-slices", type=int, default=1)
    parser.add_argument("--valid-prop", type=float, default=0.05)
    parser.add_argument("--max-rows-per-slice", type=int, default=120_000)
    parser.add_argument(
        "--debug-plot",
        metavar="LOG_FILE",
        default=None,
        help=(
            "Path to a single CrySyS .log file to parse and plot for visual inspection. "
            "Saves a PNG next to the cached CSV and then exits. "
            "Use 'first_malicious' to auto-select the first malicious log in --dataset-dir."
        ),
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    id_config = Path(args.id_config).resolve()
    signal_mask_path = Path(args.signal_mask_path).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    id_to_signal_indices = parse_selected_signals(Path(args.selection_config).resolve())
    id_to_signal_cols = signal_columns_from_indices(id_to_signal_indices)
    max_signal_index = max(idx for idxs in id_to_signal_indices.values() for idx in idxs)
    signal_mask_map = load_signal_mask_map(signal_mask_path, id_to_signal_indices)

    if args.debug_plot is not None:
        fixed_ids_dp = sorted(id_to_signal_indices.keys(), key=lambda x: int(x, 16))
        if args.debug_plot == "first_malicious":
            _, malicious_logs = crysys_files(dataset_dir)
            target_log = malicious_logs[0]
        else:
            target_log = Path(args.debug_plot).resolve()
            if not target_log.exists():
                raise FileNotFoundError(f"--debug-plot log not found: {target_log}")
        scenario = target_log.parent.name
        cache_csv = cache_dir / scenario / f"{target_log.stem}.csv"
        expected_signal_cols = {f"Signal{idx + 1}" for idxs in id_to_signal_indices.values() for idx in idxs}
        if not _cache_looks_like_mask_signals(cache_csv, expected_signal_cols):
            parse_crysys_log(
                raw_log=target_log,
                selected_ids=set(fixed_ids_dp),
                cache_csv=cache_csv,
                id_to_signal_indices=id_to_signal_indices,
                signal_mask_map=signal_mask_map,
                max_signal_index=max_signal_index,
            )
        metadata_json = resolve_metadata_json(target_log)
        default_label = 1 if "-malicious-" in target_log.name else 0
        attack_intervals = parse_attack_intervals(metadata_json, default_label=default_label)
        print(f"[debug-plot] log        : {target_log.name}")
        print(f"[debug-plot] cache_csv  : {cache_csv}")
        print(f"[debug-plot] attack_intervals: {attack_intervals}")
        plot_output = cache_csv.with_suffix(".debug_plot.png")
        plot_trace_csv(cache_csv, attack_intervals, plot_output)
        sys.exit(0)

    fixed_ids = sorted(id_to_signal_indices.keys(), key=lambda x: int(x, 16))
    id_nsig = OrderedDict((i, len(id_to_signal_cols[i])) for i in fixed_ids)

    print("python", sys.executable)
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible to PyTorch.")
    device = torch.device("cuda")
    print("gpu_name", torch.cuda.get_device_name(device))

    benign_logs, _ = crysys_files(dataset_dir)
    benign_csvs = prepare_parsed_logs(
        benign_logs,
        selected_ids=set(fixed_ids),
        cache_root=cache_dir,
        id_to_signal_indices=id_to_signal_indices,
        signal_mask_map=signal_mask_map,
        max_signal_index=max_signal_index,
    )
    assert_only_benign_labels(benign_csvs)

    split_cache = cache_dir / "canet_cache_splits"
    train_files, valid_files = build_ambient_splits(benign_csvs, cache_dir=split_cache, valid_prop=args.valid_prop)

    id_mps = resolve_id_mps(
        train_files=train_files,
        fixed_ids=fixed_ids,
        id_mps_source=args.id_mps_source,
        id_config=id_config,
        write_id_config=args.write_id_config,
    )
    print(f"selected_ids={len(fixed_ids)} total_selected_signals={sum(id_nsig.values())}")
    print("id_mps", id_mps)

    model = CANetTorch(args.hidden_size, fixed_ids=fixed_ids, id_nsig=id_nsig).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss(reduction="mean")

    train_start = args.window_size + 1
    starttime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for epoch in range(args.epochs):
        print(f"********** Epoch {epoch + 1} **********")

        for data_file in train_files:
            sliced_files = slice_data(data_file, args.train_slices, max_rows_per_slice=args.max_rows_per_slice)
            for sliced_file in sliced_files:
                try:
                    x_train, y_train = load_inputs(sliced_file, train_start, fixed_ids, id_to_signal_cols, id_mps, shuffle=True, seed=epoch)
                except RuntimeError as exc:
                    if "No rows beyond time_cutoff" in str(exc):
                        print(f"Skipping short train slice: {sliced_file.name}")
                        continue
                    raise
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
                try:
                    x_valid, y_valid = load_inputs(sliced_file, train_start, fixed_ids, id_to_signal_cols, id_mps, shuffle=False)
                except RuntimeError as exc:
                    if "No rows beyond time_cutoff" in str(exc):
                        print(f"Skipping short valid slice: {sliced_file.name}")
                        continue
                    raise
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
        weight_name = model_dir / f"CrySyS_{starttime}_epoch{epoch + 1:02d}"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden_size": args.hidden_size,
                "fixed_ids": fixed_ids,
                "id_nsig": dict(id_nsig),
                "id_mps": id_mps,
                "id_to_signal_cols": id_to_signal_cols,
                "valid_prop": args.valid_prop,
                "id_mps_source": args.id_mps_source,
                "id_config": str(id_config),
                "cache_dir": str(cache_dir),
            },
            weight_name,
        )
        print(f"Saved weights: {weight_name}")


if __name__ == "__main__":
    main()

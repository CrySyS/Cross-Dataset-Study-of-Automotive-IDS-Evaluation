from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple, List, Literal
import re

import numpy as np
import pandas as pd


LINE_RE = re.compile(r"^\((?P<ts>[0-9]+(?:\.[0-9]+)?)\)\s+(?P<iface>\S+)\s+(?P<id>[0-9A-Fa-f]+)#(?P<payload>[0-9A-Fa-f]*)\s*$")


ATTACK_TYPE_MAP: Dict[str, str] = {
    "diagnostic": "Diagnostic",
    "dosattack": "DoS",
    "fuzzing_canid": "FuzzyCANID",
    "fuzzing_payload": "FuzzyPayload",
    "replay": "Replay",
    "suspension": "Suspension",
    "spoofing_speedometer": "SpoofingSpeedometer",
}


# Attack windows from the per-vehicle README files.
# Windows are inclusive and expressed in the same timestamp origin as the source logs.
ATTACK_WINDOWS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("OpelAstra", "dosattack"): (1536574995.000091, 1536575004.999811),
    ("Prototype", "dosattack"): (1531471730.001003, 1531471740.000841),
    ("RenaultClio", "dosattack"): (1508687506.000236, 1508687515.999845),
    ("OpelAstra", "replay"): (1536575013.172200, 1536575013.247372),
    ("RenaultClio", "replay"): (1508687499.839714, 1508687499.905626),
    ("OpelAstra", "suspension"): (1536575000.000097, 1536575010.000001),
    ("Prototype", "suspension"): (1531471729.986810, 1531471740.003056),
    ("RenaultClio", "suspension"): (1508687499.999696, 1508687510.000100),
    ("Prototype", "spoofing_speedometer"): (1531321812.221116, 1531321822.214643),
}


FUZZING_PAYLOAD_ID_BY_VEHICLE: Dict[str, int] = {
    "OpelAstra": 0x0C9,
    "Prototype": 0x5A0,
    "RenaultClio": 0x18A,
}


def infer_vehicle(path_in: Path) -> Optional[str]:
    if path_in.parent.name in {"OpelAstra", "Prototype", "RenaultClio"}:
        return path_in.parent.name
    for part in path_in.parts:
        if part in {"OpelAstra", "Prototype", "RenaultClio"}:
            return part
    return None


def infer_attack_type(path_in: Path) -> Optional[str]:
    stem = path_in.stem.lower()
    return ATTACK_TYPE_MAP.get(stem)


def infer_split(path_in: Path) -> Optional[str]:
    stem = path_in.stem.lower()
    if stem in {"training", "data_capture_70"}:
        return "train"
    if stem in {"testing", "data_capture_30"}:
        return "test"
    if stem == "full_data_capture":
        return "full"
    if stem in ATTACK_TYPE_MAP:
        return "test"
    return None


def parse_log_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None

    match = LINE_RE.match(line)
    if not match:
        return None

    ts = float(match.group("ts"))
    can_id_hex = match.group("id")
    payload_hex = match.group("payload") or ""

    can_id = int(can_id_hex, 16)
    if len(payload_hex) % 2 != 0:
        payload_hex = payload_hex + "0"

    payload_bytes = [int(payload_hex[i : i + 2], 16) for i in range(0, len(payload_hex), 2)]
    dlc = min(len(payload_bytes), 8)

    padded = [0] * 8
    for idx, byte in enumerate(payload_bytes[:8]):
        padded[idx] = byte

    return {
        "timestamp": ts,
        "can_id": can_id,
        "dlc": dlc,
        "data_hex": payload_hex.upper(),
        "data0": padded[0],
        "data1": padded[1],
        "data2": padded[2],
        "data3": padded[3],
        "data4": padded[4],
        "data5": padded[5],
        "data6": padded[6],
        "data7": padded[7],
    }


def load_raw(path_in: Path) -> pd.DataFrame:
    rows: List[dict] = []
    bad_lines = 0

    with open(path_in, "r", errors="ignore") as handle:
        for line in handle:
            parsed = parse_log_line(line)
            if parsed is None:
                bad_lines += 1
                continue
            rows.append(parsed)

    if not rows:
        raise ValueError(f"No parseable CAN lines found in {path_in}")

    if bad_lines > 0:
        print(f"[can_intrusion_dataset_v2] Warning: skipped {bad_lines} unparsable lines in {path_in}")

    return pd.DataFrame(rows)


def _label_attack_rows(df: pd.DataFrame, *, vehicle: Optional[str], file_stem: str, attack_type: Optional[str]) -> pd.Series:
    n = len(df)
    if attack_type is None:
        return pd.Series(np.zeros(n, dtype=np.int8), index=df.index)

    # Prefer timestamp windows when available (for period-based attacks, including suspension).
    if vehicle is not None:
        win = ATTACK_WINDOWS.get((vehicle, file_stem))
        if win is not None:
            start_ts, end_ts = win
            labels = ((df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)).astype(np.int8)
            return labels

    # Fallback patterns for attacks without explicit windows.
    if file_stem == "diagnostic":
        return ((df["can_id"] >= 0x700) & (df["can_id"] <= 0x7FF)).astype(np.int8)

    if file_stem == "fuzzing_canid":
        return df["can_id"].isin([0x111, 0x222, 0x333, 0x444]).astype(np.int8)

    if file_stem == "fuzzing_payload":
        target_id = FUZZING_PAYLOAD_ID_BY_VEHICLE.get(vehicle or "")
        if target_id is None:
            target_mask = pd.Series(np.ones(n, dtype=bool), index=df.index)
        else:
            target_mask = df["can_id"] == target_id
        return (target_mask & (df["data_hex"] == "FFFFFFFFFFFFFFFF")).astype(np.int8)

    # If the file is attack-tagged but no rule matched, consider the whole file under attack.
    return pd.Series(np.ones(n, dtype=np.int8), index=df.index)


def to_canonical(
    df_raw: pd.DataFrame,
    *,
    dataset_name: str,
    vehicle: Optional[str],
    split: Optional[str],
    attack_type: Optional[str],
    file_stem: str,
) -> pd.DataFrame:
    n = len(df_raw)
    labels = _label_attack_rows(df_raw, vehicle=vehicle, file_stem=file_stem, attack_type=attack_type)

    out = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(df_raw["timestamp"], errors="coerce").astype(float),
            "can_id": pd.to_numeric(df_raw["can_id"], errors="coerce").fillna(0).astype(np.int64),
            "dlc": pd.to_numeric(df_raw["dlc"], errors="coerce").fillna(0).clip(0, 8).astype(np.int8),
            "label": labels.astype(np.int8),
            "attack_type": attack_type,
            "dataset": dataset_name,
            "frame_type": None,
            "vehicle": vehicle,
            "split": split,
            "idx_src": np.arange(n, dtype=np.int64),
            "inter_arrival": pd.to_numeric(df_raw["timestamp"], errors="coerce").astype(float).diff().fillna(0.0).astype(float),
            "data0": pd.to_numeric(df_raw["data0"], errors="coerce").fillna(0).clip(0, 255).astype(np.uint8),
            "data1": pd.to_numeric(df_raw["data1"], errors="coerce").fillna(0).clip(0, 255).astype(np.uint8),
            "data2": pd.to_numeric(df_raw["data2"], errors="coerce").fillna(0).clip(0, 255).astype(np.uint8),
            "data3": pd.to_numeric(df_raw["data3"], errors="coerce").fillna(0).clip(0, 255).astype(np.uint8),
            "data4": pd.to_numeric(df_raw["data4"], errors="coerce").fillna(0).clip(0, 255).astype(np.uint8),
            "data5": pd.to_numeric(df_raw["data5"], errors="coerce").fillna(0).clip(0, 255).astype(np.uint8),
            "data6": pd.to_numeric(df_raw["data6"], errors="coerce").fillna(0).clip(0, 255).astype(np.uint8),
            "data7": pd.to_numeric(df_raw["data7"], errors="coerce").fillna(0).clip(0, 255).astype(np.uint8),
        }
    )

    # Keep attack_type only on attack rows.
    if attack_type is not None:
        out.loc[out["label"] == 0, "attack_type"] = None

    return out


def convert_file(
    path_in: str | Path,
    path_out: str | Path | None = None,
    *,
    dataset_name: str = "CAN Intrusion Dataset v2",
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = "snappy",
    overwrite: bool = True,
) -> str:
    p_in = Path(path_in)
    if not p_in.exists():
        raise FileNotFoundError(f"Input file not found: {p_in}")

    df_raw = load_raw(p_in)
    vehicle = infer_vehicle(p_in)
    file_stem = p_in.stem.lower()
    split = infer_split(p_in)
    attack_type = infer_attack_type(p_in)

    canon = to_canonical(
        df_raw,
        dataset_name=dataset_name,
        vehicle=vehicle,
        split=split,
        attack_type=attack_type,
        file_stem=file_stem,
    )

    if path_out is None:
        p_out = p_in.with_suffix(".parquet")
    else:
        p_out = Path(path_out)
        if p_out.suffix == "":
            p_out = p_out / f"{p_in.stem}.parquet"

    p_out.parent.mkdir(parents=True, exist_ok=True)

    if p_out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite=False: {p_out}")

    canon.to_parquet(p_out, index=False, compression=compression)
    return str(p_out)


def convert_folder(
    root_in: str | Path,
    out_dir: str | Path | None = None,
    *,
    glob_pattern: str = "*.log",
    dataset_name: str = "CAN Intrusion Dataset v2",
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = "snappy",
    overwrite: bool = True,
) -> list[str]:
    root = Path(root_in)
    if not root.exists():
        return []

    outputs: list[str] = []

    for path_in in sorted(root.rglob(glob_pattern)):
        if not path_in.is_file():
            continue

        if out_dir is None:
            path_out = None
        else:
            rel = path_in.relative_to(root)
            path_out = Path(out_dir) / rel.parent / f"{path_in.stem}.parquet"

        out = convert_file(
            path_in,
            path_out,
            dataset_name=dataset_name,
            compression=compression,
            overwrite=overwrite,
        )
        outputs.append(out)

    return outputs

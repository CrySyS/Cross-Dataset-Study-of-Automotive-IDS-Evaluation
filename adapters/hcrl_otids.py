from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, List, Dict
import re
import numpy as np
import pandas as pd

# ---- Public API -------------------------------------------------------------

def convert_file(
    path_in: str | Path,
    path_out: str | Path | None = None,
    *,
    dataset_name: str = "HCRL-OTIDS",
    attack_type: str | None = None,
    keep_raw_line: bool = False,
    compression: str = "snappy",
    overwrite: bool = True,
) -> str:
    """
    Read a single OTIDS raw text file and write canonical Parquet. Return output path.

    - attack_type is inferred from filename when not provided.
    - Preserves all useful raw information (hex ID, flags/RTR, payload, raw line).
    """
    p_in = Path(path_in)
    if not p_in.exists():
        raise FileNotFoundError(f"Input file not found: {p_in}")

    atk = attack_type or infer_attack_type(p_in)
    df_raw = load_raw(p_in)  # parsed & sorted by timestamp
    canon = to_canonical(df_raw, dataset_name=dataset_name, attack_type=atk, keep_raw_line=keep_raw_line)

    if path_out is None:
        p_out = p_in.with_suffix(".parquet")
    else:
        p_out = Path(path_out)
        if p_out.suffix == "":  # treat as directory if no suffix
            p_out = p_out / (p_in.stem + ".parquet")
    p_out.parent.mkdir(parents=True, exist_ok=True)

    if p_out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite=False: {p_out}")

    canon.to_parquet(p_out, index=False, compression=compression)
    return str(p_out)


def convert_folder(
    root_in: str | Path,
    out_dir: str | Path | None = None,
    *,
    glob: str = "*.txt",
    **kwargs,
) -> List[str]:
    """
    Batch convert all OTIDS files under a folder. Returns list of output paths.
    """
    root = Path(root_in)
    paths = sorted(root.rglob(glob))
    outs: List[str] = []
    for p in paths:
        if out_dir is None:
            out_path: str | Path | None = None
        else:
            out_path = Path(out_dir)
        outs.append(convert_file(p, out_path, **kwargs))
    return outs


# ---- Parsing ----------------------------------------------------------------

# Example lines:
# Timestamp:          0.000462        ID: 0080    000    DLC: 8    00 17 ea 0a 20 1a 20 43
# Timestamp:          0.001684        ID: 0153    100    DLC: 0
LINE_RE = re.compile(
    r"Timestamp:\s*(?P<ts>\d+(?:\.\d+)?)\s+"
    r"ID:\s*(?P<id_hex>[0-9a-fA-F]+)\s+"
    r"(?P<flags>[01]{3})\s+"
    r"DLC:\s*(?P<dlc>\d)"
    r"(?:\s+(?P<data>(?:[0-9a-fA-F]{2}\s*){1,8}))?\s*$"
)

def infer_attack_type(path: Path) -> str:
    """Infer attack type from filename."""
    name = path.stem.lower()
    if "dos" in name:
        return "DoS"
    if "fuzzy" in name:
        return "Fuzzy"
    if "impersonation" in name or "imp" in name:
        return "Impersonation"
    if "normal" in name or "attackfree" in name or "attack_free" in name:
        return "AttackFree"
    # Fallback: treat unknown as AttackFree (caller may override)
    return "AttackFree"


def _parse_line(line: str) -> Optional[Dict]:
    m = LINE_RE.search(line)
    if not m:
        return None
    ts = float(m["ts"])
    id_hex = m["id_hex"].lower().zfill(4)
    can_id = int(id_hex, 16)
    flags = m["flags"]
    # In OTIDS, '100' indicates RTR (remote frame request). We expose both flags and boolean.
    is_rtr = flags.startswith("1")
    dlc = int(m["dlc"])
    data_str = (m["data"] or "").strip()
    data_bytes = [] if not data_str else [int(b, 16) for b in data_str.split()]
    return {
        "timestamp": ts,
        "can_id_hex": id_hex,
        "can_id": can_id,
        "flags": flags,           # '000'/'100'
        "is_rtr": is_rtr,         # True for remote request
        "dlc": dlc,
        "data_len": len(data_bytes),
        "data_bytes": data_bytes, # variable-length list (0..8)
        "data_hex": "".join(f"{b:02x}" for b in data_bytes),
        "raw_line": line.rstrip("\n"),
    }


def load_raw(path: Path) -> pd.DataFrame:
    """
    Parse an OTIDS raw text file into a minimally processed DataFrame.
    Sorted by timestamp (stable).
    """
    rows: List[Dict] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rec = _parse_line(s)
            if rec is not None:
                rows.append(rec)

    if not rows:
        raise ValueError(f"OTIDS parser could not parse any rows from {path}")

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df


# ---- Canonicalization -------------------------------------------------------

def _label_rows(df: pd.DataFrame, attack_type: str) -> pd.Series:
    """
    Per OTIDS documentation:
      - DoS: only ID 0x000 are abnormal (label=1), others normal.
      - Fuzzy & Impersonation: ts < 250s -> normal; ts >= 250s -> under attack (label=1).
      - AttackFree: all normal.
    """
    if attack_type == "DoS":
        return (df["can_id"] == 0x000).astype("int8")
    elif attack_type in {"Fuzzy", "Impersonation"}:
        return (df["timestamp"] >= 250.0).astype("int8")
    else:
        return np.zeros(len(df), dtype="int8")


def to_canonical(
    df: pd.DataFrame,
    *,
    dataset_name: str = "HCRL-OTIDS",
    attack_type: str = "AttackFree",
    keep_raw_line: bool = True,
) -> pd.DataFrame:
    """
    Convert parsed OTIDS DF to unified frame-space schema (lossless).
    Emits fixed-width payload columns while preserving exact payload & length.
    """

    # Inter-arrival (seconds)
    inter = df["timestamp"].diff().fillna(0.0).astype("float64")

    # Prepare fixed-width payload without losing true length
    n = len(df)
    padded = np.zeros((n, 8), dtype=np.uint8)
    for i, bytes_list in enumerate(df["data_bytes"].tolist()):
        if bytes_list:
            # truncate to 8 bytes just in case (defensive)
            padded[i, : len(bytes_list[:8])] = bytes_list[:8]

    # Labels by dataset rules
    label = _label_rows(df, attack_type=attack_type)

    # frame_type: reflect RTR frames when present; else 'normal'
    frame_type = np.where(df["is_rtr"].to_numpy(), "remote_req", "normal")

    out = pd.DataFrame({
        # time
        "timestamp": df["timestamp"].astype("float64"),
        "timestamp_ns": (df["timestamp"] * 1e9).round().astype("int64"),

        # id
        "can_id": df["can_id"].astype("int64"),
        "can_id_hex": df["can_id_hex"].astype("string"),

        # flags / rtr
        "flags": df["flags"].astype("string"),
        "is_rtr": df["is_rtr"].astype("bool"),
        "frame_type": pd.Series(frame_type, dtype="string"),

        # payload
        "dlc": df["dlc"].astype("int8"),
        "data_len": df["data_len"].astype("int8"),
        "data_hex": df["data_hex"].astype("string"),

        # labels & meta
        "label": pd.Series(label, dtype="int8"),
        "attack_type": pd.Series([attack_type] * n, dtype="string"),
        "dataset": pd.Series([dataset_name] * n, dtype="string"),

        # convenience / provenance
        "inter_arrival": inter,
        "idx_src": np.arange(n, dtype="int64"),
    })
    
    # CRITICAL FIX: Benign messages (label=0) should NOT have attack_type set.
    # attack_type should only be set for attack messages (label=1).
    # The inferred attack_type from filename applies only to the attack class.
    if attack_type is not None:
        out.loc[out["label"] == 0, "attack_type"] = None

    # materialize data0..data7
    for k in range(8):
        out[f"data{k}"] = padded[:, k].astype("uint8")

    # Optional: keep raw line for perfect provenance
    if keep_raw_line and "raw_line" in df.columns:
        out["raw_line"] = df["raw_line"].astype("string")

    # ---- Lightweight validations (non-fatal warnings) ----------------------
    _validate_otids_consistency(out)

    # Column order (nice-to-have; stable for parquet schema)
    col_order = [
        "timestamp", "timestamp_ns",
        "can_id", "can_id_hex",
        "flags", "is_rtr", "frame_type",
        "dlc", "data_len",
        "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7",
        "data_hex",
        "label", "attack_type", "dataset",
        "inter_arrival", "idx_src",
        "raw_line",
    ]
    out = out[[c for c in col_order if c in out.columns]]

    return out


# ---- Helpers ----------------------------------------------------------------

def _validate_otids_consistency(df: pd.DataFrame) -> None:
    """
    Print soft warnings for common inconsistencies. Never raises.
    """
    try:
        # DLC should match data_len for data frames; for RTR frames, DLC may be 0
        mask_data = ~df["is_rtr"]
        mism = (df.loc[mask_data, "dlc"] != df.loc[mask_data, "data_len"])
        if mism.any():
            cnt = int(mism.sum())
            print(f"[OTIDS] Warning: {cnt} rows where DLC != data_len (non-RTR).")

        # DoS rule sanity: if attack_type==DoS, label==1 only for can_id==0
        if "attack_type" in df.columns and "label" in df.columns:
            if (df["attack_type"] == "DoS").any():
                bad = (df["attack_type"] == "DoS") & (df["label"] == 1) & (df["can_id"] != 0)
                if bad.any():
                    print("[OTIDS] Warning: Found rows labeled attack in DoS but can_id != 0x000.")
    except Exception as e:
        print(f"[OTIDS] Validation skipped due to error: {e}")

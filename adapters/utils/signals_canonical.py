
import pandas as pd
import numpy as np
from typing import Optional, Dict, Sequence

def compute_inter_arrival_per_group(ts: pd.Series, key: pd.Series) -> pd.Series:
    # ts is seconds (float), key is grouping key (e.g., can_id)
    return ts.groupby(key).diff().fillna(0.0).astype(float)

def parse_can_id(id_val: object) -> int:
    """Convert CAN ID from string format (e.g., 'id5') to integer (5)."""
    if isinstance(id_val, (int, np.integer)):
        return int(id_val)
    s = str(id_val).strip().lower()
    if s.startswith("id"):
        s = s[2:]
    if s.isdigit():
        return int(s)
    # Fallback: try to parse as hex or return hash
    try:
        return int(s, 16)
    except ValueError:
        return abs(hash(id_val)) & 0xFFFFFFFF

def build_signals_canonical(
    df: pd.DataFrame,
    colmap: Dict[str, object],
    *,
    dataset_name: str,
    split: Optional[str],
    attack_type: Optional[str],
    expected_signal_counts: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """
    Build canonical signal-space DataFrame with unified column names.
    
    Uses same column names as frame-space for consistency:
    - timestamp (not timestamp_s)
    - can_id (not id_str) 
    - inter_arrival (not inter_arrival_s)
    
    colmap keys:
      - time_ms: str (column name with time in milliseconds)
      - id: str (column name with CAN ID, e.g., "id5")
      - label: str (column name with label 0/1)
      - signals: Sequence[str] (e.g., ["Signal1_of_ID", ... "Signal4_of_ID"])
    """
    ts = pd.to_numeric(df[colmap["time_ms"]], errors="coerce").astype(float) / 1000.0
    can_id_raw = df[colmap["id"]]
    can_id = can_id_raw.apply(parse_can_id).astype(np.int64)
    label = pd.to_numeric(df[colmap["label"]], errors="coerce").fillna(0).astype(np.int8)

    out = pd.DataFrame({
        "timestamp": ts,
        "can_id": can_id,
        "label": label,
        "dataset": dataset_name,
        "attack_type": attack_type,
        "split": split,
        "idx_src": np.arange(len(df), dtype=np.int64),
    })

    # attach signals as float columns in stable order
    for i, sig_col in enumerate(colmap["signals"], start=1):
        out[f"signal{i}"] = pd.to_numeric(df[sig_col], errors="coerce").astype(float)

    out["inter_arrival"] = compute_inter_arrival_per_group(out["timestamp"], out["can_id"])

    if expected_signal_counts is not None:
        out["n_signals_expected"] = out["can_id"].map(expected_signal_counts).astype("Int64")

    # dtype hardening
    return out.astype({
        "timestamp": float,
        "can_id": np.int64,
        "label": np.int8,
        "idx_src": np.int64,
        "inter_arrival": float,
    }, copy=False)

# utils/can_parse.py
import numpy as np
import pandas as pd

def parse_can_id(val) -> int:
    if pd.isna(val):
        raise ValueError("Empty CAN ID")
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    try:
        return int(s, 16)
    except ValueError:
        return int(s, 10)

def sanitize_dlc(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int16).clip(0, 8).astype(np.int8)

def coerce_byte(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int64)
    return vals.clip(0, 255).astype(np.uint8)

def zero_beyond_dlc(data_arr: np.ndarray, dlc: pd.Series) -> np.ndarray:
    # data_arr: (n, 8), dlc: int8
    for i in range(8):
        mask = (dlc <= i)
        if mask.any():
            data_arr[mask.values, i] = 0
    return data_arr

def compute_inter_arrival(ts: pd.Series) -> pd.Series:
    ts = pd.to_numeric(ts, errors="coerce").astype(float)
    return ts.diff().fillna(0.0).astype(float)

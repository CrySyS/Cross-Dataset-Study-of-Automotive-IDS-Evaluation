# utils/can_canonical.py
import numpy as np
import pandas as pd
from .can_parse import parse_can_id, sanitize_dlc, coerce_byte, zero_beyond_dlc, compute_inter_arrival

def build_canonical(
    df: pd.DataFrame,
    colmap: dict,
    *,
    dataset_name: str,
    attack_type: str | None,
    label_mapper: dict | None = None,  # e.g. {"T":1,"R":0}
) -> pd.DataFrame:
    # Required column keys in colmap: "timestamp", "can_id", "dlc", "flag", "data" (list of 8 names or indices)
    ts = pd.to_numeric(df[colmap["timestamp"]], errors="coerce").astype(float)
    can_ids = df[colmap["can_id"]].map(parse_can_id).astype(np.int64)
    dlc = sanitize_dlc(df[colmap["dlc"]])

    # bytes
    n = len(df)
    data_arr = np.zeros((n, 8), dtype=np.uint8)
    for i, src in enumerate(colmap["data"]):
        if src is None:
            continue
        data_arr[:, i] = coerce_byte(df[src])

    data_arr = zero_beyond_dlc(data_arr, dlc)

    # labels
    if colmap.get("flag") is not None and label_mapper is not None:
        label = (
            df[colmap["flag"]].astype(str).str.strip().str.upper().map(label_mapper).fillna(0).astype(np.int8)
        )
    else:
        label = pd.Series(np.zeros(n, dtype=np.int8))

    inter = compute_inter_arrival(ts)

    out = pd.DataFrame({
        "timestamp": ts,
        "can_id": can_ids,
        "dlc": dlc,
        "label": label,
        "attack_type": attack_type,
        "dataset": dataset_name,
        "frame_type": None,
        "vehicle": None,
        "split": None,
        "idx_src": np.arange(n, dtype=np.int64),
        "inter_arrival": inter,
    })
    for i in range(8):
        out[f"data{i}"] = data_arr[:, i]
    # dtype hardening
    return out.astype({
        "timestamp": float, "can_id": np.int64, "dlc": np.int8, "label": np.int8, "idx_src": np.int64, "inter_arrival": float
    }, copy=False)

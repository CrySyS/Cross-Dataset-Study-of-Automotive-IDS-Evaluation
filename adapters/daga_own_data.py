# adapters/daga_adapter.py
from __future__ import annotations
import io
import re
import zipfile
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# canonical column set we will produce (at minimum)
CANONICAL_COLS = [
    "timestamp","can_id","dlc",
    "label","attack_type","dataset","frame_type","vehicle","split","idx_src",
    "inter_arrival",
    "data0","data1","data2","data3","data4","data5","data6","data7"
]
# include trace_name for per-trace grouping by consumers
CANONICAL_PLUS = CANONICAL_COLS + ["trace_name"]

# ---- Robust parsers ---------------------------------------------------------

def _hex_to_int(hex_str: str) -> Optional[int]:
    if pd.isna(hex_str):
        return None
    s = str(hex_str).strip()
    if s == "":
        return None

    # Some DAGA traces contain CAN_ID values serialized like "290.0".
    # Interpret these as their original integer token before hex parsing.
    m = re.fullmatch(r"([0-9A-Fa-f]+)\.0+", s)
    if m:
        s = m.group(1)

    if s.startswith("0x") or s.startswith("0X"):
        s = s[2:]
    if s == "":
        return None
    try:
        return int(s, 16)
    except ValueError:
        return None

def _payload_to_bytes(payload_hex: str, dlc: Optional[int]) -> List[int]:
    """Return 8 integers (0..255); pad with 0."""
    if payload_hex is None or payload_hex == "" or (isinstance(payload_hex, float) and pd.isna(payload_hex)):
        return [0]*8
    s = str(payload_hex).strip()
    if s.startswith(("0x", "0X")):
        s = s[2:]
    if len(s) % 2 == 1:
        s = "0" + s
    b = []
    for i in range(0, min(16, len(s)), 2):
        try:
            b.append(int(s[i:i+2], 16))
        except ValueError:
            b.append(0)
    while len(b) < 8:
        b.append(0)
    return b

def _read_csv_from_path(path: Path) -> pd.DataFrame:
    """
    Read DAGA files (CSV-formatted but may use .txt) and .zip containers.
    """
    def _read_like_csv(fobj_or_path):
        return pd.read_csv(fobj_or_path)  # comma-separated with header

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            names = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
            if not names:
                raise RuntimeError(f"No CSV/TXT in zip {path}")
            with z.open(names[0]) as fh:
                return _read_like_csv(fh)
    else:
        return _read_like_csv(path)

# ---- Main converter ----------------------------------------------------------

def convert_daga_dataset(
    root_dir: str,
    out_dir: str,
    dataset_name: str = "daga",
    vehicle_name: str = "volvo_v40_2016",
    include_payload_fuzzing: bool = False,
):
    """
    Walk the DAGA folder tree. Expect root_dir/
      - clean/   (7 traces)
      - infected/DenialOfService/...
      - infected/OrderedSequenceReplay/...
      - infected/SingleIDReplay/...
      - infected/ArbitrarySequenceReplay/...
      - infected/MessageIDFuzzing/...
    Writes one parquet file per trace to out_dir/<split>/*.parquet
    """
    root = Path(root_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Converting DAGA dataset from {root} to {out}")

    def process_trace(df_raw: pd.DataFrame, trace_stem: str, split: str, attack_type: str) -> pd.DataFrame:
        # --- column selection (robust to case/spacing) ---
        cols_lut: Dict[str, str] = {c.lower(): c for c in df_raw.columns}

        # timestamp
        ts_col = cols_lut.get("timestamp", None) or df_raw.columns[0]

        # CAN_ID
        id_candidates = ["can_id", "can id", "canid", "id", "message_id", "arbitration_id"]
        id_col = None
        for k in id_candidates:
            if k in cols_lut:
                id_col = cols_lut[k]
                break
        if id_col is None and "CAN_ID" in df_raw.columns:
            id_col = "CAN_ID"
        if id_col is None:
            raise RuntimeError(f"{trace_stem}: Cannot find CAN ID column in {list(df_raw.columns)}")

        # DLC
        dlc_col = cols_lut.get("dlc", None)

        # payload hex
        payload_col = cols_lut.get("payload_hex", None) or ("PAYLOAD_HEX" if "PAYLOAD_HEX" in df_raw.columns else None)

        # anomaly / label
        anomaly_col = cols_lut.get("anomaly", None) or ("ANOMALY" if "ANOMALY" in df_raw.columns else None)

        # --- build canonical frame df ---
        # Parse IDs first, then keep all downstream columns aligned to kept rows.
        raw_ids = df_raw[id_col].astype(str)
        parsed_ids = raw_ids.map(_hex_to_int)

        # --- determine ID base for this trace (hex vs decimal) ---
        id_series = raw_ids.str.strip()
        has_hex_prefix  = id_series.str.startswith(("0x", "0X")).any()
        has_hex_letters = id_series.str.contains(r"[A-Fa-f]").any()
        assume_hex = bool(has_hex_prefix or has_hex_letters)  # else assume decimal

        nan_count = int(parsed_ids.isna().sum())
        total = len(parsed_ids)
        if nan_count:
            frac = nan_count / max(total, 1)
            if frac > 0.005:
                sample_bad = df_raw.loc[parsed_ids.isna(), [ts_col, id_col]].head(5)
                raise RuntimeError(
                    f"{trace_stem}: CAN_ID parse failures {nan_count}/{total} ({frac:.2%}). "
                    f"Sample bad rows:\n{sample_bad.to_string(index=False)}"
                )

        valid_mask = parsed_ids.notna()
        df_raw_kept = df_raw.loc[valid_mask].copy().reset_index(drop=True)
        parsed_ids = parsed_ids.loc[valid_mask].astype("int64").reset_index(drop=True)

        df = pd.DataFrame()
        df["timestamp"] = pd.to_numeric(df_raw_kept[ts_col], errors="coerce").astype(float)
        df["can_id"] = parsed_ids

        if nan_count > 0:
            print(f"{trace_stem}: Warning: dropped {nan_count} rows with invalid CAN_ID.")
            print(f"{trace_stem}: Remaining rows: {len(df)}")


        # dlc (bits -> bytes if needed)
        if dlc_col is not None:
            dlc_raw = pd.to_numeric(df_raw_kept[dlc_col], errors="coerce")
            dlc_bytes = dlc_raw.copy()
            dlc_bytes.loc[dlc_bytes >= 16] = (dlc_bytes.loc[dlc_bytes >= 16] // 8)
            df["dlc"] = dlc_bytes.fillna(0).clip(lower=0, upper=8).astype("int8")
        else:
            df["dlc"] = np.int8(0)

        # label
        if anomaly_col is not None:
            s = df_raw_kept[anomaly_col]
            def map_label(v):
                if pd.isna(v): return 0
                vv = str(v).strip().lower()
                if vv in ("true","1","t","yes"): return 1
                if vv in ("false","0","f","no"): return 0
                try: return int(float(v))
                except Exception: return 0
            df["label"] = pd.Series([map_label(x) for x in s], dtype="int8")
        else:
            df["label"] = np.int8(0)

        # payload -> data0..data7  (pad with 0, cast to uint8)
        if payload_col is not None:
            payloads = df_raw_kept[payload_col].astype(str).tolist()
        else:
            payloads = [""] * len(df)
        dlcs = df["dlc"].tolist()
        bytes_list = [ _payload_to_bytes(h, d) for h, d in zip(payloads, dlcs) ]
        for i in range(8):
            df[f"data{i}"] = np.array([ int(b[i]) for b in bytes_list ], dtype="uint8")

        # fixed metadata
        df["attack_type"] = attack_type
        df["dataset"] = dataset_name
        df["frame_type"] = "standard"   # DAGA uses 11-bit IDs
        df["vehicle"] = vehicle_name
        df["split"] = split

        # idx_src: 0..N-1 within this trace, keep trace_name for grouping
        df["idx_src"] = np.arange(len(df), dtype=np.int64)
        df["trace_name"] = trace_stem

        # inter-arrival (per-trace)
        ts = df["timestamp"].to_numpy(dtype=float)
        df["inter_arrival"] = np.concatenate(([0.0], np.diff(ts))) if len(ts) > 1 else np.array([0.0])

        # ensure columns exist + order
        for c in CANONICAL_PLUS:
            if c not in df.columns:
                df[c] = pd.NA

        # optional sanity message for unexpected extended ranges
        if df["can_id"].notna().any():
            max_id = int(df["can_id"].max())
            if max_id > 0x7FF and df["frame_type"].iloc[0] == "standard":
                print(f"{trace_stem}: ⚠️ can_id up to 0x{max_id:X} after parsing. "
                      f"Detected base={'hex' if assume_hex else 'decimal'}. "
                      "If truly extended IDs, set frame_type='extended'.")

        return df[CANONICAL_PLUS]

    # --- iterate clean traces ---
    clean_dir = root / "clean"
    if clean_dir.exists():
        out_clean = out / "clean"
        out_clean.mkdir(parents=True, exist_ok=True)
        for src in sorted(clean_dir.iterdir()):
            if not src.is_file() or src.suffix == '.zip':
                continue
            print("Processing clean trace:", src)
            df_raw = _read_csv_from_path(src)
            trace_stem = src.stem
            df = process_trace(df_raw, trace_stem=trace_stem, split="clean", attack_type="clean")
            (out_clean / f"{trace_stem}.parquet").write_bytes(df.to_parquet(index=False))

    # --- iterate infected traces ---
    inf_dir = root / "infected"
    if inf_dir.exists():
        out_inf = out / "infected"
        out_inf.mkdir(parents=True, exist_ok=True)
        for attack_sub in sorted(inf_dir.iterdir()):
            if not attack_sub.is_dir():
                continue
            attack_type = attack_sub.name
            if attack_type == "PayloadFuzzing" and not include_payload_fuzzing:
                print("Skipping infected trace folder PayloadFuzzing (disabled by config).")
                continue
            for src in sorted(attack_sub.iterdir()):
                if not src.is_file() or src.suffix == '.zip':
                    continue
                print(f"Processing infected trace: {attack_type} / {src.name}")
                df_raw = _read_csv_from_path(src)
                trace_stem = f"{attack_type}__{src.stem}"
                df = process_trace(df_raw, trace_stem=trace_stem, split="infected", attack_type=attack_type)
                (out_inf / f"{trace_stem}.parquet").write_bytes(df.to_parquet(index=False))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to DAGA root (contains 'clean' and 'infected' folders)")
    p.add_argument("--out", required=True, help="Output parquet folder (e.g., data_parquet/daga)")
    p.add_argument(
        "--include_payload_fuzzing",
        action="store_true",
        help="Include PayloadFuzzing traces (disabled by default).",
    )
    args = p.parse_args()

    convert_daga_dataset(
        args.root,
        args.out,
        include_payload_fuzzing=args.include_payload_fuzzing,
    )

    # quick smoke checks (adjust a filename that exists in your output)
    # df = pd.read_parquet("data_parquet/daga/clean/V40_01.can.parquet")
    # assert df["dlc"].max() <= 8
    # assert df.filter(regex=r"^data[0-7]$").dtypes.eq("uint8").all()
    # assert df["idx_src"].iloc[0] == 0 and df["idx_src"].iloc[-1] == len(df)-1
    # assert "trace_name" in df.columns

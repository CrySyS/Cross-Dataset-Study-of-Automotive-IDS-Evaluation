# adapters/road.py
from __future__ import annotations

from pathlib import Path
import io
import json
import logging
import re
import zipfile
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# reuse shared utils
from .utils.can_parse import parse_can_id
from .utils.can_canonical import build_canonical
from .utils.signals_canonical import build_signals_canonical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Attack inference from filename ----------------

ATTACK_MAP = {
    "ambient": "Ambient",          # scenario: ambient driving
    "fuzz": "Fuzzing",
    "fuzzing": "Fuzzing",
    "fabrication": "Fabrication",
    "masquerade": "Masquerade",
    "correlated": "CorrelatedSignal",
    "correlated_signal": "CorrelatedSignal",
    "speedometer": "MaxSpeedometer",
    "coolant": "MaxCoolantTemp",
    "reverse": "ReverseLight",
    "accelerator": "Accelerator",
}


def infer_attack_type(path: Path) -> Optional[str]:
    """Infer a coarse attack/scenario type from filename / directory."""
    name = path.stem.lower()
    for k, v in ATTACK_MAP.items():
        if k in name:
            return v
    parts = [p.lower() for p in path.parts]
    if "ambient" in parts:
        return "Ambient"
    if "attacks" in parts:
        return "UnknownAttack"
    return None


# ---------------- Candump (raw) loader ----------------
# Candump line example: (1609072193.123456) can0 123#1122334455667788

_CANDUMP_RE = re.compile(
    r"\(\s*(?P<ts>[\d]+\.\d+)\s*\)\s+[a-zA-Z0-9_]+\s+(?P<id>[0-9A-Fa-fx]+)\#(?P<data>[0-9A-Fa-f]*)\s*$"
)


def _parse_candump_lines(lines: List[str]) -> pd.DataFrame:
    rows = []
    for line in lines:
        m = _CANDUMP_RE.search(line)
        if not m:
            continue
        ts = float(m.group("ts"))
        cid = parse_can_id(m.group("id"))
        data_hex = (m.group("data") or "").strip()

        # DLC from payload length (ROAD says full 8 bytes padded; be tolerant anyway)
        if len(data_hex) % 2 == 1:
            data_hex = "0" + data_hex
        dlc = min(len(data_hex) // 2, 8)

        # normalize to 8 bytes (16 hex chars), padding with 0
        data_hex = data_hex[:16].ljust(16, "0")
        b = [int(data_hex[i:i + 2], 16) for i in range(0, 16, 2)]
        rows.append((ts, cid, dlc, *b))

    cols = ["timestamp", "can_id", "dlc"] + [f"data{i}" for i in range(8)]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    # dtypes
    for i in range(8):
        df[f"data{i}"] = df[f"data{i}"].astype(np.uint8, copy=False)
    df["dlc"] = df["dlc"].astype(np.int8, copy=False)
    df["can_id"] = df["can_id"].astype(np.int64, copy=False)
    df["timestamp"] = df["timestamp"].astype(float, copy=False)
    return df


def _load_candump(path: Path) -> pd.DataFrame:
    logger.info("Reading candump raw: %s", path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return _parse_candump_lines(f.readlines())


# ---------------- Signal CSV (translated) loader ----------------
# ROAD readme: columns: Label, ID (DECIMAL), Time (seconds), Signal_i_of_ID ...

_SIG_HEADER_MIN = {"label", "id", "time"}


def _normalize_signal_header_token(s: str) -> str:
    s0 = str(s).strip()
    low = s0.lower().replace(" ", "")
    if low == "label":
        return "Label"
    if low == "id":
        return "ID"
    if low == "time":
        return "Time"
    # Signal_1_of_ID / Signal-1-of-ID / Signal1
    m = re.match(r"signal[-_]?(\d+)(?:[-_]of[-_]id)?$", low)
    if m:
        return f"Signal{m.group(1)}"
    return s0


def _read_signal_csv_flexible(file_like) -> pd.DataFrame:
    raw = pd.read_csv(
        file_like,
        header=None,            # detect header ourselves
        sep=None,               # auto-detect delimiter
        engine="python",
        skip_blank_lines=True,
        dtype=object,
    )
    if raw.empty:
        return pd.DataFrame(columns=["Label", "ID", "Time"])

    first = [str(x).strip().lower() for x in raw.iloc[0].tolist()]
    if _SIG_HEADER_MIN.issubset(first):
        # header present
        name_map = {i: _normalize_signal_header_token(v) for i, v in enumerate(raw.iloc[0].tolist())}
        df = raw.iloc[1:].rename(columns=name_map).reset_index(drop=True)
    else:
        # no header: assume first 3 columns Label, ID, Time
        df = raw.rename(columns={0: "Label", 1: "ID", 2: "Time"}).reset_index(drop=True)

    # normalize signal columns
    for c in list(df.columns):
        norm = _normalize_signal_header_token(c)
        if norm != c:
            df = df.rename(columns={c: norm})

    # keep only core + any SignalN columns
    sig_cols = [c for c in df.columns if str(c).startswith("Signal") and str(c)[6:].isdigit()]
    sig_cols = sorted(sig_cols, key=lambda x: int(str(x)[6:]))

    core = ["Label", "ID", "Time"]
    for c in core:
        if c not in df.columns:
            df[c] = np.nan
    keep = core + sig_cols
    return df[keep]


def _open_first_csv_from_zip(path: Path):
    with path.open("rb") as f:
        with zipfile.ZipFile(io.BytesIO(f.read())) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV file found in ZIP")
            return io.TextIOWrapper(z.open(csv_names[0]), encoding="utf-8", errors="ignore")


def _load_signal(path: Path) -> pd.DataFrame:
    logger.info("Reading signal CSV: %s", path)
    if path.suffix.lower() == ".zip":
        fh = _open_first_csv_from_zip(path)
        try:
            return _read_signal_csv_flexible(fh)
        finally:
            try:
                fh.detach()
            except Exception:
                pass
    else:
        with path.open("rb") as f:
            return _read_signal_csv_flexible(io.TextIOWrapper(f, encoding="utf-8", errors="ignore"))


# ---------------- Metadata discovery & per-frame labeling for candump ----------------

def _find_dir_metadata_json(path: Path) -> Optional[Path]:
    """
    Metadata is aggregated at directory level:
      data/ambient/capture_metadata.json
      data/attacks/capture_metadata.json
      signal_extractions/ambient/metadata.json
      signal_extractions/attacks/metadata.json
    Return the nearest metadata json up the tree if present.
    """
    candidates = [
        path.parent / "capture_metadata.json",
        path.parent / "metadata.json",
        path.parent.parent / "capture_metadata.json",
        path.parent.parent / "metadata.json",
    ]
    for p in candidates:
        if p and p.exists():
            return p
    return None


def _load_dir_metadata(path: Path) -> Optional[Dict[str, dict]]:
    meta_path = _find_dir_metadata_json(path)
    if not meta_path:
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        logger.warning("Failed to parse metadata JSON %s: %s", meta_path, e)
        return None


def _lookup_capture_meta(dir_meta: Optional[Dict[str, dict]], path: Path) -> Optional[dict]:
    """
    Keys in metadata are capture_name. Try exact stem; if file is *_masquerade.log,
    try both with and without the suffix.
    """
    if not dir_meta:
        return None
    stem = path.stem
    if stem in dir_meta:
        return dir_meta[stem]
    # try removing masquerade suffix
    for suff in ["_masquerade", "-masquerade"]:
        if stem.endswith(suff) and stem[: -len(suff)] in dir_meta:
            return dir_meta[stem[: -len(suff)]]
    # last resort: case-insensitive match
    low = stem.lower()
    for k in dir_meta.keys():
        if k.lower() == low:
            return dir_meta[k]
    return None


def _mask_for_pattern(df: pd.DataFrame, pattern: Optional[str]) -> pd.Series:
    """
    Build a boolean mask for frames whose data bytes match a pattern string like:
      "XXXXXXXXFFXXXX00"
    where 'X' means "don't care" for that byte.

    README: 'injection_data' is such a string.
    """
    if not pattern or not isinstance(pattern, str):
        return pd.Series(True, index=df.index)

    p = pattern.strip()
    if len(p) != 16:
        p = p[:16].ljust(16, "X")

    masks: List[pd.Series] = []
    for byte_idx in range(8):
        byte_pat = p[2 * byte_idx: 2 * byte_idx + 2]
        if byte_pat.upper() == "XX":
            continue
        try:
            target_val = int(byte_pat, 16)
        except ValueError:
            continue
        masks.append(df[f"data{byte_idx}"] == target_val)

    if not masks:
        return pd.Series(True, index=df.index)

    m = masks[0]
    for mm in masks[1:]:
        m &= mm
    return m


def _label_frames_with_meta(out_frames: pd.DataFrame, attack_type: Optional[str], cap_meta: Optional[dict]) -> pd.Series:
    """
    Build per-frame labels from directory-level metadata.

    README (attacks):

      "{capture_name}" : {
          "elapsed_sec": length of capture in seconds,
          "on_dyno": bool,
          "message_confliction_present": bool,
          "injected_aid": injected AID in hex or Null,
          "injection_data": injected message hex string ('X' wildcard),
          "injection_interval":[ start, end ] in elapsed seconds
      }

    Notes:
      - injected_aid = Null:
          - fuzzing: all AIDs injected (during interval)
          - accelerator: injection happened before capture, no injected messages in log
      - We differentiate fuzzing vs accelerator using attack_type inferred
        from filename/dir.
    """
    label = pd.Series(np.zeros(len(out_frames), dtype=np.int8), index=out_frames.index)

    if not cap_meta:
        return label

    # ----- Injection interval -----
    interval = cap_meta.get("injection_interval")
    if not (isinstance(interval, (list, tuple)) and len(interval) == 2):
        # Ambient captures have no injection_interval: all benign
        return label

    start, end = float(interval[0]), float(interval[1])
    in_interval = (out_frames["timestamp"] >= start) & (out_frames["timestamp"] <= end)

    injected_aid = cap_meta.get("injected_aid")
    injection_data = cap_meta.get("injection_data")

    # Normalize "null-like" values
    def _is_null_like(v: object) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in {"null", "none", ""}:
            return True
        return False

    is_null_aid = _is_null_like(injected_aid)

    # ----- Accelerator: injected_aid = Null, but no injected messages in capture -----
    # README: "in the case of the accelerator attack, the injection happens before
    # the start of the capture ... captures have no injected messages."
    # Therefore we leave label = 0 for all frames.
    if attack_type == "Accelerator":
        return label  # all zero; compromised state but no injected frames in log

    # ----- Fuzzing: injected_aid = Null, but injection happens during capture -----
    if attack_type == "Fuzzing" and is_null_aid:
        # No single target AID, but we still have injection_data and interval.
        # Label frames in interval whose payload matches the pattern.
        m = in_interval & _mask_for_pattern(out_frames, injection_data)
        label.loc[m] = 1
        return label

    # ----- Targeted fabrication / masquerade -----
    if is_null_aid:
        # No specific AID; unknown case: conservatively mark all frames in interval as attacks
        label.loc[in_interval] = 1
        return label

    # Normal case: specific target AID + optional payload pattern
    try:
        target_id = parse_can_id(str(injected_aid))
        id_match = out_frames["can_id"] == np.int64(target_id)
    except Exception:
        # If ID can't be parsed, fall back to interval only
        label.loc[in_interval] = 1
        return label

    pattern_match = _mask_for_pattern(out_frames, injection_data)
    m = in_interval & id_match & pattern_match
    label.loc[m] = 1
    return label


# ---------------- Canonicalization ----------------

def to_canonical_candump(df_raw: pd.DataFrame, *, src_path: Optional[Path]) -> pd.DataFrame:
    """
    Use shared build_canonical for ROAD raw candump data; then label with directory metadata.

    ROAD metadata uses elapsed seconds from capture start, so we rebase timestamps to
    start-of-file before canonicalization.
    """
    # Rebase timestamps: absolute Unix -> elapsed seconds from start
    if not df_raw.empty:
        t0 = df_raw["timestamp"].iloc[0]
        df_raw = df_raw.copy()
        df_raw["timestamp"] = df_raw["timestamp"].astype(float) - float(t0)

    colmap = {
        "timestamp": "timestamp",
        "can_id": "can_id",
        "dlc": "dlc",
        "flag": None,
        "data": [f"data{i}" for i in range(8)],
    }

    attack_type = infer_attack_type(src_path) if src_path else None
    out = build_canonical(
        df_raw,
        colmap,
        dataset_name="ROAD",
        attack_type=attack_type,
        label_mapper=None,
    )

    # Label from directory-level metadata json
    dir_meta = _load_dir_metadata(src_path) if src_path else None
    cap_meta = _lookup_capture_meta(dir_meta, src_path) if dir_meta else None
    out["label"] = _label_frames_with_meta(out, attack_type, cap_meta).astype(np.int8)
    
    # CRITICAL FIX: Benign messages (label=0) should NOT have attack_type set.
    # attack_type should only be set for attack messages (label=1).
    # The inferred attack_type from filename applies only to the attack class.
    if attack_type is not None:
        out.loc[out["label"] == 0, "attack_type"] = None

    # Extra dataset-specific flags (optional)
    out["is_filler"] = (out["can_id"] == 0xFFF).astype(np.int8)
    out["obfuscated"] = True

    # Optionally propagate elapsed_sec / on_dyno if present
    if cap_meta:
        if "elapsed_sec" in cap_meta:
            out["elapsed_sec"] = float(cap_meta["elapsed_sec"])
        if "on_dyno" in cap_meta:
            out["on_dyno"] = bool(cap_meta["on_dyno"])

    return out


def to_canonical_signal(df_sig: pd.DataFrame, *, src_path: Optional[Path]) -> pd.DataFrame:
    """
    Use shared build_signals_canonical for ROAD signal-translated CSVs.
    ROAD readme:
      - ID is DECIMAL
      - Time is in SECONDS
      - Label column already encodes attack vs normal; accelerator attack CSVs
        have Label = 0 everywhere.
    """
    attack_type = infer_attack_type(src_path) if src_path else None

    # Identify all signal columns in stable order
    sig_cols = [c for c in df_sig.columns if str(c).startswith("Signal") and str(c)[6:].isdigit()]
    sig_cols = sorted(sig_cols, key=lambda x: int(str(x)[6:]))

    # Convert Time seconds -> milliseconds if your canonical uses ms
    df_local = df_sig.copy()
    df_local["Time_ms"] = pd.to_numeric(df_local["Time"], errors="coerce").astype(float) * 1000.0

    # ID is DECIMAL per ROAD
    def _id_to_int_dec(v: object) -> int:
        try:
            return int(str(v).strip())
        except Exception:
            # deterministic fallback
            return abs(hash(str(v))) & 0xFFFFFFFF

    df_local["ID_dec"] = df_sig["ID"].map(_id_to_int_dec).astype(np.int64)

    colmap = {
        "time_ms": "Time_ms",
        "id": "ID_dec",          # pass the decimal int id
        "label": "Label",
        "signals": sig_cols,     # arbitrary count per ID supported
    }

    return build_signals_canonical(
        df_local,
        colmap,
        dataset_name="ROAD-signal",
        split=None,
        attack_type=attack_type,
        expected_signal_counts=None,
    )


# ---------------- Public API ----------------

def load_raw(path: Path) -> pd.DataFrame:
    """
    Dispatch loader:
      - .csv / .zip  -> signal-translated CSV
      - .log / .txt  -> candump raw
      - otherwise: sniff short header for candump markers
    """
    suf = path.suffix.lower()
    if suf in {".csv", ".zip"}:
        return _load_signal(path)
    if suf in {".log", ".txt"}:
        return _load_candump(path)

    # Sniff
    with path.open("rb") as f:
        head = f.read(512)
    if b"#" in head and b")" in head:
        return _load_candump(path)
    return _load_signal(path)


def to_canonical(df: pd.DataFrame, *, src_path: Optional[Path]) -> pd.DataFrame:
    """
    Convert loaded ROAD data to canonical:
      - If it looks like frame-space (timestamp/can_id/dlc present) -> frame canonical
      - Else -> signal-space canonical
    """
    cols = set(df.columns)
    if {"timestamp", "can_id", "dlc"}.issubset(cols):
        return to_canonical_candump(df, src_path=src_path)
    return to_canonical_signal(df, src_path=src_path)


def convert_file(
    path_in: str | Path,
    path_out: str | Path | None = None,
    *,
    compression: str = "snappy",
    overwrite: bool = True,
) -> str:
    """
    Convert a ROAD file (candump or signal CSV/ZIP) to Parquet.
    - Candump -> frame-space canonical schema
    - Signal CSV -> signal-space canonical schema
    Suffix hint: *.parquet (frame), *.signal.parquet (signal).
    """
    p_in = Path(path_in)
    if not p_in.exists():
        raise FileNotFoundError(f"Input file not found: {p_in}")

    logger.info("Converting ROAD file: %s", p_in)
    df_raw = load_raw(p_in)
    canon = to_canonical(df_raw, src_path=p_in)

    # Decide suffix based on schema: signal-space has signal columns but not data0-data7
    suffix = ".parquet"
    if any(col.lower().startswith('signal') for col in canon.columns) and 'data0' not in canon.columns:
        suffix = ".signal.parquet"

    # Resolve output path
    if path_out is None:
        p_out = p_in.with_suffix(suffix)
    else:
        p_out = Path(path_out)
        if p_out.suffix == "":  # directory
            p_out = p_out / (p_in.stem + suffix)

    p_out.parent.mkdir(parents=True, exist_ok=True)
    if p_out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite=False: {p_out}")

    canon.to_parquet(p_out, index=False, compression=compression)
    logger.info("Wrote canonical parquet: %s", p_out)
    return str(p_out)

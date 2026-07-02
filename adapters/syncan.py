"""
SynCAN adapter that preserves decoded signal values as floats.

This adapter creates signal-space parquet files with Signal1-Signal4 columns
as float values, suitable for methods like CANet that need actual signal values
rather than quantized payload bytes.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ATTACK_MAP: dict[str, str] = {
    "plateau": "plateau",
    "continuous": "continuous",
    "playback": "playback",
    "suppress": "suppress",
    "flooding": "flooding",
}


def infer_attack_type(path: Path) -> Optional[str]:
    """Map filename substrings to attack types if needed."""
    name = path.stem.lower()
    for k, v in ATTACK_MAP.items():
        if k in name:
            return v
    return None


def _normalize_header_token(s: str) -> str:
    """Normalize raw header tokens."""
    s0 = str(s).strip()
    low = s0.lower().replace(" ", "")
    if low == "label":
        return "label"
    if low == "id":
        return "can_id"
    if low == "time":
        return "timestamp"
    for i in range(1, 5):
        if low in (f"signal{i}", f"signal{i}_of_id"):
            return f"signal{i}"
    return s0


def _read_csv_flexible(path) -> pd.DataFrame:
    """Read CSV with variable-length rows & optional header.
    
    Files with headers:
    - test_*.csv: Has "Label,Time,ID,Signal1_of_ID,..." header
    - train_1.csv: Has "Label,Time,ID,Signal1,..." header
    
    Files without headers (data only):
    - train_2.csv, train_3.csv, train_4.csv: Start directly with data rows
    """
    path_str = str(path).lower()
    
    # train_2, train_3, train_4 have no header
    if any(x in path_str for x in ["train_2", "train_3", "train_4"]):
        df = pd.read_csv(path, names=["label", "timestamp", "can_id", "signal1", "signal2", "signal3", "signal4"])
    else:
        # test_*.csv and train_1.csv have headers - read normally
        df = pd.read_csv(path)
    
    # Normalize all column names to lowercase canonical format
    df.columns = [_normalize_header_token(col) for col in df.columns]
    
    return df


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a SynCAN input file (.zip or .csv)."""
    suf = path.suffix.lower()
    logger.info("Loading SynCAN file: %s", path)
    if suf == ".csv":
        return _read_csv_flexible(path)
    else:
        raise ValueError(f"Unsupervised SynCAN file type: {path.suffix}")


def parse_syncan_id(val: object) -> int:
    """Convert SynCAN IDs like 'id5' to integer 5."""
    s = str(val).strip().lower()
    if s.startswith("id"):
        s = s[2:]
    if s.isdigit():
        return int(s)
    return abs(hash(val)) & 0xFFFFFFFF


def to_canonical_signals(
    df: pd.DataFrame,
    dataset_name: str = "SynCAN",
    attack_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert SynCAN raw DataFrame to signal-space canonical schema.
    
    Canonical columns produced:
      - timestamp (float)
      - can_id (int64)
      - label (int8)  # 0=normal, 1=attack
      - attack_type (str|None)
      - dataset (str)
      - idx_src (int64)
      - signal1, signal2, signal3, signal4 (float64) - PRESERVED as floats!
    
    This preserves the actual signal values for methods like CANet that need
    real-valued signals, not quantized bytes.
    """
    logger.info("Converting SynCAN DataFrame to signal-space canonical schema; shape=%s", df.shape)
    
    # Normalize column names
    df.columns = [_normalize_header_token(col) for col in df.columns]
    
    # Parse base fields (now using lowercase canonical names)
    ts = pd.to_numeric(df["timestamp"], errors="coerce").astype(float)
    can_ids = df["can_id"].apply(parse_syncan_id).astype(np.int64)
    label = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(np.int8)
    
    # Inter-arrival time
    inter = ts.diff().fillna(0.0).astype(float)
    
    # Set attack_type per-row based on label:
    # - Benign messages (label=0) get attack_type="benign"
    # - Attack messages (label=1) get the attack type inferred from filename (or "benign" for train files)
    # This is correct per SynCAN README: test files have both attack and benign messages
    attack_type_col = np.where(
        label == 0,
        "benign",
        attack_type if attack_type is not None else "benign"
    )
    
    # Build output dataframe
    out = pd.DataFrame({
        "timestamp": ts,
        "can_id": can_ids,
        "label": label,
        "attack_type": attack_type_col,
        "dataset": dataset_name,
        "idx_src": np.arange(len(df), dtype=np.int64),
        "inter_arrival": inter,
    })
    
    # Add signal columns - KEEP AS FLOAT, use lowercase canonical names
    signal_cols = ["signal1", "signal2", "signal3", "signal4"]
    for col in signal_cols:
        if col in df.columns:
            # Convert to numeric, keep as float
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
        else:
            out[col] = np.nan
    
    # Enforce canonical dtypes
    out = out.astype({
        "timestamp": float,
        "can_id": np.int64,
        "label": np.int8,
        "idx_src": np.int64,
        "inter_arrival": float,
    }, copy=False)
    
    return out


def load_raw(path: Path) -> pd.DataFrame:
    """Load raw SynCAN CSV file."""
    df = _load_csv(path)
    logger.info("Loaded raw SynCAN; shape=%s, columns=%s", df.shape, df.columns.tolist())
    return df


def convert_file(
    path_in: str | Path,
    path_out: str | Path | None = None,
    *,
    compression: str = "snappy",
    overwrite: bool = True,
) -> str:
    """
    Convert a SynCAN file to signal-space canonical Parquet.
    
    This creates parquet files with float-valued Signal columns suitable
    for CANet and other signal-based methods.
    """
    p_in = Path(path_in)
    if not p_in.exists():
        raise FileNotFoundError(f"Input file not found: {p_in}")
    
    logger.info("Converting SynCAN file (signal-space): %s...", p_in)
    raw = load_raw(p_in)
    atk = infer_attack_type(p_in)
    canon = to_canonical_signals(raw, attack_type=atk)
    
    # Resolve output path
    if path_out is None:
        p_out = p_in.with_name(p_in.stem + ".signals.parquet")
    else:
        p_out = Path(path_out)
        if p_out.suffix == "":  # treat as directory
            p_out = p_out / (p_in.stem + ".signals.parquet")
    p_out.parent.mkdir(parents=True, exist_ok=True)
    
    if p_out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite=False: {p_out}")
    
    canon.to_parquet(p_out, index=False, compression=compression)
    logger.info("Wrote signal-space parquet: %s", p_out)
    return str(p_out)


def convert_all(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "**/*.csv",
    compression: str = "snappy",
    overwrite: bool = False,
) -> list[str]:
    """Convert all SynCAN CSV files in a directory tree to signal-space parquet."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = sorted(input_path.glob(pattern))
    logger.info(f"Found {len(files)} SynCAN files to convert")
    
    converted = []
    for f in files:
        try:
            # Preserve directory structure relative to input_dir
            rel_path = f.relative_to(input_path)
            out_dir = output_path / rel_path.parent
            out_path = out_dir / (f.stem + ".signals.parquet")
            
            if out_path.exists() and not overwrite:
                logger.info(f"Skipping existing: {out_path}")
                continue
            
            result = convert_file(f, out_path, compression=compression, overwrite=overwrite)
            converted.append(result)
        except Exception as e:
            logger.error(f"Failed to convert {f}: {e}")
    
    logger.info(f"Converted {len(converted)}/{len(files)} files")
    return converted


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python syncan_signals.py <input_csv_or_dir> [output_dir]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if input_path.is_dir():
        output_path = output_path or input_path.parent / "syncan_signals_parquet"
        convert_all(input_path, output_path, overwrite=False)
    else:
        result = convert_file(input_path, output_path, overwrite=True)
        print(f"Converted: {result}")

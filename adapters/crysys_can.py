# adapters/crysys_can.py
"""
Adapter for the CrySyS CAN Intrusion Detection Dataset.

Dataset structure:
- Organized in scenario folders (S-1-1, S-1-2, etc. and T-1-1, T-2-1, etc.)
- Each scenario contains:
  - 1 benign trace: <scenario>-benign.log + .json
  - Multiple attack traces: <scenario>-malicious-<attack>-<params>.log + .json
  
File formats:
- .log: SocketCAN format: (timestamp) interface can_id#payload_hex
- .json: Metadata with attack markers (start/end times)

Attack types from filenames:
- msg-inj = Message Injection
- msg-mod = Message Modification
- Attack patterns: ADD-DECR, ADD-INCR, CONST, NEG-OFFSET, POS-OFFSET, REPLAY, DOUBLE
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import re
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .utils.can_parse import parse_can_id
from .utils.can_canonical import build_canonical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- Attack type mapping ----------------

ATTACK_PATTERN_MAP = {
    "msg-inj": "MessageInjection",
    "msg-mod": "MessageModification",
    "add-decr": "AddDecrement",
    "add-incr": "AddIncrement",
    "const": "Constant",
    "neg-offset": "NegativeOffset",
    "neg_offset": "NegativeOffset",
    "pos-offset": "PositiveOffset",
    "pos_offset": "PositiveOffset",
    "replay": "Replay",
    "double": "DoubleAttack",
}


def infer_attack_type(path: Path) -> Optional[str]:
    """
    Extract attack type from CrySyS filename.
    
    Example: S-1-1-malicious-ADD-DECR-msg-inj-0x410-0.4-0.6.log
    Returns: "MessageInjection-AddDecrement"
    """
    if "benign" in path.stem.lower():
        return None
    
    name = path.stem.lower()
    attack_parts = []
    
    for pattern, label in ATTACK_PATTERN_MAP.items():
        if pattern in name:
            attack_parts.append(label)
    
    if attack_parts:
        # Remove duplicates while preserving order
        unique_parts = []
        for part in attack_parts:
            if part not in unique_parts:
                unique_parts.append(part)
        return "-".join(unique_parts)
    
    if "malicious" in name:
        return "Unknown"
    
    return None


# ---------------- SocketCAN log parsing ----------------

def parse_socketcan_line(line: str) -> Optional[dict]:
    """
    Parse a single SocketCAN log line.
    
    Format: (timestamp) interface can_id#payload_hex
    Example: (0.000000) can0 110#02202e1300181300
    
    Returns dict with: timestamp, can_id, payload_bytes
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    
    # Pattern: (timestamp) interface can_id#payload
    match = re.match(r'\(([0-9.]+)\)\s+(\w+)\s+([0-9a-fA-F]+)#([0-9a-fA-F]*)', line)
    if not match:
        return None
    
    timestamp_str, interface, can_id_hex, payload_hex = match.groups()
    
    try:
        timestamp = float(timestamp_str)
        can_id = int(can_id_hex, 16)
        
        # Parse payload bytes
        payload_hex = payload_hex.strip()
        if payload_hex:
            # Split into pairs of hex digits
            payload_bytes = [int(payload_hex[i:i+2], 16) for i in range(0, len(payload_hex), 2)]
        else:
            payload_bytes = []
        
        # DLC = actual number of bytes in the original payload
        dlc = len(payload_bytes)
        
        # Pad to 8 bytes for fixed schema (zeros for missing bytes)
        while len(payload_bytes) < 8:
            payload_bytes.append(0)
        
        return {
            "timestamp": timestamp,
            "can_id": can_id,
            "dlc": dlc,
            "data0": payload_bytes[0],
            "data1": payload_bytes[1],
            "data2": payload_bytes[2],
            "data3": payload_bytes[3],
            "data4": payload_bytes[4],
            "data5": payload_bytes[5],
            "data6": payload_bytes[6],
            "data7": payload_bytes[7],
        }
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse line: {line} - {e}")
        return None


def load_socketcan_log(path: Path) -> pd.DataFrame:
    """Load a SocketCAN .log file into a DataFrame."""
    records = []
    
    with open(path, "r") as f:
        for line in f:
            parsed = parse_socketcan_line(line)
            if parsed:
                records.append(parsed)
    
    if not records:
        logger.warning(f"No valid CAN messages found in {path}")
        return pd.DataFrame()
    
    return pd.DataFrame(records)


def load_attack_markers(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load attack time windows from JSON metadata.
    
        Returns list of window dicts:
            [{"start": float, "end": float, "packet_id": Optional[int]}]
    """
    if not json_path.exists():
        return []
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        markers = data.get("markers", [])
        if not markers or len(markers) < 2:
            return []
        
        # Extract start and end times
        # Markers should have "Start of the attack" and "End of the attack"
        start_time = None
        end_time = None
        packet_id = None
        
        for marker in markers:
            desc = marker.get("description", "").lower()
            time = marker.get("time")
            pid = marker.get("packet_ID")

            if pid is not None and packet_id is None:
                try:
                    packet_id = int(parse_can_id(str(pid)))
                except Exception:
                    logger.warning(f"Failed to parse packet_ID '{pid}' in {json_path}")
            
            if time is not None:
                if "start" in desc:
                    start_time = float(time)
                elif "end" in desc:
                    end_time = float(time)
        
        if start_time is not None and end_time is not None:
            return [{"start": start_time, "end": end_time, "packet_id": packet_id}]
        
        return []
    
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse attack markers from {json_path}: {e}")
        return []


def assign_labels(df: pd.DataFrame, attack_windows: List[Dict[str, Any]]) -> pd.Series:
    """
    Assign labels based on attack windows (and packet_ID when available).
    
    Returns int8 Series: 0 for benign, 1 for attack.
    """
    if not attack_windows:
        return pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
    
    labels = np.zeros(len(df), dtype=np.int8)
    timestamps = df["timestamp"].values
    
    for window in attack_windows:
        start_time = float(window["start"])
        end_time = float(window["end"])
        packet_id = window.get("packet_id")

        mask = (timestamps >= start_time) & (timestamps <= end_time)
        if packet_id is not None and "can_id" in df.columns:
            mask &= (df["can_id"].values == np.int64(packet_id))
        labels[mask] = 1
    
    return pd.Series(labels, index=df.index)


# ---------------- Main conversion functions ----------------

def to_canonical(
    df: pd.DataFrame,
    dataset_name: str = "CrySyS",
    attack_type: Optional[str] = None,
    attack_windows: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Convert CrySyS DataFrame to canonical schema.
    """
    if df.empty:
        logger.warning("Empty DataFrame, returning empty canonical DataFrame")
        return pd.DataFrame()
    
    # Assign labels based on attack windows
    if attack_windows:
        labels = assign_labels(df, attack_windows)
    else:
        labels = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
    
    # Build canonical DataFrame
    n = len(df)
    inter_arrival = df["timestamp"].diff().fillna(0.0).astype(float)
    
    canonical = pd.DataFrame({
        "timestamp": df["timestamp"].astype(float),
        "can_id": df["can_id"].astype(np.int64),
        "dlc": df["dlc"].astype(np.int8),
        "label": labels,
        "attack_type": attack_type,
        "dataset": dataset_name,
        "frame_type": None,
        "vehicle": None,
        "split": None,
        "idx_src": np.arange(n, dtype=np.int64),
        "inter_arrival": inter_arrival,
        "data0": df["data0"].astype(np.uint8),
        "data1": df["data1"].astype(np.uint8),
        "data2": df["data2"].astype(np.uint8),
        "data3": df["data3"].astype(np.uint8),
        "data4": df["data4"].astype(np.uint8),
        "data5": df["data5"].astype(np.uint8),
        "data6": df["data6"].astype(np.uint8),
        "data7": df["data7"].astype(np.uint8),
    })
    
    # CRITICAL FIX: Benign messages (label=0) should NOT have attack_type set.
    # attack_type should only be set for attack messages (label=1).
    # The inferred attack_type from filename applies only to the attack class.
    if attack_type is not None:
        canonical.loc[canonical["label"] == 0, "attack_type"] = None
    
    return canonical


def convert_file(
    path_in: str | Path,
    path_out: str | Path | None = None,
    *,
    compression: str = "snappy",
    overwrite: bool = True,
) -> str:
    """
    Convert a single CrySyS .log file to canonical Parquet.
    
    - Reads .log file (SocketCAN format)
    - Reads corresponding .json file for attack markers
    - Assigns labels based on attack time windows
    - Writes canonical Parquet
    
    Args:
        path_in: Path to .log file
        path_out: Output path (default: same dir with .parquet suffix)
        compression: Parquet compression (default: "snappy")
        overwrite: Whether to overwrite existing output
    
    Returns:
        Output path as string
    """
    p_in = Path(path_in)
    if not p_in.exists():
        raise FileNotFoundError(f"Input file not found: {p_in}")
    
    # Load log file
    logger.info(f"Loading CrySyS log: {p_in}")
    df_raw = load_socketcan_log(p_in)
    
    if df_raw.empty:
        logger.warning(f"No data in {p_in}, skipping")
        return ""
    
    # Infer attack type from filename first (needed for metadata fallback policy)
    attack_type = infer_attack_type(p_in)

    # Load attack markers from JSON
    json_path = p_in.with_suffix(".json")
    attack_windows = load_attack_markers(json_path)

    # CrySyS contains paired traces where injected-message extracts are stored as
    # "*-inj-messages.log" without a dedicated JSON file. Reuse the paired full
    # trace JSON markers when available.
    if not attack_windows and attack_type is not None and p_in.stem.endswith("-inj-messages"):
        paired_json = p_in.with_name(p_in.stem.replace("-inj-messages", "") + ".json")
        paired_windows = load_attack_markers(paired_json)
        if paired_windows:
            attack_windows = paired_windows
            logger.warning(
                "No dedicated marker JSON for %s; using paired markers from %s",
                p_in.name,
                paired_json.name,
            )

    # Safety policy: never silently mark malicious traces fully benign when
    # metadata is missing or malformed.
    if not attack_windows and attack_type is not None:
        logger.warning(
            "Missing attack markers for malicious trace %s; skipping conversion for this trace",
            p_in.name,
        )
        return ""
    
    # Convert to canonical
    canonical = to_canonical(
        df_raw,
        dataset_name="CrySyS",
        attack_type=attack_type,
        attack_windows=attack_windows,
    )
    
    # Resolve output path
    if path_out is None:
        p_out = p_in.with_suffix(".parquet")
    else:
        p_out = Path(path_out)
        if p_out.suffix == "":  # treat as directory
            p_out = p_out / (p_in.stem + ".parquet")
    
    p_out.parent.mkdir(parents=True, exist_ok=True)
    
    if p_out.exists() and not overwrite:
        raise FileExistsError(f"Output exists and overwrite=False: {p_out}")
    
    # Write Parquet
    canonical.to_parquet(p_out, index=False, compression=compression)
    logger.info(f"Wrote {len(canonical)} frames to {p_out}")
    
    # Log attack info
    if attack_windows:
        attack_count = canonical["label"].sum()
        logger.info(f"  Attack windows: {attack_windows}")
        logger.info(f"  Attack frames: {attack_count}/{len(canonical)} ({100*attack_count/len(canonical):.1f}%)")
    
    return str(p_out)


def convert_folder(
    root_in: str | Path,
    out_dir: str | Path | None = None,
    *,
    compression: str = "snappy",
    overwrite: bool = True,
) -> List[str]:
    """
    Convert all .log files in a CrySyS dataset folder tree.
    
    Args:
        root_in: Root directory containing scenario folders
        out_dir: Output directory (default: data_parquet/06_CrySyS)
        compression: Parquet compression
        overwrite: Whether to overwrite existing files
    
    Returns:
        List of output paths
    """
    root = Path(root_in)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    
    if out_dir is None:
        out_dir = Path("data_parquet") / "06_CrySyS"
    else:
        out_dir = Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .log files
    log_files = sorted(root.rglob("*.log"))
    logger.info(f"Found {len(log_files)} .log files in {root}")
    
    output_paths = []
    
    for log_file in log_files:
        try:
            # Preserve directory structure
            rel_path = log_file.relative_to(root)
            out_path = out_dir / rel_path.with_suffix(".parquet")
            
            result = convert_file(
                log_file,
                out_path,
                compression=compression,
                overwrite=overwrite,
            )
            
            if result:
                output_paths.append(result)
        
        except Exception as e:
            logger.error(f"Failed to convert {log_file}: {e}")
            continue
    
    logger.info(f"Converted {len(output_paths)}/{len(log_files)} files")
    return output_paths


# ---------------- CLI for standalone testing ----------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CrySyS dataset to canonical Parquet")
    parser.add_argument("input", help="Input .log file or directory")
    parser.add_argument("--output", "-o", help="Output path or directory")
    parser.add_argument("--compression", default="snappy", help="Parquet compression")
    parser.add_argument("--no-overwrite", action="store_true", help="Don't overwrite existing files")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        convert_file(
            input_path,
            args.output,
            compression=args.compression,
            overwrite=not args.no_overwrite,
        )
    elif input_path.is_dir():
        convert_folder(
            input_path,
            args.output,
            compression=args.compression,
            overwrite=not args.no_overwrite,
        )
    else:
        print(f"Error: {input_path} is not a file or directory")
        exit(1)

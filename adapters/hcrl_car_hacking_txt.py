"""
Adapter for HCRL Car-Hacking Dataset (txt format - normal_run_data.txt)

Format:
Timestamp: <float>        ID: <hex>    000    DLC: <int>    <hex> <hex> <hex> ...
"""

from pathlib import Path
import pandas as pd
import re
from typing import Optional
from .utils.can_canonical import build_canonical


def parse_txt_line(line: str) -> Optional[dict]:
    """Parse a single line from the txt format."""
    # Pattern: Timestamp: <float>        ID: <hex>    000    DLC: <int>    <hex> <hex> ...
    pattern = r"Timestamp:\s+([\d.]+)\s+ID:\s+([0-9a-fA-F]+)\s+000\s+DLC:\s+(\d+)\s+(.*)"
    match = re.match(pattern, line.strip())
    
    if not match:
        return None
    
    timestamp = float(match.group(1))
    can_id = match.group(2)
    dlc = int(match.group(3))
    data_str = match.group(4).strip()
    
    # Parse data bytes (space-separated hex)
    data_bytes = data_str.split()
    
    # Ensure we have at most 8 data bytes
    if len(data_bytes) > 8:
        data_bytes = data_bytes[:8]
    
    # Pad with zeros if needed
    while len(data_bytes) < 8:
        data_bytes.append("00")
    
    return {
        "Timestamp": timestamp,
        "CAN ID": can_id,
        "DLC": dlc,
        **{f"DATA[{i}]": data_bytes[i] for i in range(8)},
        "Flag": "R"  # Normal data is received (not transmitted)
    }


def load_raw(path: Path) -> pd.DataFrame:
    """Load the txt file and parse it into a DataFrame."""
    rows = []
    
    with open(path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parsed = parse_txt_line(line)
                if parsed:
                    rows.append(parsed)
    
    if not rows:
        raise ValueError(f"No valid lines parsed from {path}")
    
    return pd.DataFrame(rows)


def to_canonical(df: pd.DataFrame, dataset_name="HCRL-CarHacking", attack_type=None) -> pd.DataFrame:
    """Convert to canonical format."""
    colmap = {
        "timestamp": "Timestamp",
        "can_id": "CAN ID",
        "dlc": "DLC",
        "flag": "Flag",
        "data": [f"DATA[{i}]" for i in range(8)],
    }
    label_mapper = {"T": 1, "R": 0}
    return build_canonical(df, colmap, dataset_name=dataset_name, attack_type=attack_type, label_mapper=label_mapper)


def convert_file(
    path_in: str | Path,
    path_out: str | Path | None = None,
    *,
    compression: str = "snappy",
    overwrite: bool = True,
) -> str:
    """
    Convert a single raw txt file to canonical parquet.

    - If `path_out` is None: writes alongside the input with `.parquet` suffix.
    - If `path_out` is a directory: writes as <input_stem>.parquet inside it.
    - If `overwrite` is False and the output exists: raises FileExistsError.
    - Returns the output path as a string.
    """
    p_in = Path(path_in)
    if not p_in.exists():
        raise FileNotFoundError(f"Input file not found: {p_in}")

    # Load + convert
    df_raw = load_raw(p_in)
    canon = to_canonical(df_raw, attack_type=None)

    # Resolve output path
    if path_out is None:
        p_out = p_in.with_suffix(".parquet")
    else:
        p_out = Path(path_out)
        if p_out.suffix == "":  # treat as directory if no suffix
            p_out = p_out / (p_in.stem + ".parquet")
    p_out.parent.mkdir(parents=True, exist_ok=True)

    if p_out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite=False: {p_out}")

    # Write parquet
    canon.to_parquet(p_out, index=False, compression=compression)

    return str(p_out)


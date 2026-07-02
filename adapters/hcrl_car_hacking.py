
from pathlib import Path
import pandas as pd
from typing import Optional
from .utils.can_canonical import build_canonical

ATTACK_MAP = {"dos":"DoS","fuzzy":"Fuzzy","gear":"GearSpoof","rpm":"RPMSpoof"}

def infer_attack_type(path: Path) -> Optional[str]:
    name = path.stem.lower()
    for k, v in ATTACK_MAP.items():
        if k in name:
            return v
    return None

def load_raw(path: Path) -> pd.DataFrame:
    # no header; fixed order columns
    names = ["Timestamp","CAN ID","DLC","DATA[0]","DATA[1]","DATA[2]","DATA[3]","DATA[4]","DATA[5]","DATA[6]","DATA[7]","Flag"]
    return pd.read_csv(path, header=None, names=names, sep=None, engine="python", skip_blank_lines=True)

def to_canonical(df: pd.DataFrame, dataset_name="HCRL-CarHacking", attack_type=None) -> pd.DataFrame:
    colmap = {
        "timestamp": "Timestamp",
        "can_id": "CAN ID",
        "dlc": "DLC",
        "flag": "Flag",
        "data": [f"DATA[{i}]" for i in range(8)],
    }
    label_mapper = {"T": 1, "R": 0}
    result = build_canonical(df, colmap, dataset_name=dataset_name, attack_type=attack_type, label_mapper=label_mapper)
    
    # CRITICAL FIX: Benign messages (label=0) should NOT have attack_type set.
    # attack_type should only be set for attack messages (label=1).
    # The inferred attack_type from filename applies only to the attack class.
    if attack_type is not None:
        result.loc[result["label"] == 0, "attack_type"] = None
    
    return result


from pathlib import Path

def convert_file(
    path_in: str | Path,
    path_out: str | Path | None = None,
    *,
    compression: str = "snappy",
    overwrite: bool = True,
) -> str:
    """
    Convert a single raw file to canonical parquet.

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
    atk = infer_attack_type(p_in)
    canon = to_canonical(df_raw, attack_type=atk)

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

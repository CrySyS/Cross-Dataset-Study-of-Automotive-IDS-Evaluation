
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


REQUIRED_FRAME_COLS = [
    "timestamp","can_id","dlc","label","attack_type","dataset",
    "frame_type","vehicle","split","idx_src","inter_arrival",
    "data0","data1","data2","data3","data4","data5","data6","data7"
]

def _enforce_canonical_dtypes(df: pd.DataFrame, *, source: Path) -> pd.DataFrame:
    """
    Strict canonicalization:
      - Coerce numeric columns with pd.to_numeric(errors="coerce")
      - Drop bad can_id rows (as before)
      - FAIL if any critical numeric columns contain NaN after coercion,
        but now prints actionable diagnostics (row samples + raw offending values).

    This does NOT add fallbacks for timestamp/inter_arrival — it remains strict.
    """
    # Replace ±inf -> NaN up front to avoid casting surprises
    df = df.replace([np.inf, -np.inf], np.nan)

    # --- timestamp & inter_arrival (float) ---
    for col in ("timestamp", "inter_arrival"):
        if col in df.columns:
            raw = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Print examples of non-null raw values that became NaN due to coercion
            bad_coerce = df[col].isna() & raw.notna()
            if bad_coerce.any():
                ex = pd.DataFrame(
                    {
                        f"{col}_raw": raw.loc[bad_coerce].head(10).astype("string"),
                    }
                )
                context_cols = [c for c in ["can_id", "idx_src", "attack_type", "dataset"] if c in df.columns]
                if context_cols:
                    ex = pd.concat([df.loc[bad_coerce, context_cols].head(10).reset_index(drop=True),
                                    ex.reset_index(drop=True)], axis=1)
                print(f"[{source}] Column '{col}': {int(bad_coerce.sum())} non-numeric values coerced to NaN. "
                      f"Examples:\n{ex}")

    # --- can_id (int64, but coerce first; drop bad rows with logging) ---
    if "can_id" in df.columns:
        before = len(df)
        raw_can = df["can_id"].copy()
        df["can_id"] = pd.to_numeric(df["can_id"], errors="coerce")
        bad_can = int(df["can_id"].isna().sum())
        if bad_can:
            '''cols = [c for c in ["timestamp", "can_id", "dlc", "attack_type", "dataset", "idx_src"] if c in df.columns]
            examples = df.loc[df["can_id"].isna(), cols].head(5)
            raw_examples = raw_can.loc[df["can_id"].isna()].head(5).astype("string")
            examples = examples.assign(can_id_raw=raw_examples.values)'''
            raise ValueError(f"[{source}] Found {bad_can} invalid can_id values; dropping these rows. Examples:\n{examples}")
        df = df[df["can_id"].notna()].copy()
        df["can_id"] = df["can_id"].astype("int64")
        if len(df) < before:
            print(f"[{source}] Dropped {before - len(df)} rows due to invalid can_id.")

    # --- dlc (0..8, int8) ---
    if "dlc" in df.columns:
        df["dlc"] = (
            pd.to_numeric(df["dlc"], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=8)
            .astype("int8")
        )

    # --- label ({0,1}, int8) ---
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0)
        non01 = int((~df["label"].isin([0, 1])).sum())
        if non01:
            print(f"[{source}] Warning: {non01} labels not in {{0,1}} — coerced to 0.")
            df.loc[~df["label"].isin([0, 1]), "label"] = 0
        df["label"] = df["label"].astype("int8")

    # --- payload bytes (uint8) ---
    for b in ["data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"]:
        if b in df.columns:
            df[b] = (
                pd.to_numeric(df[b], errors="coerce")
                .fillna(0)
                .clip(lower=0, upper=255)
                .astype("uint8")
            )

    # --- idx_src (int64; fill missing with -1) ---
    if "idx_src" in df.columns:
        df["idx_src"] = (
            pd.to_numeric(df["idx_src"], errors="coerce")
            .fillna(-1)
            .astype("int64")
        )

    # --- categorical-ish ---
    for c in ["attack_type", "dataset", "frame_type", "vehicle", "split"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # --- final sanity on critical numerics (if present) ---
    must_be_finite = [c for c in ["timestamp", "inter_arrival", "can_id", "dlc", "label"] if c in df.columns]
    leftover = {c: int(df[c].isna().sum()) for c in must_be_finite if df[c].isna().any()}

    if leftover:
        # Print examples of offending rows for each column
        for c, n in leftover.items():
            mask = df[c].isna()
            cols = [x for x in ["timestamp", "inter_arrival", "can_id", "dlc", "label", "idx_src", "attack_type", "dataset"] if x in df.columns]
            ex = df.loc[mask, cols].head(10)
            print(f"[{source}] NaN detected in critical column '{c}' (count={n}). Examples:\n{ex}\n")

        raise ValueError(f"[{source}] Unexpected NaNs after coercion in: {leftover}")

    # Stable ordering if timestamp exists
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def read_parquet_glob(
    glob_path: Union[str, Path, Iterable[Union[str, Path]]],
    columns: Optional[List[str]] = None,
    with_sources: bool = False,
    adjust_timestamps: bool = True,
) -> Union[pd.DataFrame, List[Tuple[pd.DataFrame, Path]], Tuple[pd.DataFrame, List[Tuple[int, int, str]]]]:
    """
    Load one or more parquet files matching one or multiple globs.
    - If with_sources=False (default): returns a single concatenated DataFrame.
    - If with_sources=True: returns a list of (DataFrame, Path) tuples.
    - If adjust_timestamps=True (default): adjusts timestamps so files don't overlap
      (important for ROAD and other multi-file datasets where each file starts at t=0)
    """

    if isinstance(glob_path, (str, Path)):
        patterns: List[Union[str, Path]] = [glob_path]
    else:
        patterns = list(glob_path)

    paths_set = set()
    for pattern in patterns:
        for p in Path().glob(str(pattern)):
            paths_set.add(p)

    paths = sorted(paths_set)
    if not paths:
        raise FileNotFoundError(f"No parquet files match: {patterns}")

    results: List[Tuple[pd.DataFrame, Path]] = []
    timestamp_offset = 0.0
    
    for p in paths:
        df = pd.read_parquet(p, columns=columns)
        # Early diagnostic: how many can_id will be problematic (before cleaning)
        if "can_id" in df.columns:
            pre_bad = int(pd.to_numeric(df["can_id"], errors="coerce").isna().sum())
            if pre_bad:
                print(f"[{p}] NOTE: {pre_bad} invalid can_id values detected pre-clean.")

        # Enforce canonical dtypes robustly
        df = _enforce_canonical_dtypes(df, source=p)
        
        # Track source file for boundary detection in windowing
        df["_source_file"] = p.stem
        
        # Adjust timestamps to avoid overlap when concatenating multiple files
        if adjust_timestamps and "timestamp" in df.columns and len(df) > 0:
            if timestamp_offset > 0:
                df["timestamp"] = df["timestamp"] + timestamp_offset
            # Update offset for next file (add 1 second gap to avoid boundary issues)
            timestamp_offset = df["timestamp"].max() + 1.0
        
        results.append((df, p))

    if with_sources:
        return results

    # concatenate if not needed separately
    dfs = [d for d, _ in results]
    df_all = pd.concat(dfs, ignore_index=True)
    
    # NOTE: Do NOT re-sort by timestamp after adjustment!
    # When adjust_timestamps=True, files are already in the correct order (timestamp-contiguous).
    # Re-sorting would break this ordering and can mix data from different files/attack-types
    # (e.g., if one file has pre-existing large timestamps, re-sort scrambles at boundaries).
    # For datasets without timestamp adjustments, they should already be in the desired order.
    
    
    return df_all
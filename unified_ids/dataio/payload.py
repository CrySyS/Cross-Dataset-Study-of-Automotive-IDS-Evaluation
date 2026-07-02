# unified_ids/dataio/payload.py
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


def pad_payload(
    df: pd.DataFrame,
    *,
    n_bytes: int = 8,
    cols_prefix: str = "data",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Ensure payload columns data0..data{n_bytes-1} exist and contain ints in [0,255].

    - Missing columns are created and filled with zeros.
    - NaNs are filled with zeros.
    - Values are cast to int and clipped to [0,255].

    If inplace=False (default), returns a modified COPY of df.
    If inplace=True, modifies df in-place and returns the same object.
    """
    if not inplace:
        df = df.copy()

    for i in range(n_bytes):
        col = f"{cols_prefix}{i}"
        if col not in df.columns:
            df[col] = 0
        df[col] = (
            df[col]
            .fillna(0)
            .astype(int)
            .clip(0, 255)
        )

    return df


def payload_to_binary_matrix(
    df: pd.DataFrame,
    *,
    n_bytes: int = 8,
    cols_prefix: str = "data",
    msb_first: bool = True,
    auto_pad: bool = True,
    pad_inplace: bool = False,
) -> np.ndarray:
    """
    Convert payload bytes to a {0,1} bit matrix of shape (N, 8 * n_bytes).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain payload columns data0..data{n_bytes-1} (or be pad-able).
    n_bytes : int
        Number of data* columns / bytes per payload (default 8).
    cols_prefix : str
        Prefix of payload columns, e.g. 'data' for data0..data7.
    msb_first : bool
        If True, bits per byte are ordered [bit7 .. bit0].
        If False, bits per byte are ordered [bit0 .. bit7].
    auto_pad : bool
        If True, will call pad_payload() before converting.
    pad_inplace : bool
        Passed through to pad_payload(inplace=...).
        Use True in tight loops on per-ID slices to avoid unnecessary copies.

    Returns
    -------
    X : np.ndarray
        Shape (N, 8 * n_bytes), dtype float32 with values 0 or 1.
    """
    if auto_pad:
        df = pad_payload(
            df,
            n_bytes=n_bytes,
            cols_prefix=cols_prefix,
            inplace=pad_inplace,
        )

    cols = [f"{cols_prefix}{i}" for i in range(n_bytes)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing payload columns: {missing}")

    N = len(df)
    bits_per_byte = 8
    total_bits = bits_per_byte * n_bytes
    X = np.zeros((N, total_bits), dtype=np.float32)

    for row_idx, row in enumerate(df[cols].itertuples(index=False, name=None)):
        bit_list: List[int] = []
        for b in row:
            b_int = int(b)
            if msb_first:
                # bit 7 .. bit 0
                bit_list.extend([(b_int >> k) & 1 for k in range(7, -1, -1)])
            else:
                # bit 0 .. bit 7
                bit_list.extend([(b_int >> k) & 1 for k in range(8)])
        X[row_idx, :] = bit_list

    return X

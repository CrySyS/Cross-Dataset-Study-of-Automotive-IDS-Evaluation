# unified_ids/dataio/windowing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import uuid

import pandas as pd


@dataclass
class Window:
    window_id: str
    idx_start: int          # inclusive index in df
    idx_end: int            # inclusive index in df
    t_start: float          # window start time (seconds)
    t_end: float            # window end time (seconds)
    label_window: int       # 0 benign, 1 attack (any attack in window)
    frac_attack: float      # fraction of attack messages in window


def get_uniform_window_label(df_window: pd.DataFrame) -> tuple[int, float]:
    """
    Compute uniform window label following the "any attack" rule.
    
    This is the CANONICAL implementation of window labeling for fair
    comparison across all IDS methods.
    
    Rule: A window is labeled as "attack" (1) if ANY message within the
          window has label==1. Otherwise, the window is labeled "benign" (0).
    
    Args:
        df_window: DataFrame subset representing a single window
        
    Returns:
        (label_window, frac_attack) where:
            - label_window: 0 (benign) or 1 (attack)
            - frac_attack: fraction of messages labeled as attack [0.0, 1.0]
    
    Example:
        >>> window_df = df.loc[100:200]  # some window
        >>> label, frac = get_uniform_window_label(window_df)
        >>> # Use this label for yield (window_id, score, label)
    """
    if "label" not in df_window.columns:
        raise ValueError("DataFrame must contain 'label' column for window labeling")
    
    labels = df_window["label"]
    label_window = int(labels.any())  # 1 if any label==1, else 0
    frac_attack = float(labels.mean())  # fraction of attack messages
    
    return label_window, frac_attack


def windows_fixed_msgs(
    df: pd.DataFrame,
    size: int = 900,
    stride: int = 450,
) -> Iterator[Window]:
    """
    Fixed-message windows: slices df by row count.

    Each window is a block of 'size' rows, sliding by 'stride' rows.
    Indices are inclusive [idx_start, idx_end].
    
    Automatically skips windows that span multiple source files (if _source_file column exists).
    This ensures windows don't mix different datasets/contexts/vehicles.
    """
    N = len(df)
    i = 0
    has_source_col = "_source_file" in df.columns
    
    while i < N:
        j = min(i + size, N)
        sub = df.iloc[i:j]
        if len(sub) == 0:
            break

        # Skip windows that span multiple source files
        if has_source_col:
            if sub["_source_file"].nunique() > 1:
                i += stride
                continue

        wid = f"{int(sub.index[0])}:{int(sub.index[-1])}"
        t_start = float(sub["timestamp"].iloc[0])
        t_end = float(sub["timestamp"].iloc[-1])

        # Use canonical labeling function
        label, frac = get_uniform_window_label(sub)

        yield Window(
            window_id=wid,
            t_start=t_start,
            t_end=t_end,
            idx_start=int(sub.index[0]),
            idx_end=int(sub.index[-1]),
            label_window=label,
            frac_attack=frac,
        )

        if j == N:
            break
        i += stride


def windows_fixed_time(
    df: pd.DataFrame,
    span_seconds: float,
    stride_seconds: float,
) -> Iterator[Window]:
    """
    Slide a fixed-duration window over df based on the 'timestamp' column (seconds).
    Assumes df is sorted by timestamp. Inclusive start, exclusive end by time.

    Indices are inclusive [idx_start, idx_end].
    
    **Window Labeling:**
    Uses `get_uniform_window_label` which implements the "any attack" rule:
    - Window labeled as attack (1) if ANY message within has label==1
    - Window labeled as benign (0) if ALL messages have label==0
    
    Args:
        df: DataFrame with 'timestamp' and 'label' columns
        span_seconds: Window duration in seconds
        stride_seconds: Window stride in seconds
        
    Yields:
        Window objects with start/end indices, timestamps, and uniform labels
    """
    if df.empty:
        return

    has_source_col = "_source_file" in df.columns

    ts = df["timestamp"].astype(float).values
    idx = df.index.values

    t_min = float(ts[0])
    t_max = float(ts[-1])
    t_start = t_min

    # two-pointer sweep to avoid O(N^2)
    i = 0
    j = 0
    N = len(df)

    while t_start <= t_max:
        t_end = t_start + span_seconds

        # advance i to first index with ts[i] >= t_start
        while i < N and ts[i] < t_start:
            i += 1
        # advance j to first index with ts[j] >= t_end
        while j < N and ts[j] < t_end:
            j += 1

        if i >= N:
            break

        if j > i:
            sub = df.iloc[i:j]
            wid = f"{int(sub.index[0])}:{int(sub.index[-1])}"

            # Skip windows that span multiple source files (if available)
            if has_source_col and sub["_source_file"].nunique() > 1:
                # do not yield this window; move on to next time step
                pass
            else:
                # Use canonical labeling function
                label, frac = get_uniform_window_label(sub)

                yield Window(
                    window_id=wid,
                    t_start=t_start,
                    t_end=t_end,
                    idx_start=int(idx[i]),
                    idx_end=int(idx[j - 1]),
                    label_window=label,
                    frac_attack=frac,
                )

        t_start += stride_seconds

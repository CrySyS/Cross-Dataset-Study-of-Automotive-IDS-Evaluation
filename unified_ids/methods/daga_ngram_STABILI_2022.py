"""DAGA N-Gram Intrusion Detection (Stabili et al., 2022)

Detection Mechanism:
---------------------
DAGA extracts n-grams (sliding windows of length n over CAN IDs) and checks if each
n-gram exists in the trained model:
- Training: Extracts all n-grams from benign traces and stores them in a set
- Detection: For each incoming message, extracts the n-gram ending with that message.
  If the n-gram is NOT in the model, raises an anomaly for that message.

Granularity:
------------
Although DAGA uses a "sliding window" approach to extract n-grams, the atomic decision
unit is PER-MESSAGE, not per-window:
- Each message i triggers evaluation of the n-gram [msg_{i-n+1}, ..., msg_i]
- The decision (anomaly or not) is attributed to message i (the newest message)
- This matches the paper's evaluation which uses per-message labels (ANOMALY field)

Evaluation:
-----------
The paper defines TP/FP/FN in terms of "instances" (messages):
- TP: non-legit messages correctly identified as anomalies
- FP: legit messages erroneously detected as anomalies  
- FN: non-legit messages erroneously identified as legit

The code does NOT include evaluation logic, only detection. Our analysis of their
dataset structure (which has per-message ANOMALY labels) confirms that each n-gram
decision maps to the message that completed it.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Dict, Iterator, List, Tuple, Optional, Iterable, Deque, Union
import math
import numpy as np
import pandas as pd
from unified_ids.methods.base import BaseFlowIDS
from unified_ids.dataio.windowing import Window

import logging

logger = logging.getLogger(__name__)

Symbol = Tuple[int, int]  # (is_ext_flag, raw_can_id)

# ---- columns we actually need (keep this tight) ----
CORE_COLS = ["timestamp", "can_id"]
OPT_COLS = ["frame_type", "label", "trace_name"]  # included if present


def _row_symbol_from_record(can_id: int, frame_type: Optional[str]) -> Symbol:
    # Robust extended/standard detection:
    # 1) trusted 'is_extended' column if you add it later (not used here)
    # 2) else, by numeric range (29-bit > 0x7FF)
    is_ext = 0
    if frame_type is not None:
        s = str(frame_type).lower()
        if s.startswith("ext"):
            is_ext = 1
    if is_ext == 0 and int(can_id) > 0x7FF:
        is_ext = 1
    return (is_ext, int(can_id))

def _temp_code(sym: Symbol) -> int:
    # deterministic 32-bit mix; disjoint high bit
    x = (sym[0] << 29) | (sym[1] & 0x1FFFFFFF)
    x ^= (x >> 13)
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    return x | (1 << 31)

@dataclass
class DagaParams:
    n: int = 6
    id_remap: bool = True
    window_seconds: float = 1.0
    stride_seconds: float = 1.0
    window_score: str = "any"
    keep_unmap: bool = False  # new: save reverse map only if needed

PackedKey = Union[int, bytes]  # 64-bit int (fast path) or bytes (fallback)

class DagaNGram(BaseFlowIDS):
    def __init__(self, params: Optional[DagaParams] = None):
        self.p = params or DagaParams()
        self._model: set[PackedKey] = set()
        self._id_map: Dict[Symbol, int] = {}
        self._id_unmap: List[Symbol] = []
        self._fitted = False
        # packing config
        self._bits_per_symbol: int = 0
        self._pack_fits_u64: bool = True
        self._mask_u64: int = 0

    # ---- packing helpers --------------------------------------------------
    def _pack_key_u64_init(self, n: int, u: int):
        b = max(1, math.ceil(math.log2(max(1, u))))
        self._bits_per_symbol = b
        self._pack_fits_u64 = (b * n) <= 64
        if self._pack_fits_u64:
            self._mask_u64 = (1 << (b * n)) - 1

    def _pack_push_u64(self, rolling: int, sym_code: int) -> int:
        b = self._bits_per_symbol
        return ((rolling << b) | (sym_code & ((1 << b) - 1))) & self._mask_u64

    def _pack_bytes_from_window(self, window: Deque[int]) -> bytes:
        # pack each code into the minimum bytes that keep it exact
        b = self._bits_per_symbol
        width = 1 if b <= 8 else (2 if b <= 16 else 4)
        return b"".join(int(x).to_bytes(width, "big", signed=False) for x in window)

    # ---- streaming mappers ------------------------------------------------
    def _iter_trace_symbols(self, g: pd.DataFrame) -> Iterator[Symbol]:
        has_ft = "frame_type" in g.columns
        if has_ft:
            for cid, ft in g[["can_id", "frame_type"]].itertuples(index=False):
                yield _row_symbol_from_record(cid, ft)
        else:
            for (cid,) in g[["can_id"]].itertuples(index=False):
                yield _row_symbol_from_record(cid, None)

    def _iter_codes(self, symbols: Iterable[Symbol], write_new: bool, *, return_unseen: bool = False) -> Iterator[Tuple[int, bool] | int]:
        """Map symbols to integer codes.

        If return_unseen is True (used at inference), yield (code, is_unseen) so callers
        can flag an anomaly immediately when an unseen ID appears. Training keeps the
        previous behavior and ignores unseen since it builds the map first.
        """
        if not self.p.id_remap:
            for s in symbols:
                code = (s[0] << 29) | (int(s[1]) & 0x1FFFFFFF)
                yield (code, False) if return_unseen else code
            return
        for s in symbols:
            if s in self._id_map:
                code = self._id_map[s]
                unseen = False
            else:
                if write_new:
                    code = len(self._id_map)
                    self._id_map[s] = code
                    if self.p.keep_unmap:
                        self._id_unmap.append(s)
                    unseen = False
                else:
                    code = _temp_code(s)
                    unseen = True
            yield (code, unseen) if return_unseen else code

    # ---- BaseFlowIDS API --------------------------------------------------

    def fit(self, df_train: pd.DataFrame):
        self._model.clear()
        self._id_map.clear()
        self._id_unmap.clear()
        self._fitted = False
        if df_train.empty:
            self._fitted = True
            return self

        n = self.p.n

        # group by trace_name if available, else single trace
        if "trace_name" in df_train.columns:
            traces = (g for _, g in df_train.sort_values(
                ["trace_name", "timestamp"], kind="stable"
            ).groupby("trace_name", sort=False))
        else:
            traces = [df_train.sort_values("timestamp", kind="stable")]

        # ---- Pass 1: learn id_map (u) without storing sequences ------------
        for g in traces:
            for s in self._iter_trace_symbols(g):
                if not self.p.id_remap:
                    continue
                if s not in self._id_map:
                    self._id_map[s] = len(self._id_map)
                    if self.p.keep_unmap:
                        self._id_unmap.append(s)

        u = len(self._id_map) if self.p.id_remap else (1 << 30)  # upper bound for packing decision
        self._pack_key_u64_init(n, u)

        # re-iterate: need fresh generator(s)
        if "trace_name" in df_train.columns:
            traces = (g for _, g in df_train.sort_values(
                ["trace_name", "timestamp"], kind="stable"
            ).groupby("trace_name", sort=False))
        else:
            traces = [df_train.sort_values("timestamp", kind="stable")]

        # ---- Pass 2: stream, pack, insert into model -----------------------
        for g in traces:
            codes = self._iter_codes(self._iter_trace_symbols(g), write_new=False)
            if self._pack_fits_u64:
                roll = 0
                dq: Deque[int] = deque(maxlen=n)
                for c in codes:
                    dq.append(c)
                    roll = self._pack_push_u64(roll, c)
                    if len(dq) == n:
                        self._model.add(roll)
            else:
                dq: Deque[int] = deque(maxlen=n)
                for c in codes:
                    dq.append(c)
                    if len(dq) == n:
                        self._model.add(self._pack_bytes_from_window(dq))

        self._fitted = True
        return self

    def score_messages(self, df: pd.DataFrame) -> Iterator[Tuple[str, int, int]]:
        """Yield (message_id, score, label) for each message.
        
        Per-Message Attribution:
        ------------------------
        For each message at position k, we evaluate the n-gram ending at k:
            n-gram = [msg_{k-n+1}, msg_{k-n+2}, ..., msg_k]
        
        The score is attributed to message k (the newest message in the n-gram):
        - score = 1 if n-gram NOT in trained model (anomaly)
        - score = 0 if n-gram IS in trained model (benign)
        
        The first (n-1) messages receive score=0 since we don't have a complete
        n-gram yet (matches DAGA paper's behavior of raising anomalies only after
        accumulating n messages).
        
        Returns:
        --------
        Iterator of (message_id, score, label) tuples where:
        - message_id: str, DataFrame index as string (for attack_type lookup)
        - score: int (0 or 1), 1 indicates anomaly
        - label: int (0 or 1), ground truth label if available
        """
        assert self._fitted, "Call fit() before score_messages()."
        if df.empty:
            return

        if "trace_name" in df.columns:
            traces = (g for _, g in df.sort_values(
                ["trace_name", "timestamp"], kind="stable"
            ).groupby("trace_name", sort=False))
        else:
            traces = [df.sort_values("timestamp", kind="stable")]

        n = self.p.n

        for g in traces:
            idx = g.index.to_numpy(copy=False)  # Get DataFrame indices
            y = g["label"].to_numpy(dtype=int, copy=False) if "label" in g.columns else None

            codes = self._iter_codes(self._iter_trace_symbols(g), write_new=False, return_unseen=True)

            if self._pack_fits_u64:
                roll = 0
                dq: Deque[int] = deque(maxlen=n)
                for k, (c, unseen) in enumerate(codes):
                    dq.append(c)
                    roll = self._pack_push_u64(roll, c)
                    if unseen:
                        yield (str(idx[k]), 1, int(y[k]) if y is not None else 0)
                        continue
                    if len(dq) < n:
                        yield (str(idx[k]), 0, int(y[k]) if y is not None else 0)
                    else:
                        yield (str(idx[k]), int(roll not in self._model), int(y[k]) if y is not None else 0)
            else:
                dq: Deque[int] = deque(maxlen=n)
                for k, (c, unseen) in enumerate(codes):
                    dq.append(c)
                    if unseen:
                        yield (str(idx[k]), 1, int(y[k]) if y is not None else 0)
                        continue
                    if len(dq) < n:
                        yield (str(idx[k]), 0, int(y[k]) if y is not None else 0)
                    else:
                        key = self._pack_bytes_from_window(dq)
                        yield (str(idx[k]), int(key not in self._model), int(y[k]) if y is not None else 0)

    def score_windows(self, df_test: pd.DataFrame) -> Iterator[Tuple[Window, float]]:
        """
        Yield (Window, score) over sliding time windows per trace.
        score:
          - "any"  -> 1.0 if any anomalous msg in window else 0.0
          - "frac" -> (# anomalous msgs in window) / (# msgs in window)
        """
        assert self._fitted, "Call fit() before score_windows()."
        if df_test.empty:
            return

        # group per trace, sorted by time (no cross-trace windows)
        if "trace_name" in df_test.columns:
            grouped = (
                df_test.sort_values(["trace_name", "timestamp"], kind="stable")
                      .groupby("trace_name", sort=False)
            )
            traces = ((name, g.sort_values("timestamp", kind="stable").reset_index(drop=True)) for name, g in grouped)
        else:
            traces = [(None, df_test.sort_values("timestamp", kind="stable").reset_index(drop=True))]

        window_sec = float(self.p.window_seconds)
        stride = float(self.p.stride_seconds)
        mode = str(self.p.window_score).lower()

        for trace_name, g in traces:
            # Collect per-message outputs for this trace (still streaming-ish; one trace at a time)
            ts_list: List[float] = []
            y_list: List[int] = []
            a_list: List[int] = []  # anomaly flags
            timestamps = g["timestamp"].to_numpy(dtype=float, copy=False)
            indices = g.index.to_numpy()

            for msg_id, a, y in self.score_messages(g):
                # msg_id is the string representation of the DataFrame index
                # Since g has been reset_index(), indices are 0, 1, 2, ...
                k = int(msg_id)
                ts_list.append(timestamps[k])
                a_list.append(int(a))
                y_list.append(int(y))

            if not ts_list:
                continue

            ts = np.asarray(ts_list, dtype=float)
            A = np.asarray(a_list, dtype=np.int32)
            Y = np.asarray(y_list, dtype=np.int8)

            # Sliding window via two pointers
            start_time = ts[0]
            end_time = ts[-1]
            i0 = 0
            i1 = 0
            cur_start = start_time

            while cur_start <= end_time:
                cur_end = cur_start + window_sec

                # advance i0 to first idx with ts[i0] >= cur_start
                while i0 < len(ts) and ts[i0] < cur_start:
                    i0 += 1
                # advance i1 to first idx with ts[i1] >= cur_end  (window is [cur_start, cur_end))
                while i1 < len(ts) and ts[i1] < cur_end:
                    i1 += 1

                if i0 < i1:  # there is at least one message in the window
                    n_msgs = i1 - i0
                    n_anom = int(A[i0:i1].sum())
                    has_attack = int((Y[i0:i1].max()) > 0)
                    frac_attack = float(Y[i0:i1].mean()) if n_msgs > 0 else 0.0

                    if mode == "frac":
                        score = n_anom / float(n_msgs)
                    else:  # "any" (default)
                        score = 1.0 if n_anom > 0 else 0.0

                    idx_start = int(indices[i0])
                    idx_end = int(indices[i1 - 1])
                    t_start = float(ts[i0])
                    t_end = float(ts[i1 - 1])

                    if trace_name is None:
                        win_id = f"{idx_start}:{idx_end}"
                    else:
                        win_id = f"{trace_name}:{idx_start}-{idx_end}"

                    w = Window(
                        window_id=win_id,
                        idx_start=idx_start,
                        idx_end=idx_end,
                        t_start=t_start,
                        t_end=t_end,
                        label_window=has_attack,
                        frac_attack=frac_attack,
                    )

                    yield (w, float(score))

                cur_start += stride

    def paper_eval(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        out_dir: Optional[str] = None,
    ):
        """
        Paper-specific evaluation reproducing DAGA paper figures and tables.
        
        This expects the DAGA dataset structure with specific attack types
        in trace_name or filename. If not DAGA data, returns a simple note.
        
        Returns dict with keys: single_id_replay, ordered_replay, 
                                arbitrary_replay, dos_lowest_id
        """
        import os
        import numpy as np
        from glob import glob
        
        # Check if this looks like DAGA dataset
        if "trace_name" in df_test.columns:
            trace_names = df_test["trace_name"].unique()
            is_daga = any("SingleIDReplay" in str(t) or "OrderedSequence" in str(t) 
                         or "ArbitrarySequence" in str(t) or "DenialOfService" in str(t)
                         for t in trace_names)
        else:
            is_daga = False
        
        if not is_daga:
            logger.info("paper_eval: Dataset doesn't appear to be DAGA-specific; skipping paper eval")
            return {
                "note": "Paper eval requires DAGA dataset with specific attack types",
                "dataset_detected": "generic"
            }
        
        logger.info("paper_eval: Detected DAGA dataset, running paper-specific evaluation")
        
        # Import paper eval helpers
        from unified_ids.methods.helpers import daga_paper_eval as daga_eval
        
        # Setup output directory
        if out_dir is None:
            out_dir = "daga_eval"
        os.makedirs(out_dir, exist_ok=True)
        
        results = {}
        
        # Get unique n values to test (paper uses n=1..10)
        # For speed, only evaluate current n unless full_sweep=True
        full_sweep = os.environ.get("DAGA_FULL_N_SWEEP", "").lower() in ("1", "true", "yes")
        if full_sweep:
            n_values = range(1, 11)
            logger.info("paper_eval: Running full n=1..10 sweep (this will take a long time!)")
        else:
            n_values = [self.p.n]
            logger.info(f"paper_eval: Evaluating only current n={self.p.n}. Set DAGA_FULL_N_SWEEP=1 to test n=1..10")
        
        # Group test data by attack type
        single_id_traces = {}
        ordered_seq_traces = {}
        arbitrary_seq_traces = {}
        dos_traces = []
        
        for trace_name in df_test["trace_name"].unique():
            trace_str = str(trace_name)
            if "SingleIDReplay" in trace_str:
                # Extract ID from trace name (e.g., "SingleIDReplay__ID_8_...")
                if "__ID_8" in trace_str:
                    single_id_traces.setdefault("top", []).append(trace_name)
                elif "__ID_145" in trace_str:
                    single_id_traces.setdefault("mid", []).append(trace_name)
                elif "__ID_2C5" in trace_str:
                    single_id_traces.setdefault("low", []).append(trace_name)
                elif "__ID_1_" in trace_str:
                    single_id_traces.setdefault("not", []).append(trace_name)
            elif "OrderedSequence" in trace_str:
                # Extract length (e.g., "OrderedSequenceReplay__n_5_...")
                for L in range(2, 11):
                    if f"__n_{L}_" in trace_str:
                        ordered_seq_traces.setdefault(L, []).append(trace_name)
            elif "ArbitrarySequence" in trace_str:
                for L in range(2, 11):
                    if f"__n_{L}_" in trace_str:
                        arbitrary_seq_traces.setdefault(L, []).append(trace_name)
            elif "DenialOfService" in trace_str and "lowestID" in trace_str:
                dos_traces.append(trace_name)
        
        # Helper to get DataFrame for specific traces
        def get_traces_df(trace_names):
            if not trace_names:
                return pd.DataFrame()
            return df_test[df_test["trace_name"].isin(trace_names)].copy()
        
        # Model factory that retrains for each n
        def model_factory_for_n(n: int):
            from unified_ids.methods.daga_ngram_STABILI_2022 import DagaNGram, DagaParams
            model = DagaNGram(DagaParams(
                n=n,
                id_remap=True,
                window_seconds=1.0,
                stride_seconds=1.0,
                window_score="any"
            ))
            model.fit(df_train)
            return model
        
        # Convert trace-based groups to DataFrame-based groups for eval helpers
        single_id_dfs = {k: [get_traces_df([t]) for t in v] 
                        for k, v in single_id_traces.items() if v}
        ordered_seq_dfs = {L: [get_traces_df([t]) for t in v] 
                          for L, v in ordered_seq_traces.items() if v}
        arbitrary_seq_dfs = {L: [get_traces_df([t]) for t in v] 
                            for L, v in arbitrary_seq_traces.items() if v}
        dos_dfs = [get_traces_df([t]) for t in dos_traces] if dos_traces else []
        
        # Helper to compute F1 for a trace
        def compute_f1_for_trace(model, df_trace):
            """Compute window-level F1 for a single trace."""
            from sklearn.metrics import f1_score
            try:
                # score_windows returns an iterator of (trace_name, score, label)
                results = list(model.score_windows(df_trace))
                if not results:
                    return 0.0
                y_true = [label for _, _, label in results]
                y_pred = [int(score) for _, score, _ in results]
                return f1_score(y_true, y_pred, zero_division=0)
            except Exception as e:
                logger.warning(f"Error computing F1 for trace: {e}")
                return 0.0
        
        # Run evaluations with F-score computation for different n values
        if single_id_dfs:
            logger.info("Single-ID replay: computing F-scores for n=1..10")
            f1_by_group_n = {}
            for group_name, trace_dfs in single_id_dfs.items():
                f1_by_group_n[group_name] = {}
                for n in n_values:
                    model_n = model_factory_for_n(n)
                    f1_scores = [compute_f1_for_trace(model_n, df) for df in trace_dfs]
                    f1_by_group_n[group_name][n] = {
                        "f1_scores": f1_scores,
                        "median_f1": float(np.median(np.array(f1_scores))) if f1_scores else 0.0,
                        "mean_f1": float(np.mean(np.array(f1_scores))) if f1_scores else 0.0
                    }
            results["single_id_replay"] = {
                "groups": list(single_id_dfs.keys()),
                "n_traces": {k: len(v) for k, v in single_id_dfs.items()},
                "f1_by_group_and_n": f1_by_group_n
            }
            logger.info("Single-ID replay: %d groups evaluated", len(single_id_dfs))
        
        if ordered_seq_dfs:
            logger.info("Ordered sequence replay: computing F-scores for n=1..10")
            f1_by_length_n = {}
            for length, trace_dfs in ordered_seq_dfs.items():
                f1_by_length_n[length] = {}
                for n in n_values:
                    model_n = model_factory_for_n(n)
                    f1_scores = [compute_f1_for_trace(model_n, df) for df in trace_dfs]
                    f1_by_length_n[length][n] = {
                        "f1_scores": f1_scores,
                        "median_f1": float(np.median(np.array(f1_scores))) if f1_scores else 0.0,
                        "mean_f1": float(np.mean(np.array(f1_scores))) if f1_scores else 0.0
                    }
            results["ordered_replay"] = {
                "lengths": list(ordered_seq_dfs.keys()),
                "n_traces": {L: len(v) for L, v in ordered_seq_dfs.items()},
                "f1_by_length_and_n": f1_by_length_n
            }
            logger.info("Ordered sequence replay: %d length groups evaluated", len(ordered_seq_dfs))
        
        if arbitrary_seq_dfs:
            logger.info("Arbitrary sequence replay: computing F-scores for n=1..10")
            f1_by_length_n = {}
            for length, trace_dfs in arbitrary_seq_dfs.items():
                f1_by_length_n[length] = {}
                for n in n_values:
                    model_n = model_factory_for_n(n)
                    f1_scores = [compute_f1_for_trace(model_n, df) for df in trace_dfs]
                    f1_by_length_n[length][n] = {
                        "f1_scores": f1_scores,
                        "median_f1": float(np.median(np.array(f1_scores))) if f1_scores else 0.0,
                        "mean_f1": float(np.mean(np.array(f1_scores))) if f1_scores else 0.0
                    }
            results["arbitrary_replay"] = {
                "lengths": list(arbitrary_seq_dfs.keys()),
                "n_traces": {L: len(v) for L, v in arbitrary_seq_dfs.items()},
                "f1_by_length_and_n": f1_by_length_n
            }
            logger.info("Arbitrary sequence replay: %d length groups evaluated", len(arbitrary_seq_dfs))
        
        if dos_dfs:
            logger.info("DoS (lowest-ID) evaluation: computing F-scores for n=1..10")
            f1_by_n = {}
            for n in n_values:
                model_n = model_factory_for_n(n)
                f1_scores = [compute_f1_for_trace(model_n, df) for df in dos_dfs]
                f1_by_n[n] = {
                    "f1_scores": f1_scores,
                    "median_f1": float(np.median(np.array(f1_scores))) if f1_scores else 0.0,
                    "mean_f1": float(np.mean(np.array(f1_scores))) if f1_scores else 0.0
                }
            results["dos_lowest_id"] = {
                "n_traces": len(dos_dfs),
                "f1_by_n": f1_by_n
            }
            logger.info("DoS evaluation: %d traces evaluated", len(dos_dfs))
        
        results["summary"] = {
            "dataset": "DAGA",
            "n_values_tested": list(n_values),
            "current_n": self.p.n,
            "output_dir": out_dir,
            "note": "Set DAGA_FULL_N_SWEEP=1 environment variable to test all n=1..10 (slow)"
        }
        
        return results


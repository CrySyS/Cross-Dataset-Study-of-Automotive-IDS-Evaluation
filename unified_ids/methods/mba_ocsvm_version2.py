# unified_ids/methods/avatefipour_ocsvm_mba.py
# ---------------------------------------------------------------------
# Paper-faithful (unsupervised) Avatefipour et al. (2019) MBA-OCSVM
# with a paper-style evaluation protocol that handles OTIDS properly.
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, List, Dict, Any
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

from collections import deque
from pandas.api.types import is_numeric_dtype


from unified_ids.methods.base import BaseFlowIDS
from unified_ids.dataio.windowing import Window, get_uniform_window_label, windows_fixed_time


logger = logging.getLogger("unified_ids")


# ----------------------------
# Parameters
# ----------------------------

@dataclass
class AvatefipourParams:
    # Feature window definition
    window_seconds: float = 0.5
    stride_seconds: float = 0.5  # recommended for speed / non-overlap

    # Practical scikit-learn mapping of paper's (sigma, C) -> (gamma, nu)
    nu_bounds: Tuple[float, float] = (0.001, 0.2)
    gamma_bounds: Tuple[float, float] = (1e-6, 1e2)

    # MBA parameters (paper: pop=25, iters=100, lambda=gamma=0.2)
    pop_size: int = 25
    iters: int = 100
    loudness_decay: float = 0.2
    pulse_gamma: float = 0.2

    # Optional boundary complexity penalty (0.0 = strictest)
    sv_penalty: float = 0.0

    # Practical speed caps (optional; keep None for full)
    max_train_windows: Optional[int] = None
    max_val_windows: Optional[int] = None

    # Reproducibility
    random_state: int = 0


# ----------------------------
# Helpers
# ----------------------------

def _ensure_sorted_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    assert df["timestamp"].is_monotonic_increasing, (
        "timestamp must be monotonic increasing; "
    )
    return df



def _id_freq_points(sub: pd.DataFrame, window_seconds: float) -> np.ndarray:
    if sub.empty:
        return np.zeros((0, 2), dtype=np.float64)
    counts = sub["can_id"].value_counts(sort=False)
    can_ids = counts.index.to_numpy(dtype=np.float64)
    freqs = counts.to_numpy(dtype=np.float64) / float(window_seconds)
    return np.column_stack([can_ids, freqs]).astype(np.float64, copy=False)


def _window_score(model: OneClassSVM, Xw_scaled: np.ndarray) -> float:
    # higher = more anomalous
    if Xw_scaled.size == 0:
        return 0.0
    s = -model.decision_function(Xw_scaled).ravel()
    return float(np.max(s))


def _window_pred_anom(model: OneClassSVM, Xw_scaled: np.ndarray) -> int:
    """
    Paper-faithful decision rule at window level:
      mark window anomalous if ANY (ID,freq) point is outlier (-1).
    """
    if Xw_scaled.size == 0:
        return 0
    pred = model.predict(Xw_scaled)  # +1 inlier, -1 outlier
    return int(np.any(pred == -1))


def _build_windows_slow(df: pd.DataFrame, window_seconds: float, stride_seconds: float):
    X_list: List[np.ndarray] = []
    windows: List[Window] = []
    sub_list: List[pd.DataFrame] = []
    for w in windows_fixed_time(df, span_seconds=float(window_seconds), stride_seconds=float(stride_seconds)):
        sub = df.iloc[w.idx_start : w.idx_end+1]
        if sub.empty:
            continue
        X_list.append(_id_freq_points(sub, window_seconds))
        windows.append(w)
        sub_list.append(sub)
    return X_list, windows, sub_list


def _build_windows_fast_nonoverlap(df: pd.DataFrame, window_seconds: float):
    """
    Fast path for stride == window: single groupby over bins.
    Returns:
      X_windows, window_ids, df_windows (for labeling).
    """
    df = _ensure_sorted_timestamp(df)
    t = df["timestamp"].to_numpy(dtype=np.float64)
    if len(t) == 0:
        return [], [], []
    t0 = float(t[0])
    bins = np.floor((t - t0) / float(window_seconds)).astype(np.int64)

    tmp = df[["can_id"]].copy()
    tmp["bin"] = bins
    counts = tmp.groupby(["bin", "can_id"], sort=False).size()

    # unpack to per-bin matrices
    b = counts.index.get_level_values(0).to_numpy()
    cid = counts.index.get_level_values(1).to_numpy(dtype=np.int64)
    cts = counts.to_numpy(dtype=np.float64)

    order = np.argsort(b, kind="mergesort")
    b, cid, cts = b[order], cid[order], cts[order]

    uniq_bins, start = np.unique(b, return_index=True)
    start = list(start) + [len(b)]

    X_list: List[np.ndarray] = []
    windows: List[Window] = []
    sub_list: List[pd.DataFrame] = []
    W = float(window_seconds)

    for ub, a, z in zip(uniq_bins, start[:-1], start[1:]):
        X = np.column_stack([cid[a:z].astype(np.float64), (cts[a:z] / W)]).astype(np.float64, copy=False)
        X_list.append(X)
        sub = df.loc[bins == int(ub)]
        sub_list.append(sub)

        # Build deterministic Window using index span and timestamps
        t_start = float(sub["timestamp"].iloc[0])
        t_end = float(sub["timestamp"].iloc[-1])
        lab, frac = get_uniform_window_label(sub)
        win = Window(
            window_id=f"bin_{int(ub)}",
            idx_start=int(sub.index[0]),
            idx_end=int(sub.index[-1]),
            t_start=t_start,
            t_end=t_end,
            label_window=int(lab),
            frac_attack=float(frac),
        )
        windows.append(win)

    return X_list, windows, sub_list


def build_id_freq_windows(df: pd.DataFrame, window_seconds: float, stride_seconds: float):
    df = _ensure_sorted_timestamp(df)
    if float(stride_seconds) == float(window_seconds):
        return _build_windows_fast_nonoverlap(df, window_seconds)
    return _build_windows_slow(df, window_seconds, stride_seconds)


# ----------------------------
# Unsupervised 2D MBA tuner (nu, gamma)
# ----------------------------

def mba_tune_nu_gamma_unsupervised(
    X_train: np.ndarray,
    X_val: np.ndarray,
    *,
    nu_bounds: Tuple[float, float],
    gamma_bounds: Tuple[float, float],
    pop_size: int,
    iters: int,
    loudness_decay: float,
    pulse_gamma: float,
    sv_penalty: float,
    random_state: int,
    log_every_iters: int = 0,
) -> Tuple[float, float, float]:
    """
    Unsupervised MBA: minimize false alarms on NORMAL validation data.
    Returns: (nu*, gamma*, best_fitness) with fitness = normal FAR (+ optional SV penalty).
    """
    rng = np.random.default_rng(random_state)

    nu_lo, nu_hi = float(nu_bounds[0]), float(nu_bounds[1])
    g_lo, g_hi = float(gamma_bounds[0]), float(gamma_bounds[1])
    z_lo, z_hi = np.log10(g_lo), np.log10(g_hi)

    def fitness(nu: float, z: float) -> float:
        nu = float(np.clip(nu, nu_lo, nu_hi))
        z = float(np.clip(z, z_lo, z_hi))
        gamma = float(10.0 ** z)

        mdl = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(X_train)
        pred = mdl.predict(X_val)
        far = float(np.mean(pred == -1))

        if sv_penalty and sv_penalty > 0.0:
            sv_frac = float(mdl.support_.shape[0]) / float(X_train.shape[0])
            far += float(sv_penalty) * sv_frac

        return far

    # positions: [z, nu]
    X = np.zeros((pop_size, 2), dtype=np.float64)
    V = np.zeros((pop_size, 2), dtype=np.float64)
    A = np.ones(pop_size, dtype=np.float64)
    r = np.full(pop_size, 0.5, dtype=np.float64)

    X[:, 0] = rng.uniform(z_lo, z_hi, size=pop_size)
    X[:, 1] = rng.uniform(nu_lo, nu_hi, size=pop_size)

    fit = np.array([fitness(X[j, 1], X[j, 0]) for j in range(pop_size)], dtype=np.float64)
    best = int(np.argmin(fit))
    Xbest = X[best].copy()
    fbest = float(fit[best])

    for it in range(iters):
        F = rng.random(pop_size)

        # Standard BA step
        for j in range(pop_size):
            V[j] = V[j] + F[j] * (Xbest - X[j])
            Xcand = X[j] + V[j]

            if rng.random() > r[j]:
                Xcand = Xbest + 0.5 * A[j] * (rng.random(2) - 0.5)

            Xcand[0] = np.clip(Xcand[0], z_lo, z_hi)
            Xcand[1] = np.clip(Xcand[1], nu_lo, nu_hi)

            f_new = fitness(Xcand[1], Xcand[0])

            if (rng.random() < A[j]) and (f_new < fit[j]):
                X[j] = Xcand
                fit[j] = f_new
                if f_new < fbest:
                    Xbest = Xcand.copy()
                    fbest = float(f_new)

            A[j] *= (1.0 - loudness_decay)
            r[j] = r[j] + (1.0 - np.exp(-pulse_gamma * (it + 1))) * (1.0 - r[j])

        # Lightweight "modified" steps (diversity + mean guidance)
        AD = X.mean(axis=0)
        for j in range(pop_size):
            z1, z2, z3 = rng.choice(pop_size, size=3, replace=False)
            Xmut = X[z1] + rng.random() * (X[z2] - X[z3])
            
            # crossover with best
            mask = rng.random(2) < rng.random(2)
            Xtest1 = np.where(mask, Xmut, Xbest)

            # move toward best from mutation
            theta3, theta4 = rng.random(), rng.random()
            Xtest2 = theta3 * Xmut + theta4 * (Xbest - Xmut)

            # mean-guided step
            phiF = 1.0 if rng.random() < 0.5 else 2.0
            theta5 = rng.random()
            Xtest3 = X[j] + theta5 * (Xbest - phiF * AD)

            for Xt in (Xtest1, Xtest2, Xtest3):
                Xt = Xt.astype(np.float64, copy=False)
                Xt[0] = np.clip(Xt[0], z_lo, z_hi)
                Xt[1] = np.clip(Xt[1], nu_lo, nu_hi)
                f_new = fitness(Xt[1], Xt[0])
                if f_new < fit[j]:
                    X[j] = Xt
                    fit[j] = f_new
                    if f_new < fbest:
                        Xbest = Xt.copy()
                        fbest = float(f_new)

        if log_every_iters > 0 and (((it + 1) % log_every_iters == 0) or (it + 1 == iters)):
            logger.info(
                "mba_ocsvm_v2 tune progress: iter %d/%d, best_normal_val_far=%.6f",
                it + 1,
                iters,
                fbest,
            )

    nu_star = float(np.clip(Xbest[1], nu_lo, nu_hi))
    gamma_star = float(10.0 ** float(np.clip(Xbest[0], z_lo, z_hi)))
    return nu_star, gamma_star, fbest


# ----------------------------
# IDS
# ----------------------------

class AvatefipourOCSVM_MBA(BaseFlowIDS):
    """
    - fit(): uses ONLY normal traffic; MBA tuning is normal-only (paper-faithful)
    - score_windows(): framework-uniform window scoring (score + window label)
    - paper_eval(): OTIDS-aware paper-style evaluation:
        * split normal into train/val/test
        * tune on normal train/val
        * report FAR on normal test (always defined)
        * report attack-window detection rate on df_test (always defined)
        * optionally compute "constructed mixed-stream" AUC by concatenating
          normal-holdout windows with attack windows (clearly marked)
    """

    def __init__(self, params: Optional[AvatefipourParams] = None):
        self.params = params or AvatefipourParams()
        self.scaler: Optional[MinMaxScaler] = None
        self.model: Optional[OneClassSVM] = None
        self.nu_: Optional[float] = None
        self.gamma_: Optional[float] = None
        self.mba_best_far_: Optional[float] = None

    def fit(self, df_train: pd.DataFrame):
        p = self.params
        if df_train.empty:
            raise ValueError("df_train is empty")

        logger.info("mba_ocsvm_v2 fit: start (rows=%d)", len(df_train))

        # normal-only (paper-faithful)
        if "label" in df_train.columns:
            df_norm = df_train[df_train["label"] == 0].copy()
        else:
            df_norm = df_train.copy()
        df_norm = _ensure_sorted_timestamp(df_norm)
        if df_norm.empty:
            raise ValueError("No normal rows available for training")

        # ---- hard assumptions (fail loud) ----
        assert "timestamp" in df_norm.columns, "df_train must contain 'timestamp'"
        assert "can_id" in df_norm.columns, "df_train must contain 'can_id'"
        assert df_norm["timestamp"].is_monotonic_increasing, (
            "Assumption violated: df_train['timestamp'] must be monotonic increasing "
            "(fit() will NOT sort/reset index)."
        )
        assert pd.api.types.is_numeric_dtype(df_norm["timestamp"]), "timestamp must be numeric dtype"
        assert pd.api.types.is_numeric_dtype(df_norm["can_id"]), (
            "can_id must be numeric dtype (decimal IDs). Convert hex strings before fit()."
        )
        assert df_norm["timestamp"].notna().all(), "timestamp contains NaN"
        assert df_norm["can_id"].notna().all(), "can_id contains NaN"

        # build windows -> points per window
        Xw, _, _ = build_id_freq_windows(df_norm, p.window_seconds, p.stride_seconds)
        if not Xw:
            raise ValueError("No windows could be built from training data")

        logger.info("mba_ocsvm_v2 fit: built %d normal windows", len(Xw))

        # optional cap (still fail-loud in the 'too few windows' sense)
        rng = np.random.default_rng(p.random_state)
        if p.max_train_windows is not None and len(Xw) > p.max_train_windows:
            idx_cap = rng.choice(len(Xw), size=int(p.max_train_windows), replace=False)
            idx_cap = np.sort(idx_cap)  # keep deterministic ordering
            Xw = [Xw[i] for i in idx_cap]

        assert len(Xw) >= 2, (
            f"Need >=2 windows for train/val split; got {len(Xw)}. "
            "Reduce window_seconds/stride_seconds or provide more data."
        )

        # --- window-level split to avoid leakage (like paper_eval) ---
        idx = np.arange(len(Xw))
        rng.shuffle(idx)

        n_train_w = max(1, int(0.90 * len(idx)))
        assert n_train_w < len(idx), "Train split consumed all windows; increase data or adjust split"
        train_w_idx = idx[:n_train_w]
        val_w_idx = idx[n_train_w:]

        # Fit scaler on TRAIN windows' points only
        X_train_pts = np.vstack([Xw[i] for i in train_w_idx]).astype(np.float64, copy=False)
        assert np.isfinite(X_train_pts).all(), "Non-finite values in training features"
        self.scaler = MinMaxScaler().fit(X_train_pts)
        Xtr = self.scaler.transform(X_train_pts)

        # Validation points from VAL windows
        X_val_pts = np.vstack([Xw[i] for i in val_w_idx]).astype(np.float64, copy=False)
        assert np.isfinite(X_val_pts).all(), "Non-finite values in validation features"
        Xv = self.scaler.transform(X_val_pts)

        logger.info(
            "mba_ocsvm_v2 fit: train windows=%d, val windows=%d, train pts=%d, val pts=%d",
            len(train_w_idx),
            len(val_w_idx),
            Xtr.shape[0],
            Xv.shape[0],
        )

        # MBA tune
        nu_star, gamma_star, best_far = mba_tune_nu_gamma_unsupervised(
            X_train=Xtr,
            X_val=Xv,
            nu_bounds=p.nu_bounds,
            gamma_bounds=p.gamma_bounds,
            pop_size=p.pop_size,
            iters=p.iters,
            loudness_decay=p.loudness_decay,
            pulse_gamma=p.pulse_gamma,
            sv_penalty=p.sv_penalty,
            random_state=p.random_state,
            log_every_iters=max(1, p.iters // 5),
        )

        # Fit final model on TRAIN+VAL windows points using SAME scaler
        X_fit_pts = np.vstack([Xw[i] for i in idx]).astype(np.float64, copy=False)
        assert np.isfinite(X_fit_pts).all(), "Non-finite values in fit features"
        X_fit = self.scaler.transform(X_fit_pts)

        self.nu_, self.gamma_, self.mba_best_far_ = float(nu_star), float(gamma_star), float(best_far)
        self.model = OneClassSVM(kernel="rbf", nu=self.nu_, gamma=self.gamma_).fit(X_fit)
        logger.info(
            "mba_ocsvm_v2 fit: done (nu=%.6g, gamma=%.6g, best_normal_val_far=%.6f)",
            self.nu_,
            self.gamma_,
            self.mba_best_far_,
        )

    def score_windows(self, df_test: pd.DataFrame) -> Iterator[Tuple[Window, float]]:
        p = self.params
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted")

        df_test = _ensure_sorted_timestamp(df_test)
        Xw, windows, subs = build_id_freq_windows(df_test, p.window_seconds, p.stride_seconds)

        n_windows = len(windows)
        logger.info("mba_ocsvm_v2 score_windows: start (n_windows=%d)", n_windows)
        log_stride = max(1, n_windows // 10) if n_windows > 0 else 1

        for i, (X, w, sub) in enumerate(zip(Xw, windows, subs), start=1):
            lab, _ = get_uniform_window_label(sub)
            Xs = self.scaler.transform(X) if X.size > 0 else X
            sc = _window_score(self.model, Xs)

            if i == 1 or i == n_windows or (i % log_stride == 0):
                logger.info("mba_ocsvm_v2 score_windows: %d/%d", i, n_windows)

            # Ensure Window carries correct label metadata
            w = Window(
                window_id=str(w.window_id),
                idx_start=int(w.idx_start),
                idx_end=int(w.idx_end),
                t_start=float(w.t_start),
                t_end=float(w.t_end),
                label_window=int(lab),
                frac_attack=float(w.frac_attack),
            )
            yield (w, float(sc))

    def score_messages(self, df_test: pd.DataFrame) -> Iterator[Tuple[str, float, int]]:
        """
        Paper-style message scoring:
        - For each message at time t with CAN ID = id_t,
            compute freq(id_t) over a trailing window of length window_seconds,
            then score x_t = [id_t, freq(id_t)] with OCSVM decision function.

        Yields: (message_id, score, label_message)
        - message_id: uses the original df index converted to str (stable identifier)
        - score: higher = more anomalous (we use -decision_function)
        - label_message: int in {0,1} from df_test["label"]
        """
        p = self.params
        assert self.model is not None and self.scaler is not None, "Model not fitted"
        assert isinstance(df_test, pd.DataFrame), "df_test must be a pandas DataFrame"
        assert not df_test.empty, "df_test is empty"

        # ---- Hard assumptions (fail fast; no fallbacks) ----
        assert "timestamp" in df_test.columns, "df_test must contain 'timestamp' column"
        assert "can_id" in df_test.columns, "df_test must contain 'can_id' column"
        assert "label" in df_test.columns, "df_test must contain 'label' column"

        assert float(p.window_seconds) > 0.0, "params.window_seconds must be > 0"

        # No sorting/resetting index allowed: must already be time-ordered
        assert df_test["timestamp"].is_monotonic_increasing, (
            "Assumption violated: df_test['timestamp'] must be monotonic increasing "
            "(this implementation will NOT sort or reset index)."
        )

        # Enforce numeric CAN IDs + timestamps (no hex strings here)
        assert is_numeric_dtype(df_test["timestamp"]), "timestamp must be numeric dtype (float/int seconds)"
        assert is_numeric_dtype(df_test["can_id"]), (
            "can_id must be numeric dtype (decimal IDs). If you still have hex strings, "
            "convert them BEFORE calling score_messages."
        )

        # No missing values
        assert df_test["timestamp"].notna().all(), "timestamp contains NaN"
        assert df_test["can_id"].notna().all(), "can_id contains NaN"
        assert df_test["label"].notna().all(), "label contains NaN"

        # Labels must be binary {0,1}
        u = pd.unique(df_test["label"])
        assert set(map(int, u)).issubset({0, 1}), f"label must be in {{0,1}}; got unique={u!r}"

        # Index should be stable identifiers (not required to be consecutive, but should be unique)
        assert df_test.index.is_unique, "df_test index must be unique for stable message_id"

        # ---- Rolling per-ID trailing window counts (O(N)) ----
        w = float(p.window_seconds)
        per_id_times: dict[float, deque] = {}
        n_msgs = len(df_test)
        logger.info("mba_ocsvm_v2 score_messages: start (n_messages=%d)", n_msgs)
        log_stride = max(1, n_msgs // 10) if n_msgs > 0 else 1

        # Use itertuples for speed while preserving index as message_id
        for i, row in enumerate(df_test[["timestamp", "can_id", "label"]].itertuples(index=True, name=None), start=1):
            msg_idx, t, cid, lab = row

            # Additional sanity checks per row (hard fail)
            assert np.isfinite(t), f"Non-finite timestamp at index {msg_idx}: {t}"
            assert np.isfinite(cid), f"Non-finite can_id at index {msg_idx}: {cid}"

            dq = per_id_times.get(cid)
            if dq is None:
                dq = deque()
                per_id_times[cid] = dq

            cutoff = float(t) - w
            while dq and dq[0] < cutoff:
                dq.popleft()
            dq.append(float(t))

            freq = float(len(dq)) / w

            x = np.array([[float(cid), freq]], dtype=np.float64)
            xs = self.scaler.transform(x)
            score = float(-self.model.decision_function(xs).ravel()[0])

            if i == 1 or i == n_msgs or (i % log_stride == 0):
                logger.info("mba_ocsvm_v2 score_messages: %d/%d", i, n_msgs)

            yield (str(msg_idx), score, int(lab))

    def paper_eval(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        out_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Paper-style eval for scenario datasets like OTIDS:
        - Split NORMAL (df_train label==0) into train/val/test (70/10/20) at window level.
        - MBA tune on normal train/val (min FAR on normal).
        - Report FAR on normal test (always defined).
        - Evaluate df_test (attack scenario) at window level:
            * attack-window detection rate (always defined)
            * if both classes exist naturally, compute AUC over window scores
            * if single-class windows, optionally compute constructed-mix AUC by
              concatenating normal test windows + attack windows.
        """
        p = self.params
        if df_train.empty:
            raise ValueError("paper_eval: df_train is empty")
        if df_test.empty:
            raise ValueError("paper_eval: df_test is empty")

        logger.info("mba_ocsvm_v2 paper_eval: start (train_rows=%d, test_rows=%d)", len(df_train), len(df_test))

        # 1) Prepare normal-only data from df_train
        if "label" in df_train.columns:
            df_norm = df_train[df_train["label"] == 0].copy()
        else:
            df_norm = df_train.copy()

        df_norm = _ensure_sorted_timestamp(df_norm)
        if df_norm.empty:
            raise ValueError("paper_eval: no normal rows in df_train")

        # build normal windows (for split)
        Xn, nids, nsubs = build_id_freq_windows(df_norm, p.window_seconds, p.stride_seconds)
        if not Xn:
            raise ValueError("paper_eval: could not build normal windows")
        logger.info("mba_ocsvm_v2 paper_eval: built %d normal windows", len(Xn))

        # split windows (70/10/20)
        rng = np.random.default_rng(p.random_state)
        idx = np.arange(len(Xn))
        rng.shuffle(idx)

        n_train = max(1, int(0.70 * len(idx)))
        n_val = max(1, int(0.10 * len(idx)))
        n_test = max(1, len(idx) - n_train - n_val)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        # optional caps (applied after split to preserve protocol)
        if p.max_train_windows is not None and len(train_idx) > p.max_train_windows:
            train_idx = rng.choice(train_idx, size=int(p.max_train_windows), replace=False)
        if p.max_val_windows is not None and len(val_idx) > p.max_val_windows:
            val_idx = rng.choice(val_idx, size=int(p.max_val_windows), replace=False)

        # 2) Fit scaler on TRAIN normal points only
        X_train_pts = np.vstack([Xn[i] for i in train_idx]).astype(np.float64, copy=False)
        scaler = MinMaxScaler().fit(X_train_pts)
        Xtr = scaler.transform(X_train_pts)

        # 3) Build VAL points (normal-only)
        X_val_pts = np.vstack([Xn[i] for i in val_idx]).astype(np.float64, copy=False)
        Xv = scaler.transform(X_val_pts)

        # 4) MBA tune (unsupervised)
        nu_star, gamma_star, best_far = mba_tune_nu_gamma_unsupervised(
            X_train=Xtr,
            X_val=Xv,
            nu_bounds=p.nu_bounds,
            gamma_bounds=p.gamma_bounds,
            pop_size=p.pop_size,
            iters=p.iters,
            loudness_decay=p.loudness_decay,
            pulse_gamma=p.pulse_gamma,
            sv_penalty=p.sv_penalty,
            random_state=p.random_state,
            log_every_iters=max(1, p.iters // 5),
        )

        # 5) Fit final model on TRAIN+VAL points (still normal-only)
        X_fit_pts = np.vstack([Xn[i] for i in np.concatenate([train_idx, val_idx])]).astype(np.float64, copy=False)
        X_fit = scaler.transform(X_fit_pts)
        model = OneClassSVM(kernel="rbf", nu=float(nu_star), gamma=float(gamma_star)).fit(X_fit)

        # 6) FAR on NORMAL TEST windows (always defined)
        normal_test_scores: List[float] = []
        normal_test_preds: List[int] = []
        for i in test_idx:
            Xw_s = scaler.transform(Xn[i]) if Xn[i].size > 0 else Xn[i]
            normal_test_scores.append(_window_score(model, Xw_s))
            normal_test_preds.append(_window_pred_anom(model, Xw_s))  # 1 = flagged anomaly

        far_normal_test = float(np.mean(np.array(normal_test_preds, dtype=int) == 1)) if len(normal_test_preds) else float("nan")

        # 7) Evaluate ATTACK scenario df_test at window level
        df_test = _ensure_sorted_timestamp(df_test)
        Xa, aids, asubs = build_id_freq_windows(df_test, p.window_seconds, p.stride_seconds)

        n_scenario_windows_total = len(asubs)
        logger.info("mba_ocsvm_v2 paper_eval: scoring scenario windows (n=%d)", n_scenario_windows_total)
        scen_log_stride = max(1, n_scenario_windows_total // 10) if n_scenario_windows_total > 0 else 1

        attack_scores: List[float] = []
        attack_labels: List[int] = []
        attack_preds: List[int] = []

        for i, (Xw, sub) in enumerate(zip(Xa, asubs), start=1):
            lab, _ = get_uniform_window_label(sub)  # uniform window label (any attack -> 1)
            Xw_s = scaler.transform(Xw) if Xw.size > 0 else Xw
            sc = _window_score(model, Xw_s)
            pred = _window_pred_anom(model, Xw_s)  # 1 = flagged anomaly
            attack_scores.append(sc)
            attack_labels.append(int(lab))
            attack_preds.append(int(pred))
            if i == 1 or i == n_scenario_windows_total or (i % scen_log_stride == 0):
                logger.info("mba_ocsvm_v2 paper_eval scenario scoring: %d/%d", i, n_scenario_windows_total)

        n_attack_windows = int(np.sum(np.array(attack_labels) == 1))
        n_benign_windows = int(np.sum(np.array(attack_labels) == 0))
        total_windows = len(attack_labels)

        # "attack-window detection rate": fraction flagged among attack windows
        if n_attack_windows > 0:
            det_rate_attack_windows = float(np.mean((np.array(attack_preds)[np.array(attack_labels) == 1]) == 1))
        else:
            det_rate_attack_windows = float("nan")

        # "false alarm on benign windows within this scenario" (if any benign windows exist)
        if n_benign_windows > 0:
            far_in_scenario = float(np.mean((np.array(attack_preds)[np.array(attack_labels) == 0]) == 1))
        else:
            far_in_scenario = float("nan")

        # AUC on scenario windows if both classes exist
        auc_scenario = None
        if (n_attack_windows > 0) and (n_benign_windows > 0):
            try:
                auc_scenario = float(roc_auc_score(np.array(attack_labels, dtype=int), np.array(attack_scores, dtype=float)))
            except Exception:
                auc_scenario = None

        # 8) Optional constructed-mix AUC (normal test windows + attack scenario windows)
        # This is useful when the scenario has single-class windows (e.g., OTIDS DoS).
        constructed_auc = None
        constructed_note = None

        if auc_scenario is None:
            # build constructed labels/scores:
            # - normal_test windows are all label 0
            # - scenario windows keep their window labels (often all 1 for dense DoS)
            y_mix = np.concatenate([
                np.zeros(len(normal_test_scores), dtype=int),
                np.array(attack_labels, dtype=int),
            ])
            s_mix = np.concatenate([
                np.array(normal_test_scores, dtype=float),
                np.array(attack_scores, dtype=float),
            ])

            if np.unique(y_mix).size == 2:
                try:
                    constructed_auc = float(roc_auc_score(y_mix, s_mix))
                    constructed_note = "constructed_mix = normal_holdout_windows + scenario_windows (for ROC validity)"
                except Exception:
                    constructed_auc = None

        # Store fitted params back into object for debugging/use
        self.scaler = scaler
        self.model = model
        self.nu_, self.gamma_, self.mba_best_far_ = nu_star, gamma_star, best_far
        logger.info(
            "mba_ocsvm_v2 paper_eval: done (nu=%.6g, gamma=%.6g, far_normal_test=%.6f)",
            self.nu_,
            self.gamma_,
            far_normal_test,
        )

        return {
            "params": {
                "window_seconds": float(p.window_seconds),
                "stride_seconds": float(p.stride_seconds),
                "nu_bounds": list(map(float, p.nu_bounds)),
                "gamma_bounds": list(map(float, p.gamma_bounds)),
                "pop_size": int(p.pop_size),
                "iters": int(p.iters),
                "loudness_decay": float(p.loudness_decay),
                "pulse_gamma": float(p.pulse_gamma),
                "sv_penalty": float(p.sv_penalty),
                "random_state": int(p.random_state),
            },
            "tuned": {
                "nu": float(nu_star),
                "gamma": float(gamma_star),
                "mba_best_normal_val_far": float(best_far),
            },
            "normal_split_windows": {
                "n_total": int(len(Xn)),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
            },
            "normal_test": {
                "far_normal_test": float(far_normal_test),
                "n_windows": int(len(normal_test_scores)),
            },
            "scenario_test": {
                "n_windows": int(total_windows),
                "n_attack_windows": int(n_attack_windows),
                "n_benign_windows": int(n_benign_windows),
                "attack_window_detection_rate": float(det_rate_attack_windows),
                "far_on_benign_windows_within_scenario": float(far_in_scenario),
                "auc_scenario": auc_scenario,  # None if single-class windows
            },
            "constructed_mix": {
                "auc": constructed_auc,  # None if still not computable
                "note": constructed_note,
                "n_normal_holdout_windows": int(len(normal_test_scores)),
                "n_scenario_windows": int(total_windows),
            },
            "notes": [
                "Paper-faithful: MBA tuning and validation use ONLY normal data.",
                "Scenario ROC/AUC is reported only when both benign and attack windows exist.",
                "Constructed_mix AUC is optional and explicitly marked as constructed.",
            ],
        }

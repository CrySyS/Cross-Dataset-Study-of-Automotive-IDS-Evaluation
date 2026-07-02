# simple_ocsvm_ids.py (debuggable)
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple, List, Optional
import uuid
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
import logging
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import json

from unified_ids.dataio.windowing import Window, windows_fixed_time
from unified_ids.methods.base import BaseFlowIDS

logger = logging.getLogger(__name__)


# ---------------- Per-window features ----------------

def _maybe_to_int(x: str | int) -> int:
    if isinstance(x, str):
        s = x.strip()
        return int(s, 16) if s.lower().startswith("0x") else int(s)
    return int(x)

def id_freq_samples(df: pd.DataFrame, window_duration_sec: float) -> Tuple[np.ndarray, List[int]]:
    """
    Returns:
      X2d: shape (n_ids, 2) with columns [ID_decimal, frequency_hz]
      ids: list of the CAN IDs included in the window (ints)
    """
    if len(df) == 0:
        return np.zeros((0, 2), float), []
    cids = [_maybe_to_int(v) for v in df["can_id"].tolist()]
    unique, counts = np.unique(np.asarray(cids, int), return_counts=True)
    freq = counts.astype(float) / float(max(1e-6, window_duration_sec))
    X = np.column_stack([unique.astype(float), freq])
    return X, unique.tolist()


# ----------------- Simple OCSVM IDS (with DEBUG logs) ------------------

@dataclass
class OCSVMParams:
    window_seconds: float = 0.5
    stride_seconds: float = 0.5
    aggregate: str = "max"               # "max" or "mean"
    gamma_grid: Tuple[float, ...] = tuple(10.0**p for p in range(-4, 3))  # 1e-4..1e2
    nu_grid: Optional[Tuple[float, ...]] = (0.1,)  # fix nu to 0.1 by default
    target_fpr: float = 0.01
    random_state: int = 0

class SimpleOCSVMIDS(BaseFlowIDS):
    """
    Simple One-Class SVM for CAN intrusion detection using frequency-based features.
    
    **Method Overview:**
    - Extracts per-CAN-ID frequency features from fixed-size time windows
    - Trains One-Class SVM on benign traffic only
    - Detects anomalies by comparing window scores against a threshold
    
    **Features:**
    - Per-window per-CAN-ID: [ID_decimal, frequency_hz] (2 features)
    - Scaled using MinMaxScaler fitted on training data
    
    **Window Labeling:**
    - Window labeled as attack (1) if ANY message in the window is attack
    - Uses precomputed `w.label_window` from `windows_fixed_time` utility
    
    **Scoring:**
    - Per-ID scores: `-model.decision_function(X)` (negated, so positive=anomaly)
    - Window score: max (or mean) of per-ID scores in that window
    - Threshold: chosen on training set at target FPR (default 1%)
    
    **Hyperparameters:**
    - gamma: RBF kernel bandwidth (default: grid search over 1e-4 to 1e2)
    - nu: OCSVM outlier fraction (default: 0.1)
    - Uses grid search with validation set (80/20 split of benign windows)
    
    **Note:** This is the baseline implementation without MBA (Modified Bat Algorithm)
    hyperparameter optimization. For MBA optimization, see SimpleOCSVMIDSWithMBA.
    """

    def __init__(self, params: OCSVMParams = OCSVMParams()):
        self.p = params
        self.scaler: Optional[MinMaxScaler] = None
        self.model:  Optional[OneClassSVM]  = None
        self._val_windows: List[Window] = []
        self.threshold_: float = 0.0
        self.best_params_: dict = {}

    # ---------- small helpers ----------
    @staticmethod
    def _arr_stats(name: str, a: np.ndarray) -> str:
        if a.size == 0:
            return f"{name}: empty"
        a1 = a.ravel().astype(float)
        q = np.quantile(a1, [0.05, 0.5, 0.95])
        return (f"{name}: shape={a.shape} min={a1.min():.6g} p05={q[0]:.6g} "
                f"med={q[1]:.6g} p95={q[2]:.6g} max={a1.max():.6g}")

    def _stack_samples(self, wins: List[Tuple[np.ndarray, Window]]) -> np.ndarray:
        parts = [x for (x, w) in wins if x.size]
        return np.vstack(parts) if parts else np.zeros((0, 2), float)

    def _aggregate(self, per_id_scores: np.ndarray, counts_per_window: List[int]) -> np.ndarray:
        if per_id_scores.size == 0:
            return np.zeros((0,), float)
        agg = []
        k = 0
        for n in counts_per_window:
            s = per_id_scores[k:k+n]
            agg.append(float(np.max(s)) if self.p.aggregate == "max" else float(np.mean(s)))
            k += n
        return np.asarray(agg, float)

    # ---------- API ----------
    def fit(self, df_train: pd.DataFrame) -> "SimpleOCSVMIDS":
        p = self.p
        logger.debug("[Cfg] window=%.3fs stride=%.3fs agg=%s target_fpr=%.3f",
                    p.window_seconds, p.stride_seconds, p.aggregate, p.target_fpr)

        # Windowing
        all_w = list(windows_fixed_time(df_train, p.window_seconds, p.stride_seconds))
        n_all = len(all_w)
        n_pos = sum(w.label_window == 1 for w in all_w)
        logger.debug("[Split] total_windows=%d benign=%d attack=%d",
                    n_all, n_all - n_pos, n_pos)

        # Benign-only split
        benign_w = [w for w in all_w if w.label_window == 0]
        if len(benign_w) < 3:
            raise ValueError("Need at least 3 benign windows for train/val.")
        rng = np.random.default_rng(p.random_state)
        order = rng.permutation(len(benign_w)).tolist()
        benign_w = [benign_w[i] for i in order]
        cut = max(1, int(round(0.8 * len(benign_w))))
        train_w, val_w = benign_w[:cut], benign_w[cut:]
        self._val_windows = val_w
        logger.debug("[Split] train_w=%d val_w=%d (benign only)", len(train_w), len(val_w))

        # Train features (per-ID rows per window)
        train_pairs: List[Tuple[np.ndarray, Window]] = []
        for idx, w in enumerate(train_w[:3]):  # log first 3 windows
            sub = df_train.iloc[w.idx_start:w.idx_end+1]
            X2d, _ = id_freq_samples(sub, window_duration_sec=p.window_seconds)
            logger.debug("[Feat][train sample] wid=%s ids=%d %s",
                        w.window_id, X2d.shape[0],
                        self._arr_stats("freqHz", X2d[:,1] if X2d.size else np.array([])))
            train_pairs.append((X2d, w))
        for w in train_w[3:]:
            sub = df_train.iloc[w.idx_start:w.idx_end+1]
            X2d, _ = id_freq_samples(sub, window_duration_sec=p.window_seconds)
            train_pairs.append((X2d, w))

        X_train = self._stack_samples(train_pairs)
        if X_train.shape[0] == 0:
            raise ValueError("No per-ID samples in training windows.")
        logger.debug("[Train] %s", self._arr_stats("ID(col0)", X_train[:,0]))
        logger.debug("[Train] %s", self._arr_stats("Hz(col1)", X_train[:,1]))

        # Scale on TRAIN per-ID samples
        self.scaler = MinMaxScaler().fit(X_train)
        Xtr = self.scaler.transform(X_train)
        logger.debug("[Scale] data_min=%s data_max=%s", self.scaler.data_min_, self.scaler.data_max_)
        logger.debug("[Scale] %s", self._arr_stats("Xtr", Xtr))

        # Validation features
        val_pairs: List[Tuple[np.ndarray, Window]] = []
        val_counts: List[int] = []
        for idx, w in enumerate(val_w[:3]):  # log first 3 val windows
            sub = df_train.iloc[w.idx_start:w.idx_end+1]
            X2d, _ = id_freq_samples(sub, window_duration_sec=p.window_seconds)
            val_pairs.append((X2d, w))
            val_counts.append(X2d.shape[0])
            logger.debug("[Feat][val sample] wid=%s ids=%d %s",
                        w.window_id, X2d.shape[0],
                        self._arr_stats("freqHz", X2d[:,1] if X2d.size else np.array([])))
        for w in val_w[3:]:
            sub = df_train.iloc[w.idx_start:w.idx_end+1]
            X2d, _ = id_freq_samples(sub, window_duration_sec=p.window_seconds)
            val_pairs.append((X2d, w))
            val_counts.append(X2d.shape[0])

        Xv_raw = self._stack_samples(val_pairs)
        Xv = self.scaler.transform(Xv_raw) if Xv_raw.size else np.zeros((0, 2))
        logger.debug("[Val] perID=%d windows=%d; %s",
                    Xv.shape[0], len(val_w), self._arr_stats("Xv", Xv))

        # ==== NEW: window-level aggregation on TRAIN for threshold ====
        # counts per TRAIN window: number of per-ID rows in each window
        train_counts: List[int] = [x.shape[0] for (x, _) in train_pairs]
        # =============================================================

        # Grid search with WINDOW-LEVEL threshold (match test-time stat)
        best = (np.inf, None, None)  # (FPR, nu, gamma)
        nu_list = p.nu_grid if (p.nu_grid is not None and len(p.nu_grid) > 0) else (0.1,)
        logger.debug("[Grid] nu_list=%s gamma_grid=%s", nu_list, p.gamma_grid)

        for nu in nu_list:
            for gamma in p.gamma_grid:
                mdl = OneClassSVM(kernel="rbf", nu=float(nu), gamma=float(gamma)).fit(Xtr)

                # Train per-ID → per-window scores, then train-window quantile threshold
                s_tr_per_id = -mdl.decision_function(Xtr).ravel()
                agg_tr = self._aggregate(s_tr_per_id, train_counts)           # window scores on TRAIN
                thr = float(np.quantile(agg_tr, 1.0 - p.target_fpr))          # window-level threshold

                # Validation FPR at the same window-level threshold
                if Xv.size == 0:
                    fpr = float(np.mean(agg_tr >= thr))                        # fallback proxy
                else:
                    s_val_per_id = -mdl.decision_function(Xv).ravel()
                    agg_val = self._aggregate(s_val_per_id, val_counts) if val_counts else s_val_per_id
                    fpr = float(np.mean(agg_val >= thr))

                logger.debug("[Grid] nu=%.4f gamma=%g  thr(win)=%.6g  fpr_val=%.4f",
                            float(nu), float(gamma), thr, fpr)

                if fpr < best[0]:
                    best = (fpr, nu, gamma)

        _, nu_best, gamma_best = best
        logger.debug("[Best] nu=%.4f gamma=%g  fpr_val=%.4f",
                    float(nu_best), float(gamma_best), float(best[0]))

        # Final fit & WINDOW-level threshold from TRAIN
        self.model = OneClassSVM(kernel="rbf", nu=float(nu_best), gamma=float(gamma_best)).fit(Xtr)
        s_tr_per_id = -self.model.decision_function(Xtr).ravel()
        agg_tr = self._aggregate(s_tr_per_id, train_counts)
        self.threshold_ = float(np.quantile(agg_tr, 1.0 - p.target_fpr))
        self.best_params_ = {"nu": float(nu_best), "gamma": float(gamma_best), "fpr_val": float(best[0])}
        logger.debug("[Thr] window-level threshold@%.3f FPR (train windows) = %.6g",
                     p.target_fpr, self.threshold_)
        return self

    def score_windows(self, df_test: pd.DataFrame) -> Iterator[Tuple[Window, float]]:
        """
        Yields: (Window, window_score). Ground-truth label is in Window.label_window.
        """
        assert self.model is not None and self.scaler is not None, "Call fit() first."
        debug_first = 3
        k = 0
        total = 0
        alarms = 0
        p = self.p
        dbg_count = 0

        for w in windows_fixed_time(df_test, p.window_seconds, p.stride_seconds):
            sub = df_test.iloc[w.idx_start:w.idx_end+1]
            X2d, _ = id_freq_samples(sub, window_duration_sec=p.window_seconds)
            if X2d.size == 0:
                yield (w, 0.0)
                continue
            X = self.scaler.transform(X2d)
            per_id_scores = -self.model.decision_function(X).ravel()
            win_score = float(np.max(per_id_scores)) if p.aggregate == "max" else float(np.mean(per_id_scores))
            if dbg_count < 3 and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[TestWin] wid=%s label=%d perID=%d %s -> win_score=%.6g thr=%.6g alarm=%d",
                             w.window_id, w.label_window, X.shape[0],
                             self._arr_stats("scores", per_id_scores),
                             win_score, self.threshold_, int(win_score >= self.threshold_))
                dbg_count += 1
            yield (w, win_score)

        if total > 0:
            rate = alarms / total
        else:
            rate = 0.0
        logger.debug("[TestRpt] windows=%d alarms=%d rate=%.4f thr=%.6g",
                    total, alarms, rate, self.threshold_)

    def score_messages(self, df_test: pd.DataFrame) -> Iterator[Tuple[str, float, int]]:
        """
        Score each message individually by assigning it the anomaly score of its CAN ID within its window.
        
        Yields: (message_id, message_score, message_label)
        
        Strategy:
        - For each time window, compute per-CAN-ID anomaly scores
        - Each message gets the score of its CAN ID in that window
        - Messages in windows without their CAN ID get score 0.0
        """
        assert self.model is not None and self.scaler is not None, "Call fit() first."
        p = self.p
        
        for w in windows_fixed_time(df_test, p.window_seconds, p.stride_seconds):
            sub = df_test.iloc[w.idx_start:w.idx_end+1]
            if len(sub) == 0:
                continue
                
            X2d, ids = id_freq_samples(sub, window_duration_sec=p.window_seconds)
            
            if X2d.size == 0:
                # No CAN IDs in window - assign all messages score 0.0
                for idx in sub.index:
                    msg_label = int(sub.loc[idx, 'label']) if 'label' in sub.columns else 0
                    yield (str(idx), 0.0, msg_label)
                continue
            
            # Compute per-CAN-ID scores for this window
            X = self.scaler.transform(X2d)
            per_id_scores = -self.model.decision_function(X).ravel()
            
            # Create mapping: CAN_ID -> anomaly_score
            id_to_score = dict(zip(ids, per_id_scores))
            
            # Assign each message the score of its CAN ID
            for idx in sub.index:
                msg_can_id = int(_maybe_to_int(sub.loc[idx, 'can_id']))
                msg_score = float(id_to_score.get(msg_can_id, 0.0))
                msg_label = int(sub.loc[idx, 'label']) if 'label' in sub.columns else 0
                yield (str(idx), msg_score, msg_label)
        
    def validation_report(self) -> dict:
        """
        Quick sanity check on benign-only validation (held out during fit).
        """
        assert self.model is not None and self.scaler is not None, "Call fit() first."
        n_w = len(self._val_windows)
        if n_w == 0:
            logger.debug("[ValRpt] no validation windows stored")
            return {"n_val": 0}
        logger.debug("[ValRpt] n_val=%d thr=%.6g best=%s",
                     n_w, self.threshold_, self.best_params_)
        return {"n_val": int(n_w), "thr": float(self.threshold_), **self.best_params_}

    def paper_eval(self,
                df_test: pd.DataFrame,
                target_fprs: Tuple[float, ...] = (0.001, 0.01, 0.05)) -> dict:
        """
        Paper-style window-level evaluation.
        - Builds window scores on df_test (same aggregation & scaler/model as used in training).
        - Computes ROC AUC (if both classes exist).
        - Computes TPR at requested FPRs.
        - Reports confusion at the *train-derived* window-level threshold (self.threshold_).
        - Computes per-attack-type metrics if attack_type column exists.

        Returns a dict with metrics; also logs a one-line summary.
        """
        assert self.model is not None and self.scaler is not None, "Call fit() first."

        # --- collect window scores/labels/attack_types ---
        scores, labels, attack_types = [], [], []
        total = alarms = 0
        has_attack_type = 'attack_type' in df_test.columns
        
        for w in windows_fixed_time(df_test, self.p.window_seconds, self.p.stride_seconds):
            sub = df_test.iloc[w.idx_start:w.idx_end+1]
            X2d, _ = id_freq_samples(sub, window_duration_sec=self.p.window_seconds)
            if X2d.size == 0:
                continue
            X = self.scaler.transform(X2d)
            per_id_scores = -self.model.decision_function(X).ravel()
            win_score = float(np.max(per_id_scores)) if self.p.aggregate == "max" else float(np.mean(per_id_scores))
            scores.append(win_score)
            labels.append(int(w.label_window))
            if has_attack_type:
                # Get most common attack_type in window
                attack_type = sub['attack_type'].mode()[0] if len(sub) > 0 else 'Unknown'
                attack_types.append(str(attack_type))
            total += 1
            alarms += int(win_score >= self.threshold_)

        if total == 0:
            logger.debug("[PaperEval] no test windows")
            return {"n_test": 0}

        scores = np.asarray(scores, float)
        labels = np.asarray(labels, int)
        attack_types = np.asarray(attack_types) if attack_types else None

        # --- ROC & TPR@FPR ---
        metrics = {"n_test": int(total), "alarm_rate_at_train_thr": float(alarms/total), "train_window_thr": float(self.threshold_)}

        has_pos = np.any(labels == 1)
        has_neg = np.any(labels == 0)
        if has_pos and has_neg:
            try:
                auc = roc_auc_score(labels, scores)
                fpr, tpr, thr = roc_curve(labels, scores)  # increasing thresholds
                metrics["roc_auc"] = float(auc)

                # interpolate TPR at requested FPRs
                for f in target_fprs:
                    # clip in case f not covered numerically
                    ff = float(np.clip(f, fpr.min(), fpr.max()))
                    t = float(np.interp(ff, fpr, tpr))
                    metrics[f"tpr_at_fpr_{f:.3%}"] = t
            except Exception as e:
                logger.debug("[PaperEval] ROC computation failed: %s", e)

        # --- confusion at the train-derived window-level threshold ---
        yhat = (scores >= self.threshold_).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(labels, yhat, labels=[0,1]).ravel()
            metrics.update({
                "CM_train_thr": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                "fpr_at_train_thr": float(fp / max(tn + fp, 1)),
                "tpr_at_train_thr": float(tp / max(tp + fn, 1)),
                "precision_at_train_thr": float(tp / max(tp + fp, 1)),
                "recall_at_train_thr": float(tp / max(tp + fn, 1)),  # == tpr
            })
        except Exception:
            # in case only one class present
            metrics["CM_train_thr"] = None

        # --- per-attack-type metrics ---
        if has_attack_type and attack_types is not None:
            metrics["per_attack_type"] = {}
            unique_types = np.unique(attack_types)
            
            for atype in unique_types:
                mask = attack_types == atype
                type_scores = scores[mask]
                type_labels = labels[mask]
                type_yhat = (type_scores >= self.threshold_).astype(int)
                
                n_windows = int(mask.sum())
                n_attack = int(type_labels.sum())
                n_benign = n_windows - n_attack
                
                type_metrics = {
                    "n_windows": n_windows,
                    "n_attack": n_attack,
                    "n_benign": n_benign
                }
                
                # Compute metrics if we have both classes
                if n_attack > 0 and n_benign > 0:
                    try:
                        type_auc = roc_auc_score(type_labels, type_scores)
                        type_metrics["roc_auc"] = float(type_auc)
                    except:
                        pass
                    
                    # Confusion matrix
                    try:
                        tn, fp, fn, tp = confusion_matrix(type_labels, type_yhat, labels=[0,1]).ravel()
                        type_metrics.update({
                            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                            "tpr": float(tp / max(tp + fn, 1)),
                            "fpr": float(fp / max(tn + fp, 1)),
                            "precision": float(tp / max(tp + fp, 1))
                        })
                    except:
                        pass
                elif n_attack > 0:
                    # Only attack windows
                    tp = int(type_yhat.sum())
                    fn = n_attack - tp
                    type_metrics.update({
                        "tp": tp, "fn": fn,
                        "tpr": float(tp / n_attack) if n_attack > 0 else 0.0
                    })
                elif n_benign > 0:
                    # Only benign windows
                    fp = int(type_yhat.sum())
                    tn = n_benign - fp
                    type_metrics.update({
                        "tn": tn, "fp": fp,
                        "fpr": float(fp / n_benign) if n_benign > 0 else 0.0
                    })
                
                metrics["per_attack_type"][atype] = type_metrics

        # --- log a compact summary ---
        parts = [f"AUC={metrics.get('roc_auc', 'NA')}",
                f"TPR@1%={metrics.get('tpr_at_fpr_1.000%', 'NA')}",
                f"FPR@thr={metrics.get('fpr_at_train_thr','NA'):.4f}" if "fpr_at_train_thr" in metrics else "FPR@thr=NA",
                f"TPR@thr={metrics.get('tpr_at_train_thr','NA'):.4f}" if "tpr_at_train_thr" in metrics else "TPR@thr=NA",
                f"alarm_rate={metrics['alarm_rate_at_train_thr']:.4f}"]
        logger.debug("[PaperEval] n=%d thr=%.6g | %s", total, self.threshold_, " ".join(parts))
        
        # Log per-attack-type summary
        if "per_attack_type" in metrics:
            logger.info("[PaperEval] Per-attack-type metrics:")
            for atype, m in metrics["per_attack_type"].items():
                auc_str = f"AUC={m.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in m else "AUC=N/A"
                tpr_str = f"TPR={m.get('tpr', 'N/A'):.4f}" if 'tpr' in m else "TPR=N/A"
                fpr_str = f"FPR={m.get('fpr', 'N/A'):.4f}" if 'fpr' in m else "FPR=N/A"
                logger.info(f"  {atype}: n={m['n_windows']} (attack={m['n_attack']}, benign={m['n_benign']}) | {auc_str} {tpr_str} {fpr_str}")

        return metrics

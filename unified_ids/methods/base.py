from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional, Dict, Any
import pandas as pd
from dataclasses import asdict, is_dataclass

from unified_ids.dataio.windowing import Window


class BaseFlowIDS(ABC):
    @abstractmethod
    def fit(self, df_train: pd.DataFrame):
        """
        Train the IDS on the provided training data.
        Implementations may assume df_train is pre-filtered / split as needed.
        """
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the configuration/hyperparameters used by this IDS instance.
        
        Default implementation looks for common parameter attributes:
        - self.params (dataclass or dict)
        - self.config (dataclass or dict)
        - self.p (short form for params)
        - self.cfg (short form for config)
        - Other attributes can be returned by overriding this method
        
        Returns a dict suitable for JSON serialization.
        """
        params = {}
        
        # Check for common parameter attributes (in order of preference)
        param_attrs = [
            ('params', 'params'),
            ('config', 'config'),
            ('p', 'params'),      # common shorthand
            ('cfg', 'config'),    # common shorthand
        ]
        
        for attr_name, result_key in param_attrs:
            if hasattr(self, attr_name):
                obj = getattr(self, attr_name)
                if obj is None:
                    continue
                    
                if is_dataclass(obj):
                    params[result_key] = asdict(obj)
                elif isinstance(obj, dict):
                    params[result_key] = obj
                else:
                    # Try to convert to dict
                    try:
                        params[result_key] = vars(obj) if hasattr(obj, '__dict__') else str(obj)
                    except:
                        params[result_key] = str(obj)
                break  # Use first found
        
        # Add any other common configuration attributes
        for attr in ['window_seconds', 'stride_seconds', 'window_size', 'stride_size', 
                     'signal_mask_path', 'max_ids']:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    params[attr] = val
        
        return params

    def score_messages(
        self, df_test: pd.DataFrame
    ) -> Iterator[Tuple[str, float, int]]:
        """
        Optional: yield (message_id, score, label_message).

        This is the preferred interface for the UNIFORM evaluation:
          - message_id: any stable identifier (e.g. original row index as str)
          - score: higher = more anomalous
          - label_message: 0 (benign) or 1 (attack)

        Default implementation signals that message-level scoring is not
        supported for this IDS.
        """
        raise NotImplementedError("score_messages is not implemented for this IDS")

    def score_windows(
            self, df_test: pd.DataFrame
    ) -> Iterator[Tuple[Window, float]]:
            """
            Optional: yield (window, score).

            Window objects come from the shared windowing module and already carry
            the ground-truth label (label_window), index span (idx_start/idx_end),
            timestamps, and frac_attack. Implementations must **not** change the
            label; they only return the anomaly score for the provided window.

            Used by:
                - methods that are inherently window-based (e.g. OCSVM with
                    window features), and/or
                - reproducing paper-specific window-based evaluation protocols.

            Default implementation signals that window-level scoring is not
            supported for this IDS.
            """
            raise NotImplementedError("score_windows is not implemented for this IDS")

    def paper_eval(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        out_dir: Optional[str] = None,
    ):
        """
        Optional: method-specific 'paper-style' evaluation.

        Implement this to reproduce the original paper's evaluation for
        this IDS (tables, figures, special thresholds, etc.).

        Should return a dict (JSON-serializable) that will be stored under
        `paper_eval` in metrics.json. Implementations may also write extra
        artifacts into out_dir if provided.

        Default implementation signals that no paper-style eval is available.
        """
        raise NotImplementedError("paper_eval is not implemented for this IDS")

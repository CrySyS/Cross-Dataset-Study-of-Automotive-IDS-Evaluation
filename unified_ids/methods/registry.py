
"""
Method registry for Unified IDS.

This decouples method construction from the CLI / evaluation core.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any


from unified_ids.methods.simple_ocsvm import SimpleOCSVMIDS, OCSVMParams
from unified_ids.methods.mba_ocsvm_version2 import AvatefipourOCSVM_MBA, AvatefipourParams
from unified_ids.methods.daga_ngram_STABILI_2022 import DagaNGram, DagaParams
from unified_ids.methods.assoc_rules_D_ANGELO_2023 import AssocRulesIDS
@dataclass
class MethodFactory:
    """
    Small wrapper that holds a constructor plus default kwargs.
    """

    ctor: Callable[..., Any]
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, **overrides: Any) -> Any:
        kwargs = {**self.default_kwargs, **overrides}
        return self.ctor(**kwargs)


# ----------------------------------------------------------------------
# Registry of methods
#   - keys: method names as used by CLI and EVAL_PLAN
#   - values: MethodFactory instances that build the IDS model
# ----------------------------------------------------------------------

METHOD_REGISTRY: Dict[str, MethodFactory] = {

    "simple_ocsvm": MethodFactory(
        SimpleOCSVMIDS,
        default_kwargs=dict(
            params=OCSVMParams(
                window_seconds=0.5,
                stride_seconds=0.5,
            )
        ),
    ),
    "mba_ocsvm_v2": MethodFactory(
        AvatefipourOCSVM_MBA,
        default_kwargs=dict(
            params=AvatefipourParams(
                window_seconds=1,
                stride_seconds=1,  # recommended to be the same as window_seconds
                # Fast-fair preset: materially reduces tuning time while
                # preserving the same modeling pipeline and window protocol.
                pop_size=10,
                iters=20,
                loudness_decay=0.2,
                pulse_gamma=0.2,
                nu_bounds=(0.001, 0.2),
                gamma_bounds=(1e-6, 1e2),
                sv_penalty=0.0,
                random_state=0,
                max_train_windows=800,   # optional speed cap (set None for full)
                max_val_windows=250,
            )
        ),
    ),


    "daga_ngram": MethodFactory(
        DagaNGram,
        default_kwargs=dict(
            params=DagaParams(
                n=6,  # n-gram length (paper default)
                id_remap=True,  # compact ID mapping
                window_seconds=0.5,  # for window aggregation
                stride_seconds=0.5,
                window_score="any",  # "any" will be binary or "frac" will be nonbinary output NOTE: method caps will need to change binary is true
            )
        ),
    ),

    

    "assoc_rules": MethodFactory(
        AssocRulesIDS,
        default_kwargs=dict(
            k_cla=300,  # CLA clusters per ID (paper: K=300)
            n_knn=50,  # region clusters across IDs (paper: N=50)
            n_centroid_clusters=150,  # quantization: centroid clusters
            n_bound_bins=100,  # quantization: bound bins
            window_size=900,  # messages per window (paper: ~0.5s)
            stride=450,  # window stride
            min_support=0.9,  # Apriori support threshold
            dist_threshold=0.3,  # distribution change threshold
            random_state=0,
        ),
    ),
}

# tools/road_sanity.py
from pathlib import Path
import logging

import pandas as pd

from adapters import road  # adjust import to your project layout


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def sanity_check_road_logs(root: str | Path, max_files: int = 10) -> None:
    root = Path(root)
    logs = sorted(root.rglob("*.log"))
    if not logs:
        print(f"No .log files under {root}")
        return

    for i, p in enumerate(logs):
        if i >= max_files:
            break

        print("=" * 80)
        print(f"FILE: {p}")
        df_raw = road.load_raw(p)
        df_canon = road.to_canonical(df_raw, src_path=p)

        n_total = len(df_canon)
        n_attack = int(df_canon["label"].sum())
        unique_ids = df_canon["can_id"].nunique()
        attack_ids = df_canon.loc[df_canon["label"] == 1, "can_id"].value_counts()

        print(f"  total frames      : {n_total}")
        print(f"  unique IDs        : {unique_ids}")
        print(f"  attack frames     : {n_attack}")
        print(f"  attack_type       : {df_canon['attack_type'].iloc[0] if n_total else None}")

        if n_attack > 0:
            t_min = df_canon.loc[df_canon["label"] == 1, "timestamp"].min()
            t_max = df_canon.loc[df_canon["label"] == 1, "timestamp"].max()
            print(f"  attack time range : [{t_min:.3f}, {t_max:.3f}] s (elapsed)")
            print("  top attack IDs:")
            for cid, cnt in attack_ids.head(5).items():
                print(f"    ID {cid:>5}: {cnt} frames")
        else:
            print("  (no attack frames labeled)")

        # Quick check filler frames
        if "is_filler" in df_canon.columns:
            n_filler = int(df_canon["is_filler"].sum())
            if n_filler > 0:
                print(f"  filler frames (ID 0xFFF): {n_filler}")

        print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m tools.road_sanity <root_folder> [max_files]")
        raise SystemExit(1)
    root = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sanity_check_road_logs(root, max_files=max_files)

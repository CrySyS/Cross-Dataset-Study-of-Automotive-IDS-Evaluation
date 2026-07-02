#!/usr/bin/env python3
"""
Derive CANShield feature grouping and reordered feature list from Pearson correlation.

This script is standalone and intended to reproduce the paper-style initialization step:
1) compute signal correlation matrix,
2) perform hierarchical clustering,
3) output a reordered feature list that keeps correlated signals close.

It does NOT modify config files automatically. It writes artifacts and a ready-to-paste
YAML block so you can update config files manually (safer workflow).

Usage examples (from CANShield-main/src):

 .venv/bin/python derive_feature_order_from_correlation.py \
      --config syncan

  .venv/bin/python derive_feature_order_from_correlation.py \
      --config road --max-files 20 --max-rows-per-file 50000 --cluster-distance-threshold 0.35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from dataset.load_dataset import generate_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_config(config_name: str):
    cfg_path = Path(__file__).resolve().parent / ".." / "config" / f"{config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file is not a mapping: {cfg_path}")
    return cfg, cfg_path


def _list_csv_files(data_dir: Path, max_files: int) -> List[Path]:
    files = sorted(data_dir.glob("*.csv"))
    if max_files > 0:
        files = files[:max_files]
    return files


def _sample_rows(df: pd.DataFrame, max_rows_per_file: int) -> pd.DataFrame:
    if max_rows_per_file <= 0 or len(df) <= max_rows_per_file:
        return df

    # Deterministic sub-sampling by stride keeps ordering stable and avoids random drift.
    stride = max(1, len(df) // max_rows_per_file)
    sampled = df.iloc[::stride]
    if len(sampled) > max_rows_per_file:
        sampled = sampled.iloc[:max_rows_per_file]
    return sampled


def _load_feature_frame(
    csv_files: List[Path],
    features: List[str],
    org_columns: List[str],
    max_rows_per_file: int,
    strict_missing: bool,
) -> pd.DataFrame:
    chunks = []
    feature_set = set(features)

    for csv_path in csv_files:
        try:
            df0 = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False)
        except Exception as exc:
            print(f"[WARN] {csv_path.name}: failed direct read ({exc}), trying raw->generated conversion")
            try:
                df0 = generate_dataset(csv_path.stem, csv_path, org_columns)
            except Exception as conv_exc:
                print(f"[WARN] {csv_path.name}: conversion failed ({conv_exc}), skipping")
                continue

        present = [c for c in features if c in df0.columns]
        missing = [c for c in features if c not in df0.columns]

        # Raw files may not yet contain expanded Sig_*_of_ID_* columns.
        if not present:
            try:
                df0 = generate_dataset(csv_path.stem, csv_path, org_columns)
                present = [c for c in features if c in df0.columns]
                missing = [c for c in features if c not in df0.columns]
            except Exception as conv_exc:
                print(f"[WARN] {csv_path.name}: conversion to generated format failed ({conv_exc})")

        if missing:
            msg = f"{csv_path.name}: missing {len(missing)} feature columns"
            if strict_missing:
                raise ValueError(msg)
            print(f"[WARN] {msg}")

        if not present:
            print(f"[WARN] {csv_path.name}: no target features present, skipping")
            continue

        df = df0[present].copy()
        for col in present:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.ffill().bfill().dropna(axis=0, how="any")
        df = _sample_rows(df, max_rows_per_file)
        if len(df) == 0:
            print(f"[WARN] {csv_path.name}: empty after cleaning, skipping")
            continue

        chunks.append(df)

    if not chunks:
        raise ValueError("No usable data rows found across CSV files.")

    merged = pd.concat(chunks, axis=0, ignore_index=True)
    present_global = [c for c in features if c in merged.columns]
    missing_global = sorted(feature_set - set(present_global))
    if missing_global:
        raise ValueError(
            "Some config features were never found in loaded data: "
            f"{missing_global}"
        )

    merged = merged[features].copy()
    return merged


def _stable_corr(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(method="pearson")
    corr = corr.replace([np.inf, -np.inf], np.nan)

    # Constant columns can produce NaN correlations; keep self-corr at 1.0, others at 0.0.
    corr = corr.fillna(0.0).copy()
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    corr.iloc[:, :] = values
    return corr


def _cluster_from_corr(corr: pd.DataFrame, linkage_method: str, use_abs_corr: bool):
    base = corr.to_numpy(copy=True)
    if use_abs_corr:
        base = np.abs(base)

    distance = 1.0 - base
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 2.0)

    condensed = squareform(distance, checks=False)
    z = linkage(condensed, method=linkage_method)
    order_idx = leaves_list(z)
    ordered_features = [corr.columns[i] for i in order_idx]
    return z, ordered_features, distance


def _features_by_cluster(
    z,
    features: List[str],
    cluster_distance_threshold: float,
) -> Dict[str, List[str]]:
    labels = fcluster(z, t=cluster_distance_threshold, criterion="distance")
    pairs = list(zip(features, labels))

    # Keep output deterministic and easy to read.
    clusters: Dict[str, List[str]] = {}
    for feat, label in sorted(pairs, key=lambda x: (int(x[1]), x[0])):
        key = f"cluster_{int(label):03d}"
        clusters.setdefault(key, []).append(feat)
    return clusters


def _plot_heatmap(corr: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.to_numpy(), aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Signals")
    ax.set_ylabel("Signals")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_dendrogram(z, labels: List[str], out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(z, labels=labels, leaf_rotation=90, leaf_font_size=6, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Distance (1 - |r| if abs corr enabled)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _yaml_list_block(key: str, values: List[str]) -> str:
    lines = [f"{key} : ["]
    for i, v in enumerate(values):
        suffix = "," if i < len(values) - 1 else ""
        lines.append(f"        '{v}'{suffix}")
    lines.append("]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Derive CANShield feature order via Pearson correlation clustering.")
    parser.add_argument("--config", required=True, choices=["syncan", "road", "crysys"], help="Config name from ../config.")
    parser.add_argument("--data-dir", default=None, help="Optional override for input CSV directory.")
    parser.add_argument("--use-test-data", action="store_true", help="Use test_data_dir instead of train_data_dir.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of CSV files (0 = all).")
    parser.add_argument("--max-rows-per-file", type=int, default=100000, help="Max rows loaded per CSV after cleaning (0 = all).")
    parser.add_argument("--linkage-method", default="average", choices=["single", "complete", "average", "weighted", "ward"], help="Hierarchical linkage method.")
    parser.add_argument("--no-abs-corr", action="store_true", help="Use signed correlation directly; default uses absolute correlation.")
    parser.add_argument("--cluster-distance-threshold", type=float, default=0.30, help="Distance cutoff for cluster assignment output.")
    parser.add_argument("--strict-missing", action="store_true", help="Fail if any feature is missing in a loaded file.")
    args = parser.parse_args()

    cfg, cfg_path = _load_config(args.config)
    dataset_name = str(cfg.get("dataset_name", args.config))
    features = list(cfg["features"])
    org_columns = list(cfg["org_columns"])

    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser()
    else:
        key = "test_data_dir" if args.use_test_data else "train_data_dir"
        cfg_data_dir = Path(str(cfg[key]))
        if cfg_data_dir.is_absolute():
            data_dir = cfg_data_dir
        else:
            data_dir = (cfg_path.parent / cfg_data_dir).resolve()

    data_dir = data_dir.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = _list_csv_files(data_dir, args.max_files)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"[INFO] dataset={dataset_name}")
    print(f"[INFO] files={len(csv_files)} data_dir={data_dir}")
    print(f"[INFO] features={len(features)}")

    df = _load_feature_frame(
        csv_files=csv_files,
        features=features,
        org_columns=org_columns,
        max_rows_per_file=int(args.max_rows_per_file),
        strict_missing=bool(args.strict_missing),
    )
    print(f"[INFO] merged rows={len(df)}")

    corr = _stable_corr(df)
    z, ordered_features, _ = _cluster_from_corr(
        corr=corr,
        linkage_method=str(args.linkage_method),
        use_abs_corr=not bool(args.no_abs_corr),
    )

    ordered_corr = corr.loc[ordered_features, ordered_features]
    clusters = _features_by_cluster(
        z,
        ordered_features,
        cluster_distance_threshold=float(args.cluster_distance_threshold),
    )

    out_dir = Path(__file__).resolve().parent / ".." / "artifacts" / "grouping_reorder" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    corr.to_csv(out_dir / "pearson_corr_original.csv")
    ordered_corr.to_csv(out_dir / "pearson_corr_reordered.csv")

    with open(out_dir / "recommended_feature_order.json", "w", encoding="utf-8") as fp:
        json.dump(ordered_features, fp, indent=2)

    with open(out_dir / "clusters_by_distance.json", "w", encoding="utf-8") as fp:
        json.dump(clusters, fp, indent=2)

    yaml_block = _yaml_list_block("features", ordered_features)
    with open(out_dir / "recommended_features_yaml_block.txt", "w", encoding="utf-8") as fp:
        fp.write(yaml_block + "\n")

    summary = {
        "dataset_name": dataset_name,
        "config": args.config,
        "data_dir": str(data_dir),
        "n_files": len(csv_files),
        "n_rows_merged": int(len(df)),
        "n_features": int(len(features)),
        "linkage_method": args.linkage_method,
        "use_absolute_correlation": not bool(args.no_abs_corr),
        "cluster_distance_threshold": float(args.cluster_distance_threshold),
        "outputs": {
            "corr_original_csv": str(out_dir / "pearson_corr_original.csv"),
            "corr_reordered_csv": str(out_dir / "pearson_corr_reordered.csv"),
            "recommended_order_json": str(out_dir / "recommended_feature_order.json"),
            "recommended_features_yaml_block": str(out_dir / "recommended_features_yaml_block.txt"),
            "clusters_json": str(out_dir / "clusters_by_distance.json"),
            "heatmap_original": str(out_dir / "heatmap_original.png"),
            "heatmap_reordered": str(out_dir / "heatmap_reordered.png"),
            "dendrogram": str(out_dir / "dendrogram.png"),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    _plot_heatmap(corr, out_dir / "heatmap_original.png", f"{dataset_name}: correlation (original config order)")
    _plot_heatmap(ordered_corr, out_dir / "heatmap_reordered.png", f"{dataset_name}: correlation (cluster reordered)")
    _plot_dendrogram(z, ordered_features, out_dir / "dendrogram.png", f"{dataset_name}: hierarchical clustering dendrogram")

    print(f"[DONE] Wrote artifacts to: {out_dir}")
    print(f"[DONE] Recommended ordered feature list: {out_dir / 'recommended_feature_order.json'}")
    print(f"[DONE] YAML block to paste into config: {out_dir / 'recommended_features_yaml_block.txt'}")


if __name__ == "__main__":
    main()

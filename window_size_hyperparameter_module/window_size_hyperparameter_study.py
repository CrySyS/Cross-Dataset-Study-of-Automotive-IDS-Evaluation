#!/usr/bin/env python3
"""
Generic Window Size Hyperparameter Study for IDS Methods

Usage:
    python window_size_hyperparameter_study.py \\
        --dataset_name "Car-Hacking" \\
        --train_glob "data_parquet/03_Car-HackingDataset/normal_run_data.parquet" \\
        --test_glob "data_parquet/03_Car-HackingDataset/*.parquet" \\
        --output_dir "results/window_study"

    python window_size_hyperparameter_study.py \\
        --dataset_name "OTIDS" \\
        --train_glob "data_parquet/04_CAN-IntrusionDataset_OTIDS/Attack_free_dataset.parquet" \\
        --test_glob "data_parquet/04_CAN-IntrusionDataset_OTIDS/*.parquet" \\
        --output_dir "results/window_study"
"""

import sys
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unified_ids.dataio.loaders import read_parquet_glob
from unified_ids.methods.mba_ocsvm_version2 import AvatefipourOCSVM_MBA, AvatefipourParams
from unified_ids.eval.metrics import roc_curve, roc_auc




def run_experiment(dataset_name, train_glob, test_glob, window_size):
    """Run single experiment with given window size"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} - Window Size: {window_size}s")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data...")
    df_train = read_parquet_glob(train_glob)
    df_test = read_parquet_glob(test_glob)
    print(f"  Train: {len(df_train):,} rows")
    print(f"  Test: {len(df_test):,} rows")
    
    # Create method with specific window size
    params = AvatefipourParams(
        window_seconds=window_size,
        stride_seconds=window_size,  # stride = window
        pop_size=25,
        iters=100,
        loudness_decay=0.2,
        pulse_gamma=0.2,
        nu_bounds=(0.001, 0.2),
        gamma_bounds=(1e-6, 1e2),
        sv_penalty=0.0,
        random_state=0,
        max_train_windows=2000,
        max_val_windows=None,
    )
    
    model = AvatefipourOCSVM_MBA(params=params)
    
    # Train
    print("Training...")
    start_time = time.time()
    model.fit(df_train)
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.1f}s")
    
    # Score
    print("Scoring...")
    start_time = time.time()
    results_iter = model.score_windows(df_test)
    window_results = list(results_iter)
    score_time = time.time() - start_time

    # Support both tuple formats:
    # - Current framework: (Window, score)
    # - Legacy scripts:   (window_id, score, label)
    if not window_results:
        print("  Warning: No windows were scored")
        return None

    if len(window_results[0]) == 2:
        y_scores = np.array([score for _, score in window_results], dtype=float)
        y_true = np.array([int(window.label_window) for window, _ in window_results], dtype=int)
    elif len(window_results[0]) == 3:
        y_true = np.array([label for _, _, label in window_results], dtype=int)
        y_scores = np.array([score for _, score, _ in window_results], dtype=float)
    else:
        raise ValueError(
            f"Unsupported score_windows output tuple length: {len(window_results[0])}"
        )
    
    print(f"  Scoring time: {score_time:.1f}s")
    print(f"  Test windows: {len(y_true)} (benign: {sum(y_true==0)}, attack: {sum(y_true==1)})")
    
    # Compute metrics
    if len(np.unique(y_true)) < 2:
        print("  ⚠️  Warning: Only one class in test set, skipping ROC")
        return None
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_roc_val = roc_auc(y_scores, y_true)
    
    # Youden index
    youden_idx = np.argmax(tpr - fpr)
    youden_threshold = thresholds[youden_idx]
    youden_fpr = fpr[youden_idx]
    youden_tpr = tpr[youden_idx]
    
    print(f"  AUC-ROC: {auc_roc_val:.4f}")
    print(f"  Youden: FPR={youden_fpr:.4f}, TPR={youden_tpr:.4f}, threshold={youden_threshold:.6f}")
    
    return {
        'dataset': dataset_name,
        'window_size': window_size,
        'auc_roc': auc_roc_val,
        'n_windows': len(y_true),
        'n_benign': int(sum(y_true==0)),
        'n_attack': int(sum(y_true==1)),
        'fpr_youden': youden_fpr,
        'tpr_youden': youden_tpr,
        'threshold_youden': youden_threshold,
        'train_time': train_time,
        'score_time': score_time,
        'avg_latency': window_size / 2.0  # Average detection latency
    }


def plot_results(df, dataset_name, output_dir):
    """Create analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Window Size Analysis - {dataset_name} (MBA OCSVM V2)', 
                 fontsize=16, fontweight='bold')
    
    # 1. AUC-ROC vs Window Size
    ax = axes[0, 0]
    ax.plot(df['window_size'], df['auc_roc'], 'o-', linewidth=2, markersize=10, color='#2E86AB')
    ax.set_xlabel('Window Size (seconds)', fontsize=11)
    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_title('Detection Performance vs Window Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    
    # 2. FPR and TPR vs Window Size
    ax = axes[0, 1]
    ax.plot(df['window_size'], df['fpr_youden'], 'o-', linewidth=2, markersize=10, 
            color='#E63946', label='FPR (Youden)')
    ax.plot(df['window_size'], df['tpr_youden'], 's-', linewidth=2, markersize=10, 
            color='#06A77D', label='TPR (Youden)')
    ax.set_xlabel('Window Size (seconds)', fontsize=11)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_title('False Alarm & Detection Rates vs Window Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.legend()
    
    # 3. Detection Latency vs Window Size
    ax = axes[1, 0]
    ax.plot(df['window_size'], df['avg_latency'], 'o-', linewidth=2, markersize=10, color='#8338EC')
    ax.set_xlabel('Window Size (seconds)', fontsize=11)
    ax.set_ylabel('Avg Detection Latency (seconds)', fontsize=11)
    ax.set_title('Detection Latency vs Window Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Runtime vs Window Size
    ax = axes[1, 1]
    ax.plot(df['window_size'], df['train_time'], 'o-', linewidth=2, markersize=10, 
            color='#FF6B35', label='Train Time')
    ax.plot(df['window_size'], df['score_time'], 's-', linewidth=2, markersize=10, 
            color='#F7B801', label='Score Time')
    ax.set_xlabel('Window Size (seconds)', fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Runtime vs Window Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Window Size Hyperparameter Study for MBA OCSVM V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--dataset_name', required=True, 
                        help='Name of dataset (for labeling)')
    parser.add_argument('--train_glob', required=True, 
                        help='Glob pattern for training data')
    parser.add_argument('--test_glob', required=True, 
                        help='Glob pattern for test data')
    parser.add_argument('--output_dir', default='results/window_study',
                        help='Output directory for results')
    parser.add_argument('--window_sizes', type=str, default='10,30,60,120,300,600',
                        help='Comma-separated window sizes (seconds, floats allowed)')
    
    args = parser.parse_args()
    
    # Parse window sizes
    window_sizes = [float(x.strip()) for x in args.window_sizes.split(',')]
    
    print("=" * 80)
    print("WINDOW SIZE HYPERPARAMETER STUDY - MBA OCSVM V2")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Train glob: {args.train_glob}")
    print(f"Test glob: {args.test_glob}")
    print(f"Window sizes: {window_sizes}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    all_results = []
    for window_size in window_sizes:
        try:
            result = run_experiment(
                args.dataset_name,
                args.train_glob,
                args.test_glob,
                window_size
            )
        except Exception as exc:
            print(f"  Warning: window size {window_size}s failed: {exc}")
            continue

        if result:
            all_results.append(result)
    
    if not all_results:
        print("\n❌ No results collected!")
        return 1
    
    # Create results dataframe
    df_results = pd.DataFrame(all_results)
    
    # Save results
    csv_path = output_dir / f'{args.dataset_name}_window_size_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to {csv_path}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    # Plot results
    fig = plot_results(df_results, args.dataset_name, output_dir)
    
    plot_path = output_dir / f'{args.dataset_name}_window_analysis.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {plot_path}")
    plt.close(fig)
    
    # Save summary JSON
    summary = {
        'dataset': args.dataset_name,
        'window_sizes': window_sizes,
        'n_experiments': len(all_results),
        'best_auc': {
            'value': float(df_results['auc_roc'].max()),
            'window_size': float(df_results.loc[df_results['auc_roc'].idxmax(), 'window_size'])
        },
        'best_tpr': {
            'value': float(df_results['tpr_youden'].max()),
            'window_size': float(df_results.loc[df_results['tpr_youden'].idxmax(), 'window_size'])
        },
        'lowest_latency': {
            'value': float(df_results['avg_latency'].min()),
            'window_size': float(df_results.loc[df_results['avg_latency'].idxmin(), 'window_size'])
        }
    }
    
    summary_path = output_dir / f'{args.dataset_name}_window_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {csv_path}")
    print(f"  - {plot_path}")
    print(f"  - {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

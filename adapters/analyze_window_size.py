"""
Analyze dataset to determine appropriate window size for benign windows.
"""
import pandas as pd
import numpy as np


def analyze_window_requirements(parquet_path):
    """
    Analyzes attack distribution to recommend window sizes that allow benign windows.
    
    Returns the maximum window size that could contain only benign messages,
    based on the gaps between attack messages.
    """
    df = pd.read_parquet(parquet_path)
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {parquet_path.split('/')[-1]}")
    print(f"{'='*70}")
    
    # Basic stats
    t_min = df['timestamp'].min()
    t_max = df['timestamp'].max()
    duration = t_max - t_min
    
    print(f"\nDataset duration: {duration:.2f} seconds")
    print(f"Total messages: {len(df):,}")
    print(f"Attack messages: {(df['label']==1).sum():,}")
    print(f"Benign messages: {(df['label']==0).sum():,}")
    
    # Get attack timestamps sorted
    attack_ts = np.sort(df.loc[df['label'] == 1, 'timestamp'].to_numpy())
    
    if len(attack_ts) == 0:
        print("\nNo attacks found!")
        return
    
    # Find all attack-free intervals
    # Include: [t_min -> first_attack], [attack_i -> attack_i+1], [last_attack -> t_max]
    attack_free_intervals = []
    
    # Gap before first attack
    if attack_ts[0] > t_min:
        gap_before = attack_ts[0] - t_min
        attack_free_intervals.append(('before_attacks', t_min, attack_ts[0], gap_before))
    
    # Gaps between consecutive attacks
    for i in range(len(attack_ts) - 1):
        gap = attack_ts[i+1] - attack_ts[i]
        if gap > 0:
            attack_free_intervals.append(('between_attacks', attack_ts[i], attack_ts[i+1], gap))
    
    # Gap after last attack
    if attack_ts[-1] < t_max:
        gap_after = t_max - attack_ts[-1]
        attack_free_intervals.append(('after_attacks', attack_ts[-1], t_max, gap_after))
    
    # Sort by gap size
    attack_free_intervals.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n{'─'*70}")
    print("ATTACK-FREE INTERVALS (top 10):")
    print(f"{'─'*70}")
    for i, (where, start, end, gap) in enumerate(attack_free_intervals[:10], 1):
        print(f"{i:2}. {gap:10.6f}s  [{start:10.3f} → {end:10.3f}]  ({where})")
    
    if len(attack_free_intervals) > 10:
        print(f"... and {len(attack_free_intervals)-10} more")
    
    # Maximum attack-free gap
    max_gap = attack_free_intervals[0][3] if attack_free_intervals else 0
    
    print(f"\n{'─'*70}")
    print("WINDOW SIZE RECOMMENDATIONS:")
    print(f"{'─'*70}")
    print(f"Maximum attack-free gap: {max_gap:.6f} seconds")
    print(f"\n⚠️  IMPORTANT: Due to window alignment, you need gaps LARGER than window size!")
    print(f"\n➤ To GUARANTEE at least one benign window:")
    print(f"  Window size must be ≤ {max_gap / 2:.6f}s (50% of max gap)")
    print(f"\n➤ To LIKELY have benign windows (with good alignment):")
    print(f"  Window size could be ≤ {max_gap * 0.9:.6f}s (90% of max gap)")
    print(f"  But this depends on alignment luck!")
    
    # Test window sizes around the max gap
    print(f"\n{'─'*70}")
    print("BENIGN WINDOW COUNT AT DIFFERENT SIZES:")
    print(f"{'─'*70}")
    
    # Generate test sizes: always include standard sizes + sizes around max gap
    standard_sizes = [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    # Add sizes around the max gap
    if max_gap > 0:
        gap_based = [max_gap * mult for mult in [2, 1.5, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]]
        gap_based = [round(s, 4) for s in gap_based if s > 0]
        test_sizes = sorted(set(standard_sizes + gap_based), reverse=True)
    else:
        test_sizes = standard_sizes
    
    for w in test_sizes:
        if w <= 0:
            continue
        win_idx = np.floor((df['timestamp'] - t_min) / w).astype(int)
        temp_df = df.copy()
        temp_df['win'] = win_idx
        win_labels = temp_df.groupby('win')['label'].max()
        
        n_total = len(win_labels)
        n_attack = int((win_labels == 1).sum())
        n_benign = int((win_labels == 0).sum())
        
        marker = "✓" if n_benign > 0 else "✗"
        print(f"{marker} {w:8.4f}s → {n_total:5} total | {n_attack:5} attack | {n_benign:5} benign")
    
    print(f"{'='*70}\n")
    return max_gap


if __name__ == "__main__":
    # Test datasets
    datasets = [
        "data_parquet/03_Car-HackingDataset/DoS_attack.parquet",
        "data_parquet/04_CAN-IntrusionDataset_OTIDS/DoS_attack_dataset.parquet",
    ]
    
    for ds in datasets:
        try:
            analyze_window_requirements(ds)
        except Exception as e:
            print(f"Error processing {ds}: {e}")

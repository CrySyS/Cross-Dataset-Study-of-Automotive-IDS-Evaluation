import os, json
from typing import Dict, List, Iterable, Tuple, Sequence, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

OUTDIR = 'daga_eval'
_ensure_outdir(OUTDIR)

def _collect_window_labels_preds(model, test_paths: Sequence[str]):
    y_true, y_pred = [], []
    # Load test data and score with standard interface
    df_test = pd.concat([pd.read_parquet(p) for p in test_paths], ignore_index=True)
    for w, score in model.score_windows(df_test):
        y_true.append(int(w.label_window)); y_pred.append(int(score >= 1.0))
    import numpy as np
    if not y_true:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)

def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def _f1(y_true, y_pred) -> float:
    if y_true.size == 0:
        return 0.0
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return _f1_from_counts(tp, fp, fn)

def run_single_id_replay_boxplot(model_factory, train_paths, test_groups_by_label, n_values=range(1,11), out_name='fig4_single_id_replay'):
    results = {k: {n: [] for n in n_values} for k in test_groups_by_label.keys()}
    # Load training data once
    df_train = pd.concat([pd.read_parquet(p) for p in train_paths], ignore_index=True)
    for n in n_values:
        model = model_factory(n)
        model.fit(df_train)
        for label_name, paths in test_groups_by_label.items():
            for p in paths:
                y_true, y_pred = _collect_window_labels_preds(model, [p])
                results[label_name][n].append(_f1(y_true, y_pred))
    saved_paths = []
    for label_name, per_n in results.items():
        fig, ax = plt.subplots(figsize=(8,5))
        data = [per_n[n] for n in n_values]
        ax.boxplot(data, positions=list(range(1, len(n_values)+1)))
        import numpy as np
        medians = [np.median(per_n[n]) if len(per_n[n]) else 0.0 for n in n_values]
        ax.plot(range(1, len(n_values)+1), medians, marker='o')
        ax.set_title(f'Single-ID replay – {label_name}')
        ax.set_xlabel('n-gram length (n)')
        ax.set_ylabel('F1-score')
        ax.set_xticks(range(1, len(n_values)+1))
        ax.set_xticklabels(list(n_values))
        out_path = os.path.join(OUTDIR, f"{out_name}_{label_name}.png")
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
        saved_paths.append(out_path)
    fig, ax = plt.subplots(figsize=(8,5))
    for label_name, per_n in results.items():
        import numpy as np
        medians = [np.median(per_n[n]) if len(per_n[n]) else 0.0 for n in n_values]
        ax.plot(list(n_values), medians, marker='o', label=label_name)
    ax.set_title('Single-ID replay – median F1 by n')
    ax.set_xlabel('n-gram length (n)'); ax.set_ylabel('Median F1-score'); ax.legend()
    out_path = os.path.join(OUTDIR, f"{out_name}_medians_overlay.png")
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
    saved_paths.append(out_path)
    import json
    return json.dumps({'saved': saved_paths}, indent=2)

def run_sequence_heatmap(model_factory, train_paths, test_groups_by_length, n_values=range(1,11), lengths=range(2,11), title='Sequence replay (median F1)', out_name='fig_sequence_heatmap'):
    lengths = list(lengths); n_values = list(n_values)
    import numpy as np
    matrix = np.zeros((len(lengths), len(n_values)), dtype=float)
    # Load training data once
    df_train = pd.concat([pd.read_parquet(p) for p in train_paths], ignore_index=True)
    for j, n in enumerate(n_values):
        model = model_factory(n)
        model.fit(df_train)
        for i, L in enumerate(lengths):
            f1s = []
            for p in test_groups_by_length.get(L, []):
                y_true, y_pred = _collect_window_labels_preds(model, [p])
                f1s.append(_f1(y_true, y_pred))
            matrix[i, j] = float(np.median(f1s)) if f1s else 0.0
    fig, ax = plt.subplots(figsize=(9,5))
    ax.imshow(matrix, aspect='auto')
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel('n-gram length (n)')
    ax.set_yticks(range(len(lengths)))
    ax.set_yticklabels([str(L) for L in lengths])
    ax.set_ylabel('Injected sequence length')
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha='center', va='center', fontsize=8)
    fig.tight_layout()
    out_img = os.path.join(OUTDIR, f"{out_name}.png")
    fig.savefig(out_img, dpi=160); plt.close(fig)
    import pandas as pd
    out_csv = os.path.join(OUTDIR, f"{out_name}.csv")
    pd.DataFrame(matrix, index=[f'L{L}' for L in lengths], columns=[f'n{n}' for n in n_values]).to_csv(out_csv)
    import json
    return json.dumps({'image': out_img, 'csv': out_csv}, indent=2)

def run_dos_table(model_factory, train_paths, dos_test_paths, n_values=range(1,11), out_name='table_dos_lowest_id'):
    n_values = list(n_values)
    per_n_f1s = {n: [] for n in n_values}
    # Load training data once
    df_train = pd.concat([pd.read_parquet(p) for p in train_paths], ignore_index=True)
    for n in n_values:
        model = model_factory(n)
        model.fit(df_train)
        for p in dos_test_paths:
            y_true, y_pred = _collect_window_labels_preds(model, [p])
            per_n_f1s[n].append(_f1(y_true, y_pred))
    rows = []
    for n in n_values:
        import numpy as np
        vals = per_n_f1s[n]
        med = float(np.median(vals)) if vals else 0.0
        rows.append({'n': n, 'median_F1': med})
    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTDIR, f"{out_name}.csv")
    df.to_csv(out_csv, index=False)
    lines = []
    lines.append(f"{'n':>3} | {'median_F1':>10}")
    lines.append('-'*18)
    for _, r in df.iterrows():
        lines.append(f"{int(r['n']):>3} | {r['median_F1']:.4f}")
    out_txt = os.path.join(OUTDIR, f"{out_name}.txt")
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines))
    import json
    return json.dumps({'csv': out_csv, 'txt': out_txt}, indent=2)
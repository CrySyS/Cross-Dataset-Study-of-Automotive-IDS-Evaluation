import argparse
import csv
import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.metrics import roc_auc_score


def parse_csv_list(text, cast):
    values = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(cast(stripped))
    if not values:
        raise ValueError(f"No values provided in '{text}'")
    return values


def format_token(value):
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).replace(".", "p")
    return str(value)


def run_command(cmd, cwd, log_fp, step_name, trial_idx, total_trials, namespace):
    print(f"[{trial_idx}/{total_trials}] {step_name} -> {namespace}")
    log_fp.write("\n" + "=" * 120 + "\n")
    log_fp.write(f"[{datetime.now().isoformat()}] {step_name} {trial_idx}/{total_trials} namespace={namespace}\n")
    log_fp.write("COMMAND: " + " ".join(cmd) + "\n")
    log_fp.flush()
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"[{trial_idx}/{total_trials}] {step_name} done")
    except subprocess.CalledProcessError as exc:
        print(f"[{trial_idx}/{total_trials}] {step_name} failed (exit={exc.returncode})")
        log_fp.write(f"[{datetime.now().isoformat()}] FAILED exit={exc.returncode}\n")
        log_fp.flush()
        raise


def parse_label_filename(path):
    stem = path.stem
    if not stem.startswith("label_"):
        return None
    body = stem[len("label_") :]
    parts = body.split("_")
    if len(parts) < 4:
        return None
    per_of_samples = parts[-1]
    sampling_period = parts[-2]
    time_step = parts[-3]
    file_name = "_".join(parts[:-3])
    return file_name, time_step, sampling_period, per_of_samples


def parse_prediction_filename(path):
    stem = path.stem
    if not stem.startswith("prediction_"):
        return None
    body = stem[len("prediction_") :]
    parts = body.split("_")
    if len(parts) < 6:
        return None
    per_of_samples = parts[-1]
    time_factor = parts[-2]
    loss_factor = parts[-3]
    sampling_period = parts[-4]
    time_step = parts[-5]
    file_name = "_".join(parts[:-5])
    return file_name, time_step, sampling_period, loss_factor, time_factor, per_of_samples


def collect_trial_metrics(data_root, namespace, eval_type, excluded_prefixes):
    label_dir = data_root / "label" / namespace
    pred_dir = data_root / "prediction" / f"{namespace}_{eval_type}"

    if not label_dir.exists() or not pred_dir.exists():
        return [], pd.Series(dtype=float), pd.Series(dtype=float)

    labels = {}
    for label_file in sorted(label_dir.glob("label_*.csv")):
        parsed = parse_label_filename(label_file)
        if parsed is None:
            continue
        key = (parsed[0], parsed[1], parsed[2], parsed[3])
        labels[key] = label_file

    rows = []
    labels_all = []
    scores_all = []
    per_file_data = []
    for pred_file in sorted(pred_dir.glob("prediction_*.csv")):
        parsed = parse_prediction_filename(pred_file)
        if parsed is None:
            continue

        file_name, time_step, sampling_period, loss_factor, time_factor, per_of_samples = parsed
        if excluded_prefixes and any(file_name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        label_key = (file_name, time_step, sampling_period, per_of_samples)
        label_file = labels.get(label_key)
        if label_file is None:
            continue

        label_df = pd.read_csv(label_file)
        pred_df = pd.read_csv(pred_file)
        if "Label" not in label_df.columns or pred_df.shape[1] == 0:
            continue

        y_true = label_df["Label"].astype(float).to_numpy()
        y_score = pred_df.iloc[:, 0].astype(float).to_numpy()
        n = min(len(y_true), len(y_score))
        if n == 0:
            continue
        y_true = y_true[:n]
        y_score = y_score[:n]
        labels_all.append(y_true)
        scores_all.append(y_score)
        per_file_data.append(
            {
                "file_name": file_name,
                "time_step": int(time_step),
                "sampling_period": int(sampling_period),
                "loss_factor": float(loss_factor),
                "time_factor": float(time_factor),
                "per_of_samples": float(per_of_samples),
                "labels": y_true,
                "scores": y_score,
            }
        )

        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc = float("nan")

        rows.append(
            {
                "namespace": namespace,
                "file_name": file_name,
                "time_step": int(time_step),
                "sampling_period": int(sampling_period),
                "loss_factor": float(loss_factor),
                "time_factor": float(time_factor),
                "per_of_samples": float(per_of_samples),
                "auroc": auc,
            }
        )

    labels_concat = pd.Series(dtype=float)
    scores_concat = pd.Series(dtype=float)
    if labels_all:
        labels_concat = pd.Series(np.concatenate(labels_all))
        scores_concat = pd.Series(np.concatenate(scores_all))

    return rows, labels_concat, scores_concat, per_file_data


def compute_metrics_at_threshold(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if labels.size == 0 or scores.size == 0:
        return {
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()

    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, y_pred)),
        "f1": float(f1_score(labels, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, y_pred)),
        "precision": float(precision_score(labels, y_pred, zero_division=0)),
        "recall": float(recall_score(labels, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_macro_file_metrics(per_file_data, threshold):
    if not per_file_data or pd.isna(threshold):
        return {
            "macro_balanced_accuracy_best_f1": float("nan"),
            "macro_f1_best_f1": float("nan"),
            "macro_mcc_best_f1": float("nan"),
            "macro_precision_best_f1": float("nan"),
            "macro_recall_best_f1": float("nan"),
        }

    metric_rows = []
    for item in per_file_data:
        metric_rows.append(compute_metrics_at_threshold(item["labels"], item["scores"], threshold))

    metric_df = pd.DataFrame(metric_rows)
    return {
        "macro_balanced_accuracy_best_f1": float(metric_df["balanced_accuracy"].mean(skipna=True)),
        "macro_f1_best_f1": float(metric_df["f1"].mean(skipna=True)),
        "macro_mcc_best_f1": float(metric_df["mcc"].mean(skipna=True)),
        "macro_precision_best_f1": float(metric_df["precision"].mean(skipna=True)),
        "macro_recall_best_f1": float(metric_df["recall"].mean(skipna=True)),
    }


def compute_best_f1_metrics(labels_series, scores_series):
    if labels_series.empty or scores_series.empty:
        return {
            "overall_auroc": float("nan"),
            "best_f1_threshold": float("nan"),
            "balanced_accuracy_best_f1": float("nan"),
            "f1_best_f1": float("nan"),
            "mcc_best_f1": float("nan"),
            "precision_best_f1": float("nan"),
            "recall_best_f1": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    labels = labels_series.astype(int).to_numpy()
    scores = scores_series.astype(float).to_numpy()

    try:
        overall_auroc = float(roc_auc_score(labels, scores))
    except ValueError:
        overall_auroc = float("nan")

    if len(set(labels.tolist())) < 2:
        return {
            "overall_auroc": overall_auroc,
            "best_f1_threshold": float("nan"),
            "balanced_accuracy_best_f1": float("nan"),
            "f1_best_f1": float("nan"),
            "mcc_best_f1": float("nan"),
            "precision_best_f1": float("nan"),
            "recall_best_f1": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return {
            "overall_auroc": overall_auroc,
            "best_f1_threshold": float("nan"),
            "balanced_accuracy_best_f1": float("nan"),
            "f1_best_f1": float("nan"),
            "mcc_best_f1": float("nan"),
            "precision_best_f1": float("nan"),
            "recall_best_f1": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    f1_vals = 2.0 * precision[:-1] * recall[:-1]
    denom = precision[:-1] + recall[:-1]
    f1_vals = pd.Series(np.divide(f1_vals, denom, out=np.zeros_like(f1_vals), where=denom > 0))
    best_idx = int(f1_vals.idxmax())
    best_threshold = float(thresholds[best_idx])
    y_pred = (scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()

    return {
        "overall_auroc": overall_auroc,
        "best_f1_threshold": best_threshold,
        "balanced_accuracy_best_f1": float(balanced_accuracy_score(labels, y_pred)),
        "f1_best_f1": float(f1_score(labels, y_pred, zero_division=0)),
        "mcc_best_f1": float(matthews_corrcoef(labels, y_pred)),
        "precision_best_f1": float(precision_score(labels, y_pred, zero_division=0)),
        "recall_best_f1": float(recall_score(labels, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run isolated CANShield train/eval sweep and collect AUROC plus thresholded summary metrics.")
    parser.add_argument("--config-name", default="crysys")
    parser.add_argument("--dataset-name", default="crysys")
    parser.add_argument("--eval-type", default="original")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-epochs", default="500")
    parser.add_argument("--time-steps", default="50")
    parser.add_argument("--sampling-periods", default="1,5,10")
    parser.add_argument("--window-step-train", default="10")
    parser.add_argument("--window-step-valid", default="10")
    parser.add_argument("--window-step-test", default="10")
    parser.add_argument("--loss-factors", default="95")
    parser.add_argument("--time-factors", default="99")
    parser.add_argument("--signal-factors", default="95")
    parser.add_argument("--per-of-samples", default="1.0")
    parser.add_argument(
        "--exclude-file-prefixes",
        default="accelerator_attack_",
        help="Comma-separated file-name prefixes to exclude from summary/detail metrics",
    )
    parser.add_argument("--sweep-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--summary-dir", default="../artifacts/sweeps")
    parser.add_argument("--log-file", default="", help="Optional log file path. Defaults to <sweep_dir>/sweep.log")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_root = Path(__file__).resolve().parent
    data_root = (src_root / ".." / "data").resolve()
    summary_root = (src_root / args.summary_dir).resolve()

    max_epochs = parse_csv_list(args.max_epochs, int)
    time_steps = parse_csv_list(args.time_steps, int)
    sampling_periods = parse_csv_list(args.sampling_periods, int)
    window_step_train = parse_csv_list(args.window_step_train, int)
    window_step_valid = parse_csv_list(args.window_step_valid, int)
    window_step_test = parse_csv_list(args.window_step_test, int)
    loss_factors = parse_csv_list(args.loss_factors, float)
    time_factors = parse_csv_list(args.time_factors, float)
    signal_factors = parse_csv_list(args.signal_factors, float)
    per_of_samples = parse_csv_list(args.per_of_samples, float)
    excluded_prefixes = [token.strip() for token in args.exclude_file_prefixes.split(",") if token.strip()]

    if not (len(window_step_valid) == len(window_step_test) == 1):
        raise ValueError("window_step_valid and window_step_test must each contain exactly one value")
    if not (len(signal_factors) == 1):
        raise ValueError("signal_factors must contain exactly one value for this runner")

    grid = list(
        itertools.product(
            max_epochs,
            time_steps,
            sampling_periods,
            window_step_train,
            loss_factors,
            time_factors,
            per_of_samples,
        )
    )

    sweep_dir = summary_root / f"sweep_{args.sweep_tag}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else (sweep_dir / "sweep.log")
    if not log_file.is_absolute():
        log_file = (sweep_dir / log_file).resolve()

    print(f"Running {len(grid)} trials with sweep_tag={args.sweep_tag}")
    print(f"Log file: {log_file}")

    summary_rows = []
    detail_rows = []

    with open(log_file, "a", encoding="utf-8") as log_fp:
        log_fp.write(f"\n[{datetime.now().isoformat()}] Sweep start tag={args.sweep_tag} trials={len(grid)}\n")

        for trial_idx, (max_epoch, time_step, sampling_period, wst, loss_factor, time_factor, pos) in enumerate(grid, start=1):
            namespace = (
                f"{args.dataset_name}_swp_{args.sweep_tag}"
                f"_i{trial_idx:03d}"
                f"_ts{time_step}_sp{sampling_period}"
                f"_e{max_epoch}_ws{wst}"
                f"_lf{format_token(loss_factor)}_tf{format_token(time_factor)}"
                f"_p{format_token(pos)}"
            )

            overrides = [
                f"--config-name={args.config_name}",
                f"+output_dataset_name={namespace}",
                "debug_outputs=false",
                "debug_input_pipeline=false",
                f"eval_type={args.eval_type}",
                f"per_of_samples={pos}",
                f"max_epoch={max_epoch}",
                f"time_steps=[{time_step}]",
                f"sampling_periods=[{sampling_period}]",
                f"window_step_train={wst}",
                f"window_step_valid={window_step_valid[0]}",
                f"window_step_test={window_step_test[0]}",
                f"loss_factors=[{loss_factor}]",
                f"time_factors=[{time_factor}]",
                f"signal_factors=[{signal_factors[0]}]",
            ]

            train_cmd = [args.python, "run_development_canshield.py", *overrides]
            eval_cmd = [args.python, "run_evaluation_canshield.py", *overrides]

            print(f"[{trial_idx}/{len(grid)}] namespace={namespace}")

            if args.dry_run:
                print("  DRY train:", " ".join(train_cmd))
                print("  DRY eval :", " ".join(eval_cmd))
                continue

            if not args.skip_train:
                run_command(train_cmd, cwd=src_root, log_fp=log_fp, step_name="train", trial_idx=trial_idx, total_trials=len(grid), namespace=namespace)

            if not args.skip_eval:
                run_command(eval_cmd, cwd=src_root, log_fp=log_fp, step_name="eval", trial_idx=trial_idx, total_trials=len(grid), namespace=namespace)

            trial_details, labels_concat, scores_concat, per_file_data = collect_trial_metrics(
                data_root,
                namespace,
                args.eval_type,
                excluded_prefixes,
            )
            if trial_details:
                detail_rows.extend(trial_details)
                trial_df = pd.DataFrame(trial_details)
                threshold_metrics = compute_best_f1_metrics(labels_concat, scores_concat)
                macro_metrics = compute_macro_file_metrics(per_file_data, threshold_metrics["best_f1_threshold"])
                mean_auroc = float(trial_df["auroc"].mean(skipna=True))
                best_auroc = float(trial_df["auroc"].max(skipna=True))
                print(
                    f"[{trial_idx}/{len(grid)}] metrics n={trial_df.shape[0]} "
                    f"mean_auroc={mean_auroc:.4f} ba={threshold_metrics['balanced_accuracy_best_f1']:.4f} "
                    f"f1={threshold_metrics['f1_best_f1']:.4f} mcc={threshold_metrics['mcc_best_f1']:.4f} "
                    f"macro_ba={macro_metrics['macro_balanced_accuracy_best_f1']:.4f}"
                )
                summary_rows.append(
                    {
                        "namespace": namespace,
                        "trial_index": trial_idx,
                        "max_epoch": max_epoch,
                        "time_step": time_step,
                        "sampling_period": sampling_period,
                        "window_step_train": wst,
                        "loss_factor": float(loss_factor),
                        "time_factor": float(time_factor),
                        "per_of_samples": float(pos),
                        "n_predictions": int(trial_df.shape[0]),
                        "overall_auroc": threshold_metrics["overall_auroc"],
                        "mean_auroc": mean_auroc,
                        "median_auroc": float(trial_df["auroc"].median(skipna=True)),
                        "best_auroc": best_auroc,
                        "best_f1_threshold": threshold_metrics["best_f1_threshold"],
                        "balanced_accuracy_best_f1": threshold_metrics["balanced_accuracy_best_f1"],
                        "f1_best_f1": threshold_metrics["f1_best_f1"],
                        "mcc_best_f1": threshold_metrics["mcc_best_f1"],
                        "precision_best_f1": threshold_metrics["precision_best_f1"],
                        "recall_best_f1": threshold_metrics["recall_best_f1"],
                        "macro_balanced_accuracy_best_f1": macro_metrics["macro_balanced_accuracy_best_f1"],
                        "macro_f1_best_f1": macro_metrics["macro_f1_best_f1"],
                        "macro_mcc_best_f1": macro_metrics["macro_mcc_best_f1"],
                        "macro_precision_best_f1": macro_metrics["macro_precision_best_f1"],
                        "macro_recall_best_f1": macro_metrics["macro_recall_best_f1"],
                        "tn": threshold_metrics["tn"],
                        "fp": threshold_metrics["fp"],
                        "fn": threshold_metrics["fn"],
                        "tp": threshold_metrics["tp"],
                    }
                )
            else:
                print(f"[{trial_idx}/{len(grid)}] metrics unavailable (no prediction files found)")
                summary_rows.append(
                    {
                        "namespace": namespace,
                        "trial_index": trial_idx,
                        "max_epoch": max_epoch,
                        "time_step": time_step,
                        "sampling_period": sampling_period,
                        "window_step_train": wst,
                        "loss_factor": float(loss_factor),
                        "time_factor": float(time_factor),
                        "per_of_samples": float(pos),
                        "n_predictions": 0,
                        "overall_auroc": float("nan"),
                        "mean_auroc": float("nan"),
                        "median_auroc": float("nan"),
                        "best_auroc": float("nan"),
                        "best_f1_threshold": float("nan"),
                        "balanced_accuracy_best_f1": float("nan"),
                        "f1_best_f1": float("nan"),
                        "mcc_best_f1": float("nan"),
                        "precision_best_f1": float("nan"),
                        "recall_best_f1": float("nan"),
                        "macro_balanced_accuracy_best_f1": float("nan"),
                        "macro_f1_best_f1": float("nan"),
                        "macro_mcc_best_f1": float("nan"),
                        "macro_precision_best_f1": float("nan"),
                        "macro_recall_best_f1": float("nan"),
                        "tn": 0,
                        "fp": 0,
                        "fn": 0,
                        "tp": 0,
                    }
                )

    if args.dry_run:
        print("Dry run completed; no files were written.")
        return

    summary_file = sweep_dir / "summary.csv"
    details_file = sweep_dir / "details.csv"

    write_csv(
        summary_file,
        summary_rows,
        [
            "namespace",
            "trial_index",
            "max_epoch",
            "time_step",
            "sampling_period",
            "window_step_train",
            "loss_factor",
            "time_factor",
            "per_of_samples",
            "n_predictions",
            "overall_auroc",
            "best_f1_threshold",
            "balanced_accuracy_best_f1",
            "f1_best_f1",
            "mcc_best_f1",
            "precision_best_f1",
            "recall_best_f1",
            "macro_balanced_accuracy_best_f1",
            "macro_f1_best_f1",
            "macro_mcc_best_f1",
            "macro_precision_best_f1",
            "macro_recall_best_f1",
            "mean_auroc",
            "median_auroc",
            "best_auroc",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )

    if detail_rows:
        write_csv(
            details_file,
            detail_rows,
            [
                "namespace",
                "file_name",
                "time_step",
                "sampling_period",
                "loss_factor",
                "time_factor",
                "per_of_samples",
                "auroc",
            ],
        )

    print(f"Sweep summary written to: {summary_file}")
    if detail_rows:
        print(f"Sweep details written to: {details_file}")
    print(f"Sweep log written to: {log_file}")


if __name__ == "__main__":
    main()

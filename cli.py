
import argparse
import subprocess
import signal
import sys
from pathlib import Path

from unified_ids.eval.core import run_single
from unified_ids.eval.parallel import run_eval_plan
from unified_ids.eval.shutdown import set_shutdown


def signal_handler(signum, frame):
    """Handle SIGINT and SIGTERM to ensure clean shutdown."""
    # Request graceful shutdown and return immediately so the main
    # orchestration can perform cleanup. Keep this handler minimal
    # to avoid reentrancy issues.
    try:
        print("\nReceived termination signal — requesting shutdown...")
    except Exception:
        pass
    set_shutdown()
    raise KeyboardInterrupt


def run_post_reporting(out_dir: str) -> None:
    """Run post-evaluation reporting scripts against the given output directory."""
    results_root = Path(out_dir)
    if not results_root.exists():
        print(f"[WARN] Reporting skipped: results directory does not exist ({out_dir})")
        return

    repo_root = Path(__file__).resolve().parent
    scripts = [
        (
            "comparison tables",
            repo_root / "reporting" / "generate_comparison_table.py",
            ["--results-root", str(results_root)],
        ),
        (
            "metrics plots",
            repo_root / "reporting" / "plot_all_metrics.py",
            ["--results-root", str(results_root)],
        ),
        (
            "aggregate summary",
            repo_root / "reporting" / "aggregate_results.py",
            ["--results-root", str(results_root)],
        ),
        (
            "pooled best-level metrics",
            repo_root / "reporting" / "export_pooled_best_level_metrics.py",
            ["--results-root", str(results_root)],
        ),
        (
            "pooled per-attack metrics",
            repo_root / "reporting" / "export_pooled_per_attack_metrics.py",
            ["--results-root", str(results_root)],
        ),
        (
            "extra per-attack visuals",
            repo_root / "reporting" / "plot_per_attack_extra_visuals.py",
            ["--results-root", str(results_root)],
        ),
        (
            "metrics consistency validation",
            repo_root / "reporting" / "validate_metrics_consistency.py",
            ["--results-root", str(results_root)],
        ),
    ]

    print("\nRunning post-processing reports...")
    for label, script, extra_args in scripts:
        if not script.exists():
            print(f"[WARN] Skipping {label}: script not found ({script})")
            continue

        cmd = [sys.executable, str(script), *extra_args]
        print(f"[REPORT] {label}: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] {label} failed (exit={exc.returncode}). Continuing...")


def main():
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    parser = argparse.ArgumentParser(description="Unified IDS benchmark")
    parser.add_argument(
        "method",
        choices=[
            "mba_ocsvm_v2",
            "simple_ocsvm",
            "assoc_rules",
            "daga_ngram",
            "dcnn_song",
            "lstm_qin",
            "lstm_taylor",
            "canet",
            "ctcn",
            "all",  # special: use EVAL_PLAN
            "report",  # special: reporting only on an existing results folder
        ],
        help="Which IDS method to run; use 'all' for full matrix or 'report' for reporting-only",
    )
    parser.add_argument("--train_glob", required=False, help="Glob pattern for benign-only training files")
    parser.add_argument("--test_glob", required=False, help="Glob pattern for test files (benign + attack)")
    parser.add_argument(
        "--paper_eval",
        action="store_true",
        help="Include paper-style eval blocks where supported",
    )
    parser.add_argument(
        "--out_dir",
        default="results",
        help="Root directory to store metrics and plots (default: results)",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel processes to use for dataset-level jobs when method=all",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Enable interactive progress dashboard with live updates (requires rich library)",
    )
    parser.add_argument(
        "--skip_report",
        action="store_true",
        help="When method=all, skip automatic reporting (tables + plots).",
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Reporting-only mode (no evaluation)
    # ----------------------------------------------------------------
    if args.method == "report":
        run_post_reporting(args.out_dir)
        return

    # ----------------------------------------------------------------
    # Batch mode: run full evaluation matrix (possibly in parallel)
    # ----------------------------------------------------------------
    if args.method == "all":
        try:
            run_eval_plan(
                out_dir=args.out_dir,
                paper_eval=args.paper_eval,
                log_level=args.log,
                n_jobs=args.n_jobs,
                enable_ui=args.ui,
            )
        except RuntimeError as exc:
            print(f"\nEvaluation failed or remained incomplete: {exc}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nShutdown requested — exiting benchmark run.")
            return
        if not args.skip_report:
            run_post_reporting(args.out_dir)
        return

    # ----------------------------------------------------------------
    # Single-run mode
    # ----------------------------------------------------------------
    if not args.train_glob or not args.test_glob:
        parser.error("Single-run mode requires --train_glob and --test_glob")
    
    run_single(
        method=args.method,
        train_glob=args.train_glob,
        test_glob=args.test_glob,
        out_dir=args.out_dir,
        paper_eval=args.paper_eval,
        log_level=args.log,
    )


if __name__ == "__main__":
    main()

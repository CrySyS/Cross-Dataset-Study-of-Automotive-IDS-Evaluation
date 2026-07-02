
"""
Parallel orchestration for Unified IDS evaluation.

We parallelize at *dataset* level:
  - one process per dataset config in EVAL_PLAN
  - inside each process, methods for that dataset run sequentially,
    sharing train/test data in memory.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import traceback
import gc

from unified_ids.eval.core import (
    load_train_test,
    infer_dataset_tag_from_df,
    experiment_already_done,
    _run_single_on_loaded,
    _infer_dataset_tag_from_glob,
)
from unified_ids.eval.progress import ProgressTracker
from unified_ids.config.config import EVAL_PLAN
import os
import sys


def _worker_init(ts_out_dir: str):
    """Initializer for ProcessPoolExecutor workers.

    Writes a pid file to the out_dir/inprogress/worker_pids directory
    so the main process can discover and manage worker PIDs without
    relying on private executor internals.
    Also marks the process via an env var and redirects stdout/stderr
    to devnull to prevent interference with the rich Live UI.
    """
    try:
        os.environ["_UNIFIED_IDS_IN_PROCESS_POOL"] = "1"
        pid_dir = Path(ts_out_dir) / "inprogress" / "worker_pids"
        pid_dir.mkdir(parents=True, exist_ok=True)
        pid_file = pid_dir / f"{os.getpid()}.pid"
        pid_file.write_text("\n")
        
        # Redirect stdout and stderr to devnull to prevent worker prints
        # from interfering with the rich Live UI. All logging should go
        # to per-dataset log files via the logging module.
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        os.close(devnull)
    except Exception:
        pass

def get_timestamped_out_dir(base_dir: str = "results") -> Path:
    """Create a timestamped output directory, e.g. results/2025-12-08_14-30-00"""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(base_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _infer_dataset_name_from_glob(glob_pattern: str) -> str:
    """
    Infer dataset name from glob pattern.
    
    Examples:
      "data_parquet/02_Road/..." → "Road"
      "data_parquet/06_CrySyS/..." → "CrySyS"
      "DAGA/dataset/..." → "DAGA"
      "path/to/dataset/..." → "dataset"
    """
    from pathlib import Path
    path = Path(glob_pattern)
    
    # Get path parts and filter out wildcards
    parts = [p for p in path.parts if '*' not in p and p != '/']
    
    if not parts:
        return "unknown"
    
    # Strategy 1: Look for known dataset dirs with numbers (e.g., "02_Road", "06_CrySyS")
    for part in parts:
        if part and part[0].isdigit() and '_' in part:
            # Extract name after number_underscore pattern (e.g., "02_Road" → "Road")
            name = part.split('_', 1)[1] if '_' in part else part
            return name.strip('/')
    
    # Strategy 2: Look for "data_parquet" and use next part
    if "data_parquet" in parts:
        idx = parts.index("data_parquet")
        if idx + 1 < len(parts):
            name = parts[idx + 1]
            # Handle numbered prefixes (e.g., "02_Road" → "Road")
            if name and name[0].isdigit() and '_' in name:
                name = name.split('_', 1)[1]
            return name.strip('/')
    
    # Strategy 3: Use the last meaningful directory name
    for part in reversed(parts):
        if part and part.lower() not in ('data', 'data_parquet', 'dataset', 'datasets'):
            return part.strip('/')
    
    return "dataset"


def _run_dataset_job(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Helper for parallel execution at *dataset* level.

    job is a dict with keys:
      - cfg            (one entry from EVAL_PLAN)
      - out_dir
      - paper_eval
      - log_level
      - dashboard      (optional: ProgressDashboard instance for UI)
      - rich_console   (optional: rich.console.Console for logging)
    """
    cfg = job["cfg"]
    out_dir = job["out_dir"]
    paper_eval = job["paper_eval"]
    log_level = job["log_level"]

    from unified_ids.utils.logging import setup_logging
    from unified_ids.eval import metrics as metrics_mod
    from unified_ids.eval.shutdown import is_shutdown

    dataset_name = cfg.get("name")  # Optional: None triggers auto-inference
    methods = cfg["methods"]
    train_glob = cfg.get("train_glob")
    test_glob = cfg.get("test_glob")

    # Infer dataset name from glob pattern if not provided
    if dataset_name is None and test_glob:
        dataset_name = _infer_dataset_name_from_glob(test_glob)
    if dataset_name is None:
        dataset_name = "unknown"
    
    # Create logs directory and inprogress directory
    logs_dir = Path(out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    inprogress_dir = Path(out_dir) / "inprogress"
    inprogress_dir.mkdir(parents=True, exist_ok=True)
    
    # Create lock file to indicate this dataset is running
    lock_file = inprogress_dir / f"{dataset_name}.lock"
    lock_file.write_text(f"Started at {datetime.now().isoformat()}")
    
    # Setup per-dataset logging to file (console=WARNING to keep terminal quiet during multi-job runs)
    log_file = logs_dir / f"{dataset_name}.log"
    log = setup_logging(log_level, log_file=log_file, console_level="WARNING")
    log.info("=== Dataset job start: %s; methods=%s ===", dataset_name, methods)
    
    # Initialize progress tracker
    progress = ProgressTracker(out_dir)
    progress.job_started(dataset_name, methods)

    
    # Also print to stdout with dataset prefix for visibility during execution
    # (Workers should use logging to write to `log_file`; avoid dup2-style
    # stdout/stderr redirection which can interfere with the main UI.)
    print(f"[{dataset_name}] Starting dataset job with {len(methods)} methods")

    results: List[Dict[str, Any]] = []

    try:
        # Load train/test once per dataset (shared by all methods)
        df_tr, df_te = load_train_test(
            train_glob=train_glob,
            test_glob=test_glob,
            log=log,
        )
        if dataset_name is None:
            log.info("Inferring dataset tag from test data")
            dataset_tag = infer_dataset_tag_from_df(df_te)
            if dataset_tag == "unknown":
                dataset_tag = _infer_dataset_tag_from_glob(test_glob)
                log.info("Inferred dataset tag from glob: %s", dataset_tag)
        else:
            dataset_tag = dataset_name

        for method in methods:
            if is_shutdown():
                log.warning("Shutdown requested; stopping dataset job early for %s", dataset_name)
                break

            if experiment_already_done(out_dir, dataset_tag, method):
                existing_dir = Path(out_dir) / dataset_tag / method
                log.info(
                    "Skipping %s on dataset %s: results already exist in %s",
                    method,
                    dataset_name,
                    existing_dir,
                )
                progress.method_completed(dataset_name, method, success=True)
                results.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "ok": True,
                        "error": None,
                        "traceback": None,
                    }
                )
                continue

            try:
                progress.method_started(dataset_name, method)
                log.info("Running %s on dataset %s (shared df)", method, dataset_name)
                _run_single_on_loaded(
                    method=method,
                    df_tr=df_tr,
                    df_te=df_te,
                    dataset_tag=dataset_tag,
                    out_dir_root=out_dir,
                    metrics_mod=metrics_mod,
                    paper_eval=paper_eval,
                    log=log,
                    stage_cb=lambda stage, ds=dataset_name, m=method: progress.method_stage(ds, m, stage),
                )
                progress.method_completed(dataset_name, method, success=True)
                results.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "ok": True,
                        "error": None,
                        "traceback": None,
                    }
                )
                # Explicitly free memory between methods
                gc.collect()
            except Exception as e:
                progress.method_completed(dataset_name, method, success=False, error=str(e))
                log.error("Error running %s on %s: %s", method, dataset_name, e)
                results.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "ok": False,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
                # Explicitly free memory even on error
                gc.collect()

    except Exception as e:
        # Catastrophic dataset-level failure: mark all methods as failed
        tb = traceback.format_exc()
        for method in methods:
            results.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "ok": False,
                    "error": f"[DATASET LEVEL FAILURE] {e}",
                    "traceback": tb,
                }
            )

    # Cleanup: mark job as complete and remove lock file
    try:
        has_failures = any(not r["ok"] for r in results)
        progress.job_completed(dataset_name, success=not has_failures)
        lock_file.unlink(missing_ok=True)
    except:
        pass
    
    return results


def _save_failures_if_any(failures: List[Dict[str, Any]], out_dir: str) -> None:
    if not failures:
        return

    fail_dir = Path(out_dir) / "_failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fail_file = fail_dir / f"failures_{ts}.txt"

    with fail_file.open("w") as f:
        for i, fail in enumerate(failures, 1):
            f.write(f"--- FAILURE #{i} ---\n")
            f.write(f"Dataset: {fail['dataset']}\n")
            f.write(f"Method:  {fail['method']}\n")
            f.write(f"Error:   {fail['error']}\n\n")
            if fail.get("traceback"):
                f.write(f"{fail['traceback']}\n")
            f.write("\n")


def _compute_progress_counts(progress_data: Dict[str, Any], total_methods: int) -> Dict[str, int]:
    completed = 0
    failed = 0
    running = 0
    pending = 0

    for ds in progress_data.get("jobs", {}).values():
        completed += len(ds.get("methods_completed", []))
        failed += len(ds.get("methods_failed", []))
        for method_state in ds.get("methods", {}).values():
            status = method_state.get("status", "pending")
            if status == "running":
                running += 1
            elif status == "pending":
                pending += 1

    incomplete = max(0, total_methods - completed - failed)
    return {
        "completed": completed,
        "failed": failed,
        "running": running,
        "pending": pending,
        "incomplete": incomplete,
    }


def run_eval_plan(
    out_dir: str,
    paper_eval: bool,
    log_level: str,
    n_jobs: int,
    enable_ui: bool = False,
) -> None:
    """
    Run the full EVAL_PLAN (possibly in parallel at dataset level).
    
    Args:
        out_dir: Base output directory
        paper_eval: Whether to include paper-style evaluation blocks
        log_level: Logging level (INFO, DEBUG, etc.)
        n_jobs: Number of parallel workers
        enable_ui: Enable fancy progress dashboard (requires rich library)
    """
    # Create timestamped output directory
    ts_out_dir = get_timestamped_out_dir(out_dir)
    
    # Setup dashboard if requested
    dashboard = None
    rich_console = None
    if enable_ui:
        try:
            from unified_ids.eval.ui_progress import ProgressDashboard, check_rich_available
            
            if not check_rich_available():
                print("[WARNING] --ui flag requires 'rich' library. Install with: pip install rich")
                print("[WARNING] Falling back to text-only output\n")
            else:
                # Extract dataset names and methods for dashboard
                dataset_names = []
                all_methods = set()
                for cfg in EVAL_PLAN:
                    # Infer dataset name
                    test_glob = cfg.get("test_glob", "")
                    dataset_name = cfg.get("name") or _infer_dataset_name_from_glob(test_glob)
                    dataset_names.append(dataset_name)
                    all_methods.update(cfg.get("methods", []))
                
                dashboard = ProgressDashboard(ts_out_dir, dataset_names, sorted(all_methods))
                rich_console = dashboard.console  # Get console from dashboard
                dashboard.start()
                print()  # Extra line after dashboard starts
        except ImportError:
            print("[WARNING] --ui flag requires 'rich' library. Install with: pip install rich")
            print("[WARNING] Falling back to text-only output\n")
    
    jobs = [
        {
            "cfg": cfg,
            "out_dir": ts_out_dir,
            "paper_eval": paper_eval,
            "log_level": log_level,
        }
        for cfg in EVAL_PLAN
    ]

    total_jobs = len(jobs)
    if total_jobs == 0:
        print("[WARN] EVAL_PLAN is empty. Nothing to run.")
        return

    effective_n_jobs = n_jobs
    if effective_n_jobs < 1:
        print(f"[WARN] n_jobs={effective_n_jobs} is invalid; using n_jobs=1.")
        effective_n_jobs = 1
    if effective_n_jobs > total_jobs:
        print(
            f"[WARN] n_jobs={effective_n_jobs} exceeds dataset jobs ({total_jobs}); "
            f"using n_jobs={total_jobs}."
        )
        effective_n_jobs = total_jobs

    failures: List[Dict[str, Any]] = []
    shutdown_requested = False

    try:
        from unified_ids.eval.shutdown import is_shutdown

        if effective_n_jobs == 1:
            for job in jobs:
                if is_shutdown():
                    shutdown_requested = True
                    print("\nShutdown requested — stopping remaining dataset jobs...")
                    break
                dataset_name = job["cfg"].get("name", "<auto-infer>")
                if not dashboard:  # Only print if not using dashboard
                    print(f"=== Dataset job (sequential): {dataset_name} ===")
                results = _run_dataset_job(job)
                for res in results:
                    if res["ok"]:
                        if not dashboard:
                            print(f"=== Finished {res['method']} on {res['dataset']} ===")
                    else:
                        if not dashboard:
                            print(f"*** ERROR running {res['method']} on {res['dataset']}: {res['error']} ***")
                        failures.append(res)
                if is_shutdown():
                    shutdown_requested = True
                    print("\nShutdown requested — finishing current cleanup and exiting...")
                    break
        else:
            if not dashboard:
                print(f"Running {len(jobs)} dataset jobs with n_jobs={effective_n_jobs}")
            # Use worker initializer that records PIDs; pass ts_out_dir so
            # workers can write their pid files for main-process discovery.
            # Use a 'spawn' multiprocessing context to avoid inheriting
            # threads from the parent and to make worker shutdown more robust.
            ctx = multiprocessing.get_context('spawn')
            executor = ProcessPoolExecutor(max_workers=effective_n_jobs, initializer=_worker_init, initargs=(str(ts_out_dir),), mp_context=ctx)
            try:
                future_to_job = {executor.submit(_run_dataset_job, job): job for job in jobs}
                pending = set(future_to_job.keys())

                while pending:
                    if is_shutdown():
                        shutdown_requested = True
                        print("\nShutdown requested — cancelling remaining jobs...")
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except Exception:
                            pass

                        try:
                            pid_dir = Path(ts_out_dir) / "inprogress" / "worker_pids"
                            if pid_dir.exists():
                                for pf in pid_dir.glob("*.pid"):
                                    try:
                                        pid = int(pf.stem)
                                        os.kill(pid, 15)
                                    except Exception:
                                        try:
                                            os.kill(pid, 9)
                                        except Exception:
                                            pass
                        except Exception:
                            pass
                        break
                    try:
                        for future in as_completed(pending, timeout=1.0):
                            job = future_to_job[future]
                            dataset_name = job["cfg"].get("name", "<auto-infer>")
                            pending.remove(future)
                            try:
                                results = future.result()
                            except Exception as e:
                                if not dashboard:
                                    print(f"*** ERROR (outer) in dataset job {dataset_name}: {e} ***)")
                                else:
                                    dashboard.log_error(dataset_name, "<ALL>", str(e))
                                failures.append(
                                    {
                                        "dataset": dataset_name,
                                        "method": "<ALL>",
                                        "ok": False,
                                        "error": str(e),
                                        "traceback": traceback.format_exc(),
                                    }
                                )
                                continue

                            for res in results:
                                if res["ok"]:
                                    if not dashboard:
                                        print(f"[{res['dataset']:12s}] ✓ {res['method']}")
                                else:
                                    err_msg = res['error'][:100] if res['error'] else "Unknown error"
                                    if dashboard:
                                        dashboard.log_error(res['dataset'], res['method'], err_msg)
                                    else:
                                        print(f"[{res['dataset']:12s}] ✗ {res['method']}: {err_msg}")
                                    failures.append(res)
                    except Exception:
                        # Timeout or other issue — check for shutdown request
                        if is_shutdown():
                            shutdown_requested = True
                            print("\nShutdown requested — cancelling remaining jobs...")
                            try:
                                executor.shutdown(wait=False, cancel_futures=True)
                            except Exception:
                                pass

                            # Kill any worker PIDs recorded in ts_out_dir/inprogress/worker_pids
                            try:
                                pid_dir = Path(ts_out_dir) / "inprogress" / "worker_pids"
                                if pid_dir.exists():
                                    for pf in pid_dir.glob("*.pid"):
                                        try:
                                            pid = int(pf.stem)
                                            os.kill(pid, 15)
                                        except Exception:
                                            try:
                                                os.kill(pid, 9)
                                            except Exception:
                                                pass
                            except Exception:
                                pass

                            break
            except KeyboardInterrupt:
                print("\nInterrupted! Shutting down workers...")
                # Temporarily ignore further SIGINT so cleanup is not re-entered
                try:
                    import signal
                    prev_handler = signal.getsignal(signal.SIGINT)
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                except Exception:
                    prev_handler = None

                # Request executor to stop accepting new tasks and cancel pending futures
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass

                # Force-terminate any worker processes if they remain (best-effort).
                try:
                    procs = getattr(executor, "_processes", None)
                    if procs:
                        for pid, proc in list(procs.items()):
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                except Exception:
                    pass

                # Aggressive cleanup: use psutil (if available) to terminate/kill
                # any child processes. Fall back to pgrep-based approach if psutil
                # is not installed.
                try:
                    import psutil, os, signal, time
                    p = psutil.Process(os.getpid())
                    children = p.children(recursive=True)
                    for c in children:
                        try:
                            c.terminate()
                        except Exception:
                            pass
                    # give them a moment to exit
                    time.sleep(1.0)
                    # kill any remaining
                    for c in p.children(recursive=True):
                        try:
                            c.kill()
                        except Exception:
                            pass
                except Exception:
                    try:
                        import subprocess, os, signal
                        out = subprocess.check_output(["pgrep", "-P", str(os.getpid())]).decode().strip()
                        for ln in out.splitlines():
                            try:
                                os.kill(int(ln), signal.SIGTERM)
                            except Exception:
                                try:
                                    os.kill(int(ln), signal.SIGKILL)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                if dashboard:
                    dashboard.stop(interrupted=True)

                # Restore previous SIGINT handler
                try:
                    if prev_handler is not None:
                        signal.signal(signal.SIGINT, prev_handler)
                except Exception:
                    pass

                raise
            finally:
                executor.shutdown(wait=not shutdown_requested, cancel_futures=shutdown_requested)
    finally:
        # Stop dashboard if it was started (and not already stopped)
        if dashboard and dashboard.is_running:
            dashboard.stop()

    _save_failures_if_any(failures, str(ts_out_dir))

    progress_counts = {"completed": 0, "failed": len(failures), "running": 0, "pending": 0, "incomplete": 0}
    try:
        progress_tracker = ProgressTracker(ts_out_dir)
        progress_counts = _compute_progress_counts(progress_tracker.get_summary(), total_jobs)
    except Exception:
        # If progress file is unavailable/corrupt, still rely on collected failures
        pass

    if failures:
        print(f"\n=== SUMMARY: {len(failures)} runs failed ===")
    else:
        print("\n=== All runs completed successfully ===")
    
    # Print logs and progress info
    logs_dir = Path(ts_out_dir) / "logs"
    if logs_dir.exists():
        print(f"\nDetailed logs available in: {logs_dir}")
        print("To monitor individual datasets:")
        for log_file in sorted(logs_dir.glob("*.log")):
            print(f"  tail -f {log_file}")
    
    # Print progress summary
    progress_file = Path(ts_out_dir) / "progress.json"
    if progress_file.exists():
        print(f"\nProgress tracking: {progress_file}")
    
    # Show any inprogress lock files (should be empty if all completed)
    inprogress_dir = Path(ts_out_dir) / "inprogress"
    if inprogress_dir.exists():
        locks = list(inprogress_dir.glob("*.lock"))
        if locks:
            print(f"\n⚠ WARNING: {len(locks)} datasets still marked as in-progress:")
            for lock in locks:
                print(f"  {lock.stem}")

    if failures or progress_counts["incomplete"] > 0 or progress_counts["running"] > 0 or progress_counts["pending"] > 0:
        raise RuntimeError(
            "Evaluation did not complete cleanly "
            f"(completed={progress_counts['completed']}, failed={progress_counts['failed']}, "
            f"incomplete={progress_counts['incomplete']}, running={progress_counts['running']}, "
            f"pending={progress_counts['pending']}). "
            f"See {ts_out_dir}/logs and {ts_out_dir}/progress.json"
        )


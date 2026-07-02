"""
Progress tracking for multi-job evaluation pipeline.

Provides a unified way to track which datasets/methods are running,
completed, or failed during parallel execution.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import threading
from datetime import datetime
import os
import fcntl


class ProgressTracker:
    """
    Thread-safe progress tracker for evaluation pipeline.
    
    Maintains a JSON file with current status that can be queried
    by monitoring tools or the user.
    """
    
    def __init__(self, out_dir: Path):
        """
        Initialize progress tracker.
        
        Args:
            out_dir: Base output directory where progress.json will be written
        """
        self.out_dir = Path(out_dir)
        self.progress_file = self.out_dir / "progress.json"
        self.lock = threading.Lock()
        
        # Initialize progress file only if it doesn't exist
        if not self.progress_file.exists():
            self._write_progress({
                "start_time": datetime.now().isoformat(),
                "jobs": {}
            })
    
    def _read_progress(self) -> Dict[str, Any]:
        """Read current progress file (with cross-process lock)."""
        if not self.progress_file.exists():
            return {"start_time": datetime.now().isoformat(), "jobs": {}}
        try:
            lock_path = self.progress_file.with_suffix('.lock')
            with open(lock_path, 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                fcntl.flock(lock, fcntl.LOCK_UN)
            return data
        except Exception:
            return {"start_time": datetime.now().isoformat(), "jobs": {}}

    def _locked_update(self, update_fn) -> None:
        """Atomically read-modify-write progress file under a single lock."""
        try:
            lock_path = self.progress_file.with_suffix('.lock')
            with open(lock_path, 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)

                # Read current data
                if self.progress_file.exists():
                    try:
                        with open(self.progress_file, 'r') as f:
                            data = json.load(f)
                    except Exception:
                        data = {"start_time": datetime.now().isoformat(), "jobs": {}}
                else:
                    data = {"start_time": datetime.now().isoformat(), "jobs": {}}

                # Apply update
                update_fn(data)

                # Write back atomically
                temp_file = self.progress_file.with_suffix('.json.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                os.replace(temp_file, self.progress_file)

                fcntl.flock(lock, fcntl.LOCK_UN)
        except Exception:
            # Silently fail - don't interrupt the pipeline for logging issues
            pass
    
    def _write_progress(self, data: Dict[str, Any]) -> None:
        """Write progress file (with cross-process lock to avoid corruption)."""
        try:
            lock_path = self.progress_file.with_suffix('.lock')
            with open(lock_path, 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                # Write to temp file first, then atomically move
                temp_file = self.progress_file.with_suffix('.json.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                os.replace(temp_file, self.progress_file)
                fcntl.flock(lock, fcntl.LOCK_UN)
        except Exception:
            # Silently fail - don't interrupt the pipeline for logging issues
            pass
    
    def job_started(self, dataset_name: str, methods: List[str]) -> None:
        """Mark that a dataset job has started."""
        with self.lock:
            def _update(data):
                data.setdefault("jobs", {})
                data["jobs"][dataset_name] = {
                    "status": "in-progress",
                    "methods_total": len(methods),
                    "methods_completed": [],
                    "methods_failed": [],
                    "current_method": None,
                    "methods": {
                        m: {"status": "pending", "stage": "pending", "last_update": None}
                        for m in methods
                    },
                    "start_time": datetime.now().isoformat(),
                }
            self._locked_update(_update)
    
    def method_started(self, dataset_name: str, method: str) -> None:
        """Mark that a method has started running."""
        with self.lock:
            def _update(data):
                if dataset_name in data.get("jobs", {}):
                    data["jobs"][dataset_name]["current_method"] = method
                    data["jobs"][dataset_name]["method_start_time"] = datetime.now().isoformat()
                    methods = data["jobs"][dataset_name].setdefault("methods", {})
                    methods.setdefault(method, {})
                    methods[method]["status"] = "running"
                    methods[method]["stage"] = "starting"
                    methods[method]["last_update"] = datetime.now().isoformat()
            self._locked_update(_update)

    def method_stage(self, dataset_name: str, method: str, stage: str) -> None:
        """Update current stage for a running method."""
        with self.lock:
            def _update(data):
                if dataset_name in data.get("jobs", {}):
                    methods = data["jobs"][dataset_name].setdefault("methods", {})
                    methods.setdefault(method, {})
                    methods[method]["status"] = methods[method].get("status", "running")
                    methods[method]["stage"] = stage
                    methods[method]["last_update"] = datetime.now().isoformat()
            self._locked_update(_update)
    
    def method_completed(self, dataset_name: str, method: str, success: bool = True, error: str = None) -> None:
        """Mark that a method has completed."""
        with self.lock:
            def _update(data):
                if dataset_name in data.get("jobs", {}):
                    job = data["jobs"][dataset_name]
                    job["current_method"] = None
                    job["method_start_time"] = None
                    methods = job.setdefault("methods", {})
                    methods.setdefault(method, {})
                    if success:
                        if method not in job["methods_completed"]:
                            job["methods_completed"].append(method)
                        methods[method]["status"] = "completed"
                        methods[method]["stage"] = "done"
                    else:
                        if method not in job["methods_failed"]:
                            job["methods_failed"].append(method)
                        methods[method]["status"] = "failed"
                        methods[method]["stage"] = "failed"
                        # Store error info
                        if "errors" not in job:
                            job["errors"] = {}
                        job["errors"][method] = error or "Unknown error"
                    methods[method]["last_update"] = datetime.now().isoformat()
            self._locked_update(_update)
    
    def job_completed(self, dataset_name: str, success: bool = True) -> None:
        """Mark that a dataset job has completed."""
        with self.lock:
            def _update(data):
                if dataset_name in data.get("jobs", {}):
                    job = data["jobs"][dataset_name]
                    job["status"] = "completed" if success else "failed"
                    job["end_time"] = datetime.now().isoformat()
            self._locked_update(_update)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current progress summary."""
        with self.lock:
            return self._read_progress()
    
    def print_summary(self) -> None:
        """Print human-readable progress summary."""
        data = self.get_summary()
        jobs = data.get("jobs", {})
        
        if not jobs:
            print("No jobs started yet.")
            return
        
        print("\n" + "="*80)
        print("PROGRESS SUMMARY")
        print("="*80)
        
        for dataset_name, job in jobs.items():
            status = job.get("status", "unknown")
            completed = len(job.get("methods_completed", []))
            failed = len(job.get("methods_failed", []))
            total = job.get("methods_total", 0)
            current = job.get("current_method", "—")
            
            # Status icon
            if status == "completed":
                icon = "✓"
            elif status == "failed":
                icon = "✗"
            else:
                icon = "◐"
            
            print(f"\n[{icon}] {dataset_name}")
            print(f"    Status: {status} | Progress: {completed}/{total} methods")
            if failed > 0:
                print(f"    Failed: {failed}")
            if current != "—":
                print(f"    Running: {current}")

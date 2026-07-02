"""
Interactive progress dashboard for parallel evaluation runs.

This module provides a fancy terminal UI with live progress bars, job status,
log file locations, and time estimates. It's optional and gracefully falls back
to text-only output if the 'rich' library is not available.

Usage:
    dashboard = ProgressDashboard(out_dir, datasets, methods)
    dashboard.start()
    # ... run jobs ...
    dashboard.stop()
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import threading

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ProgressDashboard:
    """
    Interactive dashboard for monitoring parallel evaluation progress.
    
    Displays:
    - Real-time progress bars per dataset
    - Method completion status
    - Log file locations
    - Time elapsed and estimated time remaining
    - Overall pipeline status
    """
    
    def __init__(self, out_dir: Path, datasets: List[str], methods: List[str], stale_threshold_minutes: int = 120):
        """
        Initialize dashboard.
        
        Args:
            out_dir: Output directory (for finding progress.json and logs/)
            datasets: List of dataset names
            methods: List of method names
            stale_threshold_minutes: Minutes after which a running method is considered stalled (default: 120 = 2 hours)
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "Progress dashboard requires 'rich' library. "
                "Install with: pip install rich"
            )
        
        self.out_dir = Path(out_dir)
        self.datasets = datasets
        self.methods = methods
        self.console = Console()
        self.start_time = datetime.now()
        self.is_running = False
        self.live = None
        self._stop_event = threading.Event()
        self.recent_errors = []  # Track recent errors for display
        self._error_lock = threading.Lock()
        self.stale_threshold = timedelta(minutes=stale_threshold_minutes)
        
        # Import here to avoid issues when rich is not available
        from unified_ids.eval.progress import ProgressTracker
        self.tracker = ProgressTracker(out_dir)
    
    def log_error(self, dataset: str, method: str, error_msg: str):
        """Log an error that should be visible in the dashboard."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._error_lock:
            self.recent_errors.append({
                "time": timestamp,
                "dataset": dataset,
                "method": method,
                "error": error_msg[:100]  # Truncate long errors
            })
            # Keep only last 5 errors
            if len(self.recent_errors) > 5:
                self.recent_errors.pop(0)
    
    def _create_dashboard(self) -> Table:
        """Create the dashboard layout with current status."""
        # Read current progress
        state = self.tracker.get_summary()
        
        # Main table
        table = Table(title=f"[bold cyan]Pipeline Progress[/bold cyan]", 
                     title_style="bold cyan",
                     show_header=True, 
                     header_style="bold magenta",
                     expand=True)
        
        table.add_column("Dataset", style="cyan", width=15)
        table.add_column("Method", style="yellow", width=20)
        table.add_column("Status", style="green", width=15)
        table.add_column("Stage", style="magenta", width=12)
        table.add_column("Log File", style="dim", width=30)
        
        # Calculate statistics
        total_jobs = len(self.datasets) * len(self.methods)
        completed_jobs = 0
        running_jobs = 0
        failed_jobs = 0
        stalled_jobs = 0
        
        # Process each dataset
        for dataset in self.datasets:
            dataset_status = state.get("jobs", {}).get(dataset, {})
            methods_status = dataset_status.get("methods", {})
            current_method = dataset_status.get("current_method")
            method_start_time_str = dataset_status.get("method_start_time")
            
            # Check if current method is stalled
            is_stalled = False
            if current_method and method_start_time_str:
                try:
                    method_start_time = datetime.fromisoformat(method_start_time_str)
                    elapsed = datetime.now() - method_start_time
                    if elapsed > self.stale_threshold:
                        is_stalled = True
                        stalled_jobs += 1
                except:
                    pass
            
            dataset_complete = dataset_status.get("status") == "completed"
            dataset_failed = dataset_status.get("status") == "failed"
            
            log_file_path = f"logs/{dataset}.log"
            
            for method in self.methods:
                method_status = methods_status.get(method, {})
                status = method_status.get("status", "pending")
                stage = method_status.get("stage", "pending")
                
                # Check if this is the stalled method
                is_method_stalled = (is_stalled and method == current_method)
                
                # Update counters
                if status == "completed":
                    completed_jobs += 1
                    status_text = "[green]✓ Done[/green]"
                    stage_text = "done"
                elif status == "failed":
                    failed_jobs += 1
                    status_text = "[red]✗ Failed[/red]"
                    stage_text = "failed"
                elif status == "running":
                    running_jobs += 1
                    if is_method_stalled:
                        status_text = "[yellow]⚠ Stalled?[/yellow]"
                        stage_text = f"{stage} ⚠"
                    else:
                        status_text = "[yellow]⚙ Running[/yellow]"
                        stage_text = stage
                else:  # pending
                    status_text = "[dim]○ Pending[/dim]"
                    stage_text = "pending"
                
                table.add_row(
                    dataset if method == self.methods[0] else "",  # Only show dataset name once
                    method,
                    status_text,
                    stage_text,
                    log_file_path if method == self.methods[0] else ""
                )
        
        # Summary panel
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        # Estimate remaining time
        if completed_jobs > 0:
            avg_time_per_job = elapsed.total_seconds() / completed_jobs
            remaining_jobs = total_jobs - completed_jobs
            eta_seconds = avg_time_per_job * remaining_jobs
            eta = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta).split('.')[0]
        else:
            eta_str = "calculating..."
        
        summary = (
            f"[bold]Overall:[/bold] {completed_jobs}/{total_jobs} jobs completed"
            f" | [yellow]{running_jobs} running[/yellow]"
            f" | [red]{failed_jobs} failed[/red]"
        )
        if stalled_jobs > 0:
            summary += f" | [yellow]⚠ {stalled_jobs} stalled[/yellow]"
        summary += (
            f"\n[bold]Elapsed:[/bold] {elapsed_str}"
            f" | [bold]ETA:[/bold] {eta_str}\n"
            f"[bold]Log directory:[/bold] {self.out_dir}/logs/\n"
            f"[bold]Progress file:[/bold] {self.out_dir}/progress.json"
        )
        
        if stalled_jobs > 0:
            summary += f"\n[yellow]⚠ {stalled_jobs} job(s) appear stalled (no progress for {self.stale_threshold.seconds // 3600}h+)[/yellow]"
        
        summary_panel = Panel(summary, title="Summary", border_style="blue")
        
        # Errors panel (if any recent errors)
        errors_panel = None
        with self._error_lock:
            if self.recent_errors:
                error_lines = []
                for err in self.recent_errors:
                    error_lines.append(
                        f"[red]{err['time']}[/red] [{err['dataset']}] {err['method']}: {err['error']}"
                    )
                errors_panel = Panel(
                    "\n".join(error_lines),
                    title="[red]Recent Errors[/red]",
                    border_style="red"
                )
        
        # Combine into layout
        layout = Table.grid(expand=True)
        layout.add_row(summary_panel)
        if errors_panel:
            layout.add_row("")
            layout.add_row(errors_panel)
        layout.add_row("")
        layout.add_row(table)
        
        return layout
    
    def start(self):
        """Start the live dashboard display."""
        if not RICH_AVAILABLE:
            self.console.print("[yellow]Warning: rich library not available, using text-only output[/yellow]")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        self._stop_event.clear()
        
        # Start live display with stdout/stderr redirection to prevent
        # any stray prints from interfering with the display
        self.live = Live(
            self._create_dashboard(),
            console=self.console,
            refresh_per_second=2,
            redirect_stdout=True,
            redirect_stderr=True
        )
        self.live.start()
        
        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Ensure stop is called even if exceptions occur in the with-block
        try:
            self.stop()
        except Exception:
            pass
    
    def _update_loop(self):
        """Background thread that updates the dashboard periodically."""
        while not self._stop_event.is_set():
            if self.is_running and self.live:
                try:
                    self.live.update(self._create_dashboard())
                except Exception as e:
                    # Silently ignore errors during update (e.g., terminal resize issues)
                    pass
            time.sleep(0.5)  # Update twice per second
    
    def stop(self, interrupted: bool = False):
        """Stop the live dashboard display.
        
        Args:
            interrupted: If True, indicates cleanup due to KeyboardInterrupt
        """
        if not RICH_AVAILABLE:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        if self.live:
            try:
                # Final update before stopping
                if not interrupted:
                    self.live.update(self._create_dashboard())
                self.live.stop()
            except Exception:
                # Ignore errors during cleanup
                pass
        
        # Wait for update thread to finish (with timeout)
        if hasattr(self, '_update_thread') and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        
        if not interrupted:
            # Print final status
            state = self.tracker.get_summary()
            total = len(self.datasets) * len(self.methods)
            completed = sum(
                1 for ds in state.get("jobs", {}).values()
                for method in ds.get("methods_completed", [])
            )
            failed = sum(
                1 for ds in state.get("jobs", {}).values()
                for method in ds.get("methods_failed", [])
            )

            running_or_pending = 0
            for ds in state.get("jobs", {}).values():
                for method_state in ds.get("methods", {}).values():
                    status = method_state.get("status", "pending")
                    if status in {"running", "pending"}:
                        running_or_pending += 1

            incomplete = max(0, total - completed - failed)
            if failed == 0 and incomplete == 0 and running_or_pending == 0:
                self.console.print("\n[bold green]Pipeline execution completed successfully![/bold green]\n")
            else:
                self.console.print("\n[bold yellow]Pipeline execution finished with failures/incomplete jobs.[/bold yellow]\n")
            
            self.console.print(
                f"[bold]Results:[/bold] {completed}/{total} completed, {failed} failed, {incomplete} incomplete"
            )
            self.console.print(f"[bold]Logs:[/bold] {self.out_dir}/logs/")
            self.console.print(f"[bold]Progress details:[/bold] {self.out_dir}/progress.json")
        else:
            # Interrupted - just print minimal message
            self.console.print("\n[yellow]Dashboard stopped due to interruption[/yellow]")


def check_rich_available() -> bool:
    """Check if rich library is available."""
    return RICH_AVAILABLE

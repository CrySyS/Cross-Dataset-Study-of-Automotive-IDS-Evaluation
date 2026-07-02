"""
Window Size Hyperparameter Study Module

A flexible toolkit for analyzing how window size affects MBA OCSVM V2 
performance across different CAN bus intrusion detection datasets.

Usage:
    python -m window_size_hyperparameter_module --dataset_name "Car-Hacking" ...
    
    Or see examples/ directory for preset configurations.
"""

__version__ = "1.0"
__author__ = "IDS Comparison Framework"

from pathlib import Path

# Module paths
MODULE_ROOT = Path(__file__).parent
RESULTS_DIR = MODULE_ROOT / "results"
EXAMPLES_DIR = MODULE_ROOT / "examples"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

__all__ = [
    'MODULE_ROOT',
    'RESULTS_DIR', 
    'EXAMPLES_DIR',
]

#!/usr/bin/env python3
"""
Entry point for window size hyperparameter study module.
Allows running as: python -m window_size_hyperparameter_module ...
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from window_size_hyperparameter_module.window_size_hyperparameter_study import main

if __name__ == "__main__":
    sys.exit(main())

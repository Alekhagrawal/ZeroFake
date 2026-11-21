# Orchestrates and compares all five ablation configurations (Configs A through E) and compiles results/results_summary.csv.
# Inputs: Saved checkpoints for Configs A-E and test split DataLoaders; Outputs: Unified ablation comparison table in results/results_summary.csv.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

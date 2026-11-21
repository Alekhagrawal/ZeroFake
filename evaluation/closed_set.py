# Evaluates trained checkpoints on known manipulation generators (test_closed split) for in-distribution detection performance.
# Inputs: Trained checkpoint and test_closed DataLoader; Outputs: Closed-set ROC-AUC, EER, F1-score, and prediction records.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

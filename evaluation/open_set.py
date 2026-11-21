# Evaluates open-set rejection metrics using the validation-calibrated threshold tau_reject on closed, unseen, and cross-dataset test splits.
# Inputs: Test predictions, uncertainty scores U(x), and val-calibrated tau_reject; Outputs: Selective Accuracy, Coverage, and Real/Fake Rejection Rates.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Runs robustness stress tests under JPEG compression (Q75, Q50), Gaussian blur, Gaussian noise, and 112x112 low-resolution downsampling.
# Inputs: Trained checkpoint and clean test face crops; Outputs: Perturbation-specific ROC-AUC, EER, and selective accuracy tables.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

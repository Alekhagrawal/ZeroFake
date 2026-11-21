# Evaluates zero-day generalization on the held-out Leave-One-Generator-Out split (test_unseen) and cross-dataset Celeb-DF v2 benchmark.
# Inputs: Trained checkpoint, test_unseen DataLoader, and Celeb-DF v2 DataLoader; Outputs: Unseen-generator and cross-dataset ROC-AUC, EER, and F1 metrics.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

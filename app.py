# Interactive Gradio web application for live deepfake detection displaying REAL/FAKE/UNKNOWN badge, uncertainty gauge, prototype bar chart, and Grad-CAM.
# Inputs: User-uploaded face image/video or pre-loaded sample buttons plus trained ZeroFake checkpoint; Outputs: Interactive visual dashboard in browser.

import sys
from pathlib import Path

# Insert repository root at sys.path[0] so Colab's pre-installed Hugging Face 'datasets' package
# never shadows ZeroFake's local 'datasets.deepfake_dataset' module.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
if "datasets" in sys.modules and not hasattr(sys.modules["datasets"], "__path__"):
    del sys.modules["datasets"]

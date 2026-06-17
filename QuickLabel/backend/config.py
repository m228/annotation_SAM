"""Runtime configuration and paths.

QuickLabel is self-contained: it ships its own copy of the ``ml_backend`` SAM
service source (``QuickLabel/ml_backend``) and the model checkpoints
(``QuickLabel/models``). Nothing here depends on the original VisoLabel folder.
Everything can be overridden with environment variables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# QuickLabel/backend/config.py -> QuickLabel
QUICKLABEL_DIR = Path(__file__).resolve().parents[1]

# Local copy of the SAM service package and the model checkpoints.
ML_BACKEND_DIR = QUICKLABEL_DIR / "ml_backend"
MODELS_DIR = Path(os.environ.get("QUICKLABEL_MODELS", QUICKLABEL_DIR / "models"))

# Where annotation projects live on disk.
PROJECTS_DIR = Path(os.environ.get("QUICKLABEL_PROJECTS", QUICKLABEL_DIR / "projects"))
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# Static web assets (vanilla HTML/JS/CSS, no build step).
WEB_DIR = QUICKLABEL_DIR / "web"

HOST = os.environ.get("QUICKLABEL_HOST", "127.0.0.1")
PORT = int(os.environ.get("QUICKLABEL_PORT", "8765"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_ml_backend_importable() -> None:
    """Make ``import ml_backend...`` resolve to the bundled copy, and point the
    SAM service at the bundled checkpoints regardless of the current directory."""
    root = str(QUICKLABEL_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Reduce CUDA fragmentation OOM on small (8 GB) GPUs shared with the desktop.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Explicit checkpoint paths so resolution never depends on cwd.
    sam2 = MODELS_DIR / "sam2.1_hiera_large.pt"
    sam3 = MODELS_DIR / "sam3.pt"
    if sam2.is_file():
        os.environ.setdefault("ML_BACKEND_SAM2_PATH", str(sam2))
    if sam3.is_file():
        os.environ.setdefault("ML_BACKEND_SAM3_PATH", str(sam3))

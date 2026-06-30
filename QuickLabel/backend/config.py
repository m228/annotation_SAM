"""Runtime configuration and paths.

QuickLabel is self-contained: it ships its own copy of the ``ml_backend`` SAM
service source (``QuickLabel/ml_backend``) and the model checkpoints
(``QuickLabel/models``). Nothing here depends on the original VisoLabel folder.
Everything can be overridden with environment variables.

PyInstaller frozen support: when running as a frozen bundle (sys.frozen=True),
BUNDLE_DIR is ``sys._MEIPASS`` (the ``_internal/`` folder) for bundled code and
data, while DATA_DIR is the folder next to the exe for user data (models,
projects). ML subprocesses still use the .venv Python found via
``find_python_executable()``, and import ml_backend via BUNDLE_DIR on PYTHONPATH.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _frozen() -> bool:
    return getattr(sys, "frozen", False)


def _bundle_dir() -> Path:
    """Folder that holds bundled code/data (sys._MEIPASS when frozen)."""
    if _frozen():
        return Path(sys._MEIPASS)          # type: ignore[attr-defined]
    # Development: this file is at QuickLabel/backend/config.py
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    """Folder next to the exe for user data; project root when from source."""
    if _frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


# BUNDLE_DIR: bundled code/assets (sys._MEIPASS or project root).
BUNDLE_DIR = _bundle_dir()
# DATA_DIR / QUICKLABEL_DIR: user data root (next to exe or project root).
DATA_DIR = _data_dir()
QUICKLABEL_DIR = DATA_DIR

# ml_backend source lives in BUNDLE_DIR (as datas in the spec) so the
# subprocess can import it by adding BUNDLE_DIR to PYTHONPATH.
ML_BACKEND_DIR = BUNDLE_DIR / "ml_backend"

MODELS_DIR = Path(os.environ.get("QUICKLABEL_MODELS", DATA_DIR / "models"))

PROJECTS_DIR = Path(os.environ.get("QUICKLABEL_PROJECTS", DATA_DIR / "projects"))
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# Static web assets live in BUNDLE_DIR (included as datas).
WEB_DIR = BUNDLE_DIR / "web"

HOST = os.environ.get("QUICKLABEL_HOST", "127.0.0.1")
PORT = int(os.environ.get("QUICKLABEL_PORT", "8765"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

GITHUB_REPO = "m228/annotation_SAM"


def read_version() -> str:
    """Read VERSION file from DATA_DIR or BUNDLE_DIR."""
    for base in (DATA_DIR, BUNDLE_DIR):
        vf = base / "VERSION"
        if vf.is_file():
            try:
                return vf.read_text(encoding="utf-8").strip()
            except Exception:
                pass
    return "0.0.0"


def find_python_executable() -> str:
    """Return a Python interpreter path suitable for ML subprocesses.

    When the server runs from source ``sys.executable`` is the correct Python.
    When frozen, ``sys.executable`` is the frozen exe itself — useless for
    running ``python -m ml_backend``. In that case we hunt for the .venv Python.
    """
    if not _frozen():
        return sys.executable

    override = os.environ.get("QUICKLABEL_PYTHON", "").strip()
    if override and Path(override).is_file():
        return override

    base = Path(sys.executable).resolve().parent
    for candidate in (
        base / ".venv" / "Scripts" / "python.exe",
        base.parent / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return str(candidate)

    return "python"


def ensure_ml_backend_importable() -> None:
    """Make ``import ml_backend...`` resolve to the bundled copy and set
    checkpoint env vars so the subprocess always finds the models."""
    # BUNDLE_DIR holds ml_backend source (as datas when frozen, as source dir
    # when running from code). DATA_DIR is needed for the non-frozen dev case.
    for root in (str(BUNDLE_DIR), str(DATA_DIR)):
        if root not in sys.path:
            sys.path.insert(0, root)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    sam2 = MODELS_DIR / "sam2.1_hiera_large.pt"
    sam3 = MODELS_DIR / "sam3.pt"
    if sam2.is_file():
        os.environ.setdefault("ML_BACKEND_SAM2_PATH", str(sam2))
    if sam3.is_file():
        os.environ.setdefault("ML_BACKEND_SAM3_PATH", str(sam3))

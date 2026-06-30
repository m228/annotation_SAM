# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the full QuickLabel server bundle (one-folder).
#
# What gets bundled:
#   Python runtime + fastapi + uvicorn + numpy + pillow + opencv + scipy +
#   psutil + python-multipart + all QuickLabel backend source.
#
# What stays on disk (user must provide / setup.ps1 creates):
#   .venv/   — heavy ML deps: torch, sam2, sam3 (used only in subprocesses)
#   models/  — SAM weight files (sam2.1_hiera_large.pt, sam3.pt)
#
# Build:  pyinstaller app.spec   (from QuickLabel/ directory)
# Output: dist/QuickLabel/       (the distributable folder)

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

HERE = Path(SPECPATH)          # QuickLabel/ directory

block_cipher = None

# ── Collect packages with native extensions ──────────────────────────────────
cv2_d, cv2_b, cv2_h = collect_all('cv2')
scipy_d, scipy_b, scipy_h = collect_all('scipy')
numpy_d, numpy_b, numpy_h = collect_all('numpy')

uvicorn_h = collect_submodules('uvicorn')
anyio_h   = collect_submodules('anyio')
starlette_h = collect_submodules('starlette')

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    [str(HERE / 'run.py')],
    pathex=[str(HERE)],
    binaries=cv2_b + scipy_b + numpy_b,
    datas=(
        cv2_d + scipy_d + numpy_d
        # Static web UI (html/js/css)
        + [(str(HERE / 'web'), 'web')]
        # ml_backend SOURCE files — the ML subprocess (venv python) imports them.
        + [(str(HERE / 'ml_backend'), 'ml_backend')]
        # VERSION for read_version()
        + [(str(HERE / 'VERSION'), '.')]
    ),
    hiddenimports=(
        cv2_h + scipy_h + numpy_h + uvicorn_h + anyio_h + starlette_h
        + [
            # Local backend modules (PyInstaller can't see them via string imports)
            'backend', 'backend.config', 'backend.server', 'backend.store',
            'backend.sam_runtime', 'backend.train_runtime', 'backend.jobs',
            'backend.export_common', 'backend.yolo_export', 'backend.coco_export',
            'backend.dataset_import',
            # FastAPI / Starlette internals
            'fastapi', 'fastapi.responses', 'fastapi.routing', 'fastapi.middleware',
            'fastapi.staticfiles',
            # HTTP & async
            'httptools', 'websockets', 'h11', 'click',
            'multipart', 'python_multipart',
            # Imaging / numerics
            'PIL', 'PIL.Image', 'PIL.ImageOps',
            'psutil',
        ]
    ),
    excludes=[
        # Heavy ML libs — stay in .venv, used only by ML subprocesses
        'torch', 'torchvision', 'torchaudio',
        'sam2', 'sam3', 'triton',
        'rfdetr', 'ultralytics',
        # Unused large packages
        'matplotlib', 'pandas', 'sklearn', 'tensorflow', 'keras',
        'tkinter', 'PySide6', 'PyQt5', 'PyQt6', 'wx',
        'IPython', 'jupyter', 'notebook',
        'test', 'unittest',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # one-folder: binaries live in COLLECT
    name='QuickLabel',
    console=True,            # keep console so server logs are visible
    debug=False,
    strip=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='QuickLabel',
)

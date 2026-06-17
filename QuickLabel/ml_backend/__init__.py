"""
ML Backend - Standalone ML processing backend for client

This is a separate project with its own virtual environment and dependencies.
It communicates with the main client app via CLI (subprocess + JSON Lines protocol).

Services:
- SAM segmentation (long-running process, stdin/stdout)
- RF-DETR model training (single-shot process, stdout streaming)

This module is NOT compiled into the Nuitka .exe - it ships alongside it
and runs in its own Python interpreter with its own venv.
"""

__version__ = "1.0.0"


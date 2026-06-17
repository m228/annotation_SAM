"""Compatibility helpers for running SAM3 in VisoLabel's local backend."""
from __future__ import annotations

import importlib.util
import sys
import types
from typing import Callable


def _edt_opencv(data):
    """CPU fallback for SAM3's CUDA/Triton EDT helper."""
    import cv2
    import numpy as np
    import torch

    if data.dim() != 3:
        raise AssertionError("edt_triton expects a tensor with shape (B, H, W)")

    device = data.device
    masks = data.detach().to("cpu").numpy()
    output = np.empty(masks.shape, dtype=np.float32)
    for index in range(masks.shape[0]):
        output[index] = cv2.distanceTransform(
            masks[index].astype(np.uint8),
            cv2.DIST_L2,
            0,
        )

    result = torch.from_numpy(output)
    if getattr(device, "type", "cpu") != "cpu":
        result = result.to(device=device)
    return result


def install_sam3_edt_fallback(log_func: Callable[[str], None] | None = None) -> bool:
    """Install a minimal ``sam3.model.edt`` module when Triton is unavailable.

    The SAM3 package imports its Triton-only EDT helper while importing
    ``sam3.model_builder``. On macOS CPU installs Triton is not available, and
    the Triton implementation would not be usable anyway because it asserts a
    CUDA tensor. Providing this module lets SAM3 import successfully while still
    preserving the EDT behavior through OpenCV if that tracker utility is used.
    """
    if importlib.util.find_spec("triton") is not None:
        return False

    existing = sys.modules.get("sam3.model.edt")
    if existing is not None:
        return bool(getattr(existing, "_visolabel_fallback", False))

    module = types.ModuleType("sam3.model.edt")
    module.__doc__ = "VisoLabel OpenCV fallback for SAM3 EDT when Triton is unavailable."
    module._visolabel_fallback = True
    module.edt_triton = _edt_opencv
    sys.modules["sam3.model.edt"] = module
    if log_func is not None:
        log_func("Installed SAM3 OpenCV EDT fallback because Triton is unavailable.")
    return True

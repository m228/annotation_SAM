"""Out-of-process bridge to the SAM 2 / SAM 3 backend.

SAM inference runs in a SEPARATE subprocess (ml_backend's JSON-lines SAM
service) so a native CUDA / torch crash or out-of-memory abort can never take
down the QuickLabel web server: the child is killed, restarted on the next
request, and the call returns a clean error. One JSON request line goes to the
child's stdin; JSON response line(s) come back on its stdout (progress lines are
forwarded to the caller). stderr is inherited so model logs reach the console.

Only one model fits on an 8 GB GPU, so switching between interactive (SAM 2) and
auto (SAM 3) work recycles the subprocess to free VRAM cleanly.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading

from .config import QUICKLABEL_DIR, ensure_ml_backend_importable


class SamRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: "subprocess.Popen | None" = None
        self._mode: "str | None" = None       # "interactive" | "auto"
        self._device_cache: "str | None" = None
        atexit.register(self._kill)

    # ── subprocess lifecycle ─────────────────────────────────────
    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
            self._mode = None

    def _start(self) -> None:
        ensure_ml_backend_importable()  # populates ML_BACKEND_SAM*_PATH + alloc conf
        env = os.environ.copy()
        env["PYTHONPATH"] = str(QUICKLABEL_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "ml_backend", "sam"],
            cwd=str(QUICKLABEL_DIR),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env=env, text=True, encoding="utf-8", bufsize=1,
        )

    def _ensure(self, mode: str) -> None:
        if self._mode is not None and self._mode != mode:
            self._kill()                # recycle to free GPU memory on switch
        if not self._alive():
            self._start()
        self._mode = mode

    # ── request / response ───────────────────────────────────────
    def _request(self, req: dict, mode: str, progress_callback=None) -> dict:
        with self._lock:
            try:
                self._ensure(mode)
                self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError, AttributeError):
                self._kill()
                return {"status": "error", "message": "SAM-процесс недоступен. Повторите."}

            while True:
                try:
                    line = self._proc.stdout.readline()
                except Exception:
                    line = ""
                if not line:
                    # EOF → the child crashed (often CUDA OOM). Recover cleanly.
                    self._kill()
                    return {"status": "error",
                            "message": "SAM-процесс аварийно завершился (вероятно, нехватка "
                                       "видеопамяти). Закройте приложения, использующие GPU, и повторите."}
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if resp.get("status") == "progress":
                    if progress_callback:
                        progress_callback(resp.get("step", ""), resp.get("message", ""))
                    continue
                return resp

    # ── public API (same signatures the server already uses) ─────
    def set_image(self, image_path: str, image_id: str,
                  expected_width: int = 0, expected_height: int = 0) -> dict:
        return self._request({
            "cmd": "set_image", "image_path": image_path, "image_id": image_id,
            "expected_width": expected_width, "expected_height": expected_height,
        }, "interactive")

    def predict_points(self, points: list[dict], image_id: str) -> dict:
        return self._request({
            "cmd": "predict_points", "points": points, "image_id": image_id,
            "multimask": True, "decode_mask": False,
        }, "interactive")

    def predict_box(self, box: dict, image_id: str) -> dict:
        return self._request({
            "cmd": "predict_box", "box": box, "image_id": image_id,
            "multimask": True, "decode_mask": False,
        }, "interactive")

    def auto_segment(self, image_path: str, text_prompt: str,
                     confidence: float = 0.5, image_id: str = "",
                     progress_callback=None) -> dict:
        return self._request({
            "cmd": "auto_segment", "image_path": image_path, "image_id": image_id,
            "text_prompt": text_prompt, "confidence_threshold": confidence,
            "decode_mask": False,
        }, "auto", progress_callback=progress_callback)

    def health(self) -> dict:
        """Lightweight CUDA probe in-process (import + query never crash; only
        model inference does, and that is isolated in the subprocess)."""
        if self._device_cache:
            return {"status": "ok", "type": "health",
                    "model_loaded": self._alive(), "device": self._device_cache}
        try:
            ensure_ml_backend_importable()
            import torch
            if torch.cuda.is_available():
                self._device_cache = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device_cache = "mps"
            else:
                self._device_cache = "cpu"
            return {"status": "ok", "type": "health",
                    "model_loaded": self._alive(), "device": self._device_cache}
        except Exception as exc:  # torch not importable
            return {"status": "error", "message": str(exc)}


# Single shared runtime for the whole server process.
runtime = SamRuntime()

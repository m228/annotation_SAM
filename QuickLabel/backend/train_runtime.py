"""Out-of-process bridge to the RF-DETR / YOLO training services.

Training runs in a SEPARATE single-shot subprocess
(``python -m ml_backend train|train-yolo --config <json>``) so a native
CUDA/torch crash or OOM during training can never take down the QuickLabel web
server — exactly like ``sam_runtime.py`` does for inference.

The dataset is built first with the existing exporters (COCO for RF-DETR, YOLO
for Ultralytics) into ``<project>/_train/<run_id>/dataset``. The subprocess then
streams JSON-lines progress on **stdout** (parsed into a single live ``status``
dict + per-epoch ``history``) and human logs on **stderr** (kept as a ring
buffer = the "live terminal"). Stopping writes a ``.stop`` sentinel the trainer
polls. Only one training run is allowed at a time (one GPU).
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional

from .config import QUICKLABEL_DIR, BUNDLE_DIR, find_python_executable, ensure_ml_backend_importable
from .store import ProjectStore
from . import coco_export, yolo_export

LOG_LINES = 300
# Fields carried inside a protocol message that should NOT be merged verbatim
# into the public status dict.
_INTERNAL_FIELDS = {"type", "epoch_record", "job_id"}
_ACTIVE = {"preparing", "building_dataset", "training", "evaluating"}


class TrainManager:
    """Owns at most one active training run for the whole server process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._slug: Optional[str] = None
        self._run_id: Optional[str] = None
        self._output_dir: Optional[Path] = None
        self._t0: float = 0.0
        self._status: dict = {"status": "idle"}
        self._history: list[dict] = []
        self._log: "deque[str]" = deque(maxlen=LOG_LINES)
        self._registered = False
        self._stop_requested = False
        self._t_out: Optional[threading.Thread] = None
        self._t_err: Optional[threading.Thread] = None
        # Kill the training subprocess if the server exits, so a run in progress
        # doesn't orphan and keep holding RAM/GPU (like sam_runtime does for SAM).
        atexit.register(self._kill_proc)

    # ── helpers ──────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _kill_proc(self) -> None:
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    def _append_log(self, line: str) -> None:
        line = line.rstrip("\r\n")
        if line:
            self._log.append(line)

    # ── dependency / device check ────────────────────────────────
    def check_deps(self) -> dict:
        """Report which training frameworks are importable + the torch device.
        Uses importlib.util.find_spec (no heavy import side effects)."""
        import importlib.util as iu
        ensure_ml_backend_importable()
        out = {"rfdetr": False, "ultralytics": False, "torch": False,
               "device": "cpu", "cuda": False}
        try:
            out["rfdetr"] = iu.find_spec("rfdetr") is not None
        except Exception:
            pass
        try:
            out["ultralytics"] = iu.find_spec("ultralytics") is not None
        except Exception:
            pass
        try:
            import torch
            out["torch"] = True
            if torch.cuda.is_available():
                out["cuda"] = True
                out["device"] = "cuda"
                try:
                    out["device_name"] = torch.cuda.get_device_name(0)
                except Exception:
                    pass
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                out["device"] = "mps"
        except Exception as exc:
            out["torch_error"] = str(exc)
        return out

    # ── start a run ──────────────────────────────────────────────
    def start(self, slug: str, cfg: dict) -> dict:
        with self._lock:
            if self.is_running():
                raise RuntimeError("Обучение уже выполняется. Остановите текущий запуск.")

            store = ProjectStore(slug)
            if not store.exists():
                raise RuntimeError(f"Проект '{slug}' не найден")
            data = store.load()

            framework = (cfg.get("framework") or "rfdetr").lower()
            task_type = cfg.get("task_type", "object_detection")
            if framework not in ("rfdetr", "yolo"):
                raise RuntimeError(f"Неизвестный фреймворк: {framework}")

            run_id = uuid.uuid4().hex[:12]
            output_dir = store.run_dir(run_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset_dir = output_dir / "dataset"

            # If anything below fails before the subprocess starts, don't leave
            # a half-built dataset folder behind.
            try:
                self._prepare_and_spawn(framework, task_type, data, dataset_dir,
                                        output_dir, run_id, cfg, store, slug)
            except Exception:
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
                raise
            return {"run_id": run_id, "dataset": self._status["dataset"]}

    def _prepare_and_spawn(self, framework, task_type, data, dataset_dir,
                           output_dir, run_id, cfg, store, slug) -> None:
            # 1) Build the dataset with the existing exporters.
            counts = self._build_dataset(framework, task_type, data, dataset_dir, cfg)
            if counts.get("train", 0) <= 0:
                raise RuntimeError("В обучающей выборке нет размеченных изображений. "
                                   "Разметьте кадры или включите предложения.")
            if counts.get("val", 0) <= 0:
                raise RuntimeError("Валидационная выборка пуста. Уменьшите долю валидации "
                                   "или добавьте размеченные изображения.")

            # 2) Write the trainer config.
            sorted_classes = sorted(data["classes"], key=lambda c: c["id"])
            classes = [c["name"] for c in sorted_classes]
            class_colors = [c.get("color", "#4f8cff") for c in sorted_classes]
            train_cfg = {
                "job_id": run_id,
                "model_name": cfg.get("model_name", "RF-DETR-S"),
                "task_type": task_type,
                "epochs": int(cfg.get("epochs", 50)),
                "batch_size": int(cfg.get("batch_size", 4)),
                "learning_rate": float(cfg.get("learning_rate", 1e-4)),
                "patience": int(cfg.get("patience", 0) or 0),
                "warmup_epochs": float(cfg.get("warmup_epochs", 0) or 0),
                "weight_decay": float(cfg.get("weight_decay", 0) or 0),
                "image_size": int(cfg.get("image_size", 0)) or None,
                "use_gpu": bool(cfg.get("use_gpu", True)),
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "classes": classes,
            }
            config_path = output_dir / "train_config.json"
            config_path.write_text(json.dumps(train_cfg, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

            # 3) Spawn the trainer subprocess.
            ensure_ml_backend_importable()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(BUNDLE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"
            # Force UTF-8 stdio in the child so Russian/log text isn't mangled
            # (Windows defaults stdout/stderr pipes to cp1251 → mojibake here).
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            python = find_python_executable()
            command = "train-yolo" if framework == "yolo" else "train"
            self._proc = subprocess.Popen(
                [python, "-m", "ml_backend", command, "--config", str(config_path)],
                cwd=str(QUICKLABEL_DIR),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, text=True, encoding="utf-8", errors="replace", bufsize=1,
            )

            # 4) Reset live state.
            self._slug, self._run_id, self._output_dir = slug, run_id, output_dir
            self._t0 = time.time()
            self._registered = False
            self._stop_requested = False
            self._history = []
            self._log = deque(maxlen=LOG_LINES)
            self._status = {
                "status": "preparing",
                "framework": framework,
                "model_name": train_cfg["model_name"],
                "task": task_type,
                "run_id": run_id,
                "total_epochs": train_cfg["epochs"],
                "image_size": train_cfg["image_size"],
                "classes": classes,
                "class_colors": class_colors,
                "dataset": {"train": counts.get("train", 0), "val": counts.get("val", 0),
                            "test": counts.get("test", 0), "classes": counts.get("classes", 0),
                            "instances": counts.get("instances", 0),
                            "tiles": counts.get("tiles", 0)},
                "message": "Сборка датасета завершена, запуск обучения…",
            }

            self._t_out = threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True)
            self._t_err = threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True)
            self._t_out.start()
            self._t_err.start()
            threading.Thread(target=self._wait, args=(self._proc, slug, run_id), daemon=True).start()

    def _build_dataset(self, framework: str, task_type: str, data: dict,
                       dataset_dir: Path, cfg: dict) -> dict:
        val_split = float(cfg.get("val_split", 0.1))
        test_split = float(cfg.get("test_split", 0.0))
        augment = bool(cfg.get("augment", False))
        angles = cfg.get("angles") or []
        aug = dict(flip_h=bool(cfg.get("flip_h", False)),
                   brightness=bool(cfg.get("brightness", False)),
                   grayscale=bool(cfg.get("grayscale", False)))
        # Any augmentation toggle implies the augmentation pass should run.
        if any(aug.values()):
            augment = True
        # Tiling: slice big frames into smaller windows so tiny objects survive
        # downscaling (a separate train-only pass, see export_common.iter_samples).
        tile = dict(tile=bool(cfg.get("tile", False)),
                    tile_size=int(cfg.get("tile_size", 640) or 640),
                    tile_overlap=float(cfg.get("tile_overlap", 0.2)),
                    tile_max_images=int(cfg.get("tile_max_images", 0) or 0),
                    tile_empty_ratio=float(cfg.get("tile_empty_ratio", 0.15)))
        include_suggested = bool(cfg.get("include_suggested", False))
        if framework == "yolo":
            fmt = "segment" if task_type == "instance_segmentation" else "detect"
            return yolo_export.export_project(
                data, dataset_dir, fmt=fmt, val_split=val_split, test_split=test_split,
                augment=augment, angles=angles, include_suggested=include_suggested,
                **aug, **tile)
        counts = coco_export.export_project(
            data, dataset_dir, val_split=val_split, test_split=test_split,
            augment=augment, angles=angles, include_suggested=include_suggested,
            project_name=data.get("name", "dataset"), **aug, **tile)
        # RF-DETR's loader requires train/ + valid/ + test/ (Roboflow layout) and
        # runs a final test-set evaluation. If no test split was requested, mirror
        # valid/ into test/ so neither the loader nor the eval hit a missing/empty set.
        test_dir = dataset_dir / "test"
        if not test_dir.exists():
            import shutil
            shutil.copytree(dataset_dir / "valid", test_dir)
            counts["test"] = counts.get("val", 0)
        return counts

    # ── reader threads ───────────────────────────────────────────
    def _read_stdout(self, proc: subprocess.Popen) -> None:
        """Parse JSON-lines protocol messages into the live status dict."""
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Not protocol JSON — treat as a log line.
                self._append_log(line)
                continue
            self._handle_message(msg)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            self._append_log(line)

    def _handle_message(self, msg: dict) -> None:
        with self._lock:
            mtype = msg.get("type")
            if mtype == "error":
                self._status["status"] = "error"
                self._status["error"] = msg.get("message", "Ошибка обучения")
                self._status["message"] = msg.get("message", "Ошибка обучения")
                self._register_model()
                return

            # Merge all metric/progress fields (so partial messages keep prior values).
            is_epoch = bool(msg.get("epoch_record"))
            for k, v in msg.items():
                if k in _INTERNAL_FIELDS:
                    continue
                self._status[k] = v

            if mtype == "result":
                # Final message: model_path + terminal status arrive here.
                self._status["model_path"] = msg.get("model_path", "")
                self._status["status"] = msg.get("status", self._status.get("status"))
                self._register_model()

            if is_epoch:
                self._append_history(msg)

    def _append_history(self, msg: dict) -> None:
        ep = int(msg.get("current_epoch", len(self._history) + 1))
        point = {
            "epoch": ep,
            "loss": _num(msg.get("loss")),
            "map_50": _num(msg.get("map_50")),
            "map_50_95": _num(msg.get("map_50_95")),
            "lr": _num(msg.get("learning_rate")),
        }
        # Replace if we already recorded this epoch, else append.
        for i, h in enumerate(self._history):
            if h.get("epoch") == ep:
                self._history[i] = point
                return
        self._history.append(point)

    def _wait(self, proc: subprocess.Popen, slug: str, run_id: str) -> None:
        proc.wait()
        # Drain stdout/stderr fully before deciding the outcome, so a terminal
        # protocol message (e.g. a fast "package not installed" error) is parsed
        # before we fall back to a generic exit-code message.
        for t in (self._t_out, self._t_err):
            if t is not None:
                t.join(timeout=3)
        with self._lock:
            if self._run_id != run_id:
                return                          # superseded by a newer run
            st = self._status.get("status")
            if st in _ACTIVE:
                # Process exited without a terminal protocol message.
                code = proc.returncode
                if self._stop_requested:
                    # We asked it to stop (graceful .stop or a hard kill).
                    self._status["status"] = "stopped"
                    self._status["message"] = "Обучение остановлено пользователем"
                elif code == 0:
                    self._status["status"] = "completed"
                    self._status.setdefault("message", "Обучение завершено")
                else:
                    self._status["status"] = "error"
                    self._status.setdefault("error", f"Процесс обучения завершился с кодом {code}")
                    self._status.setdefault("message", self._status.get("error"))
            self._register_model()

    def _register_model(self) -> None:
        """Persist a trained-model record once the run reaches a terminal state."""
        if self._registered or not self._slug or not self._run_id:
            return
        st = self._status.get("status")
        if st not in ("completed", "stopped", "error"):
            return
        self._registered = True
        try:
            store = ProjectStore(self._slug)
            elapsed = round(time.time() - self._t0, 1) if self._t0 else None
            store.add_trained_model({
                "run_id": self._run_id,
                "framework": self._status.get("framework"),
                "model_name": self._status.get("model_name"),
                "task": self._status.get("task"),
                "status": st,
                "epochs": self._status.get("epochs_reached") or self._status.get("total_epochs"),
                "total_epochs": self._status.get("total_epochs"),
                "best_map_50": self._status.get("best_map_50"),
                "map_50_95": self._status.get("map_50_95"),
                "model_path": self._status.get("model_path", ""),
                "image_size": self._status.get("image_size"),
                "classes": self._status.get("classes"),
                "class_colors": self._status.get("class_colors"),
                "dataset": self._status.get("dataset"),
                # Per-model dashboard snapshot, so clicking this model later can
                # re-render its own metrics/chart/log (not just the last run).
                "history": list(self._history),
                "metrics": {
                    "loss": self._status.get("loss"),
                    "map_50_95": self._status.get("map_50_95"),
                    "best_map_50": self._status.get("best_map_50"),
                    "precision": self._status.get("precision"),
                    "recall": self._status.get("recall"),
                    "learning_rate": self._status.get("learning_rate"),
                    "throughput": self._status.get("throughput"),
                    "elapsed_seconds": elapsed,
                },
                "log": list(self._log),
                "message": self._status.get("message"),
                "error": self._status.get("error"),
            })
        except Exception as exc:
            self._append_log(f"[train_runtime] could not register model: {exc}")

    # ── status / stop ────────────────────────────────────────────
    def status(self) -> dict:
        with self._lock:
            out = dict(self._status)
            out["history"] = list(self._history)
            out["log"] = list(self._log)
            out["running"] = self.is_running()
            if out.get("status") in _ACTIVE and self._t0:
                out["elapsed_seconds"] = round(time.time() - self._t0, 1)
            return out

    def stop(self) -> bool:
        with self._lock:
            if not self.is_running() or not self._output_dir:
                return False
            self._stop_requested = True
            try:
                (self._output_dir / ".stop").write_text("stop", encoding="utf-8")
            except Exception:
                pass
            self._status["message"] = ("Остановка… (если эпоха длинная, процесс будет "
                                        "принудительно завершён через ~10 с)")
            proc, run_id = self._proc, self._run_id
        # Watchdog OUTSIDE the lock: give the trainer a few seconds to stop
        # gracefully at an epoch boundary (saves a checkpoint if it can), then
        # force-kill so Stop is always responsive even mid-epoch.
        threading.Thread(target=self._force_kill_after, args=(proc, run_id, 5),
                         daemon=True).start()
        return True

    def _force_kill_after(self, proc: subprocess.Popen, run_id: str, grace: float) -> None:
        end = time.time() + grace
        while time.time() < end:
            if proc.poll() is not None:
                return                          # stopped on its own
            time.sleep(0.5)
        if self._run_id != run_id or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except Exception:
            pass
        # Last resort if terminate is ignored.
        for _ in range(6):
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        try:
            proc.kill()
        except Exception:
            pass


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# Single shared manager for the whole server process.
manager = TrainManager()

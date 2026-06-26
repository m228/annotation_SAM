"""
YOLO Training Service — single-shot process for Ultralytics YOLO training.

QuickLabel's web server launches this as a subprocess
(``python -m ml_backend train-yolo --config <json>``) and reads the same
JSON-lines progress protocol RF-DETR uses (see ``training_service.py`` and
``protocol.py``), so a single dashboard renders both frameworks.

Config JSON keys: model_name (e.g. "YOLO11n"|"YOLO11s"|"YOLO11m"),
task_type ("object_detection"|"instance_segmentation"), epochs, batch_size,
learning_rate, image_size, use_gpu, dataset_dir (folder with data.yaml),
output_dir, job_id.

Stop: create ``{output_dir}/.stop`` — checked every epoch/iteration.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import uuid
from typing import Any, Dict, Optional

from .protocol import (
    training_progress_message,
    training_result_message,
    training_error_message,
    write_json_line,
    log,
)

FRAMEWORK = "yolo"

# Quiet Ultralytics' own logging / tqdm bars BEFORE importing it: the parent
# reads stderr line-by-line, and tqdm's carriage-return updates (no newline)
# would otherwise stall the reader / flood the pipe.
os.environ.setdefault("YOLO_VERBOSE", "False")

_protocol_stdout = sys.stdout
_log_file = None


def _init_file_logger(output_dir: str) -> None:
    global _log_file
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "training.log")
        _log_file = open(path, "w", buffering=1, encoding="utf-8")
        _log_to_file(f"=== YOLO training log started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        log(f"Training log file: {path}")
    except Exception as e:
        log(f"WARNING: could not create log file: {e}")


def _log_to_file(msg: str) -> None:
    if _log_file and not _log_file.closed:
        try:
            _log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            _log_file.flush()
        except Exception:
            pass


def _log_both(msg: str) -> None:
    log(msg)
    _log_to_file(msg)


def _emit_progress(**kwargs) -> None:
    kwargs.setdefault("framework", FRAMEWORK)
    write_json_line(training_progress_message(**kwargs), stream=_protocol_stdout)


def _emit_result(**kwargs) -> None:
    kwargs.setdefault("framework", FRAMEWORK)
    write_json_line(training_result_message(**kwargs), stream=_protocol_stdout)


def _emit_error(message: str, job_id: str = "") -> None:
    msg = training_error_message(message, job_id=job_id)
    msg["framework"] = FRAMEWORK
    write_json_line(msg, stream=_protocol_stdout)


def _check_stop_requested(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, ".stop"))


def _yolo_weights(model_name: str, is_seg: bool) -> str:
    """Map a QuickLabel model name ("YOLO11n", "YOLO26m", …) to an Ultralytics
    checkpoint stem ("yolo11n.pt", "yolo26m-seg.pt", …)."""
    import re
    m = re.match(r"yolo(\d+)([nsmlx])", str(model_name or "").lower().replace("-", ""))
    ver, size = (m.group(1), m.group(2)) if m else ("11", "n")
    return f"yolo{ver}{size}-seg.pt" if is_seg else f"yolo{ver}{size}.pt"


def _data_yaml(dataset_dir: str) -> Optional[str]:
    cand = os.path.join(dataset_dir, "data.yaml")
    return cand if os.path.isfile(cand) else None


def run_yolo_training(config_path: str) -> None:
    """Run an Ultralytics YOLO training job from a config JSON file."""
    global _START_TIME
    _START_TIME = time.time()
    # Send any stray library prints to stderr; JSON protocol uses _protocol_stdout.
    sys.stdout = sys.stderr

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        _emit_error(f"Failed to load config: {e}")
        sys.exit(1)

    job_id = config.get("job_id", str(uuid.uuid4()))
    model_name = config.get("model_name", "YOLO11n")
    task_type = config.get("task_type", "object_detection")
    # `or <default>` guards against explicit null/0 from the caller ("use default").
    epochs = int(config.get("epochs") or 50)
    batch_size = int(config.get("batch_size") or 16)
    learning_rate = float(config.get("learning_rate") or 0.01)
    image_size = int(config.get("image_size") or 640)
    patience = int(config.get("patience") or 0)
    warmup_epochs = float(config.get("warmup_epochs") or 0)
    weight_decay = float(config.get("weight_decay") or 0)
    use_gpu = config.get("use_gpu", True)
    dataset_dir = config.get("dataset_dir", "")
    output_dir = config.get("output_dir", "training_output")

    _init_file_logger(output_dir)
    _log_both(f"YOLO job {job_id}: model={model_name}, task={task_type}, epochs={epochs}, "
              f"batch={batch_size}, imgsz={image_size}, dataset={dataset_dir}")

    _emit_progress(status="preparing", message="Подготовка обучения…",
                   total_epochs=epochs, job_id=job_id)

    data_yaml = _data_yaml(dataset_dir)
    if not data_yaml:
        _emit_error(f"data.yaml not found in {dataset_dir}", job_id=job_id)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    stop_file = os.path.join(output_dir, ".stop")
    if os.path.exists(stop_file):
        os.remove(stop_file)

    try:
        from ultralytics import YOLO
    except ImportError:
        _emit_error("Пакет ultralytics не установлен. Запустите setup.ps1 "
                    "(или: pip install ultralytics).", job_id=job_id)
        sys.exit(1)

    device = 0
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = 0
                _log_both(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                _log_both("GPU requested but not available, using CPU")
        except ImportError:
            device = "cpu"
    else:
        device = "cpu"

    is_seg = task_type == "instance_segmentation"
    weights = _yolo_weights(model_name, is_seg)
    _emit_progress(status="preparing", message=f"Загрузка модели {weights}…",
                   total_epochs=epochs, job_id=job_id)

    try:
        _run_yolo(YOLO, weights, data_yaml, epochs, batch_size, learning_rate,
                  image_size, device, output_dir, is_seg, job_id, patience,
                  warmup_epochs, weight_decay)
    except Exception as e:
        _log_both(f"Training failed: {traceback.format_exc()}")
        if _is_oom(e):
            _emit_error(
                f"Недостаточно видеопамяти GPU для «{model_name}» при batch={batch_size}, "
                f"image size={image_size}. Уменьшите Batch size (например 4, затем 2), "
                f"снизьте Image size или выберите модель меньше (n/s/m вместо l/x).",
                job_id=job_id)
        else:
            _emit_error(f"Training failed: {e}", job_id=job_id)
        sys.exit(1)


def _is_oom(exc: Exception) -> bool:
    return "OutOfMemoryError" in type(exc).__name__ or "out of memory" in str(exc).lower()


def _metric(metrics: dict, *keys: str) -> Optional[float]:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                continue
    return None


def _run_yolo(YOLO, weights, data_yaml, epochs, batch_size, learning_rate,
              image_size, device, output_dir, is_seg, job_id, patience=0,
              warmup_epochs=0.0, weight_decay=0.0) -> None:
    best_metrics = {"best_map_50": 0.0, "best_map_50_95": 0.0, "best_accuracy": 0.0}
    state = {"n_train": 0, "epoch_t0": None, "batch_seen": 0, "batches_per_epoch": 0,
             "stopped": False}

    model = YOLO(weights)

    def _request_stop(trainer):
        state["stopped"] = True
        # Ultralytics checks `trainer.stop` at epoch boundaries (and broadcasts
        # it under DDP); set it so training halts gracefully after this epoch.
        try:
            trainer.stop = True
        except Exception:
            pass
        try:
            if getattr(trainer, "stopper", None) is not None:
                trainer.stopper.possible_stop = True
        except Exception:
            pass

    def on_train_start(trainer):
        try:
            state["n_train"] = len(trainer.train_loader.dataset)
        except Exception:
            state["n_train"] = 0
        try:
            state["batches_per_epoch"] = len(trainer.train_loader)
        except Exception:
            state["batches_per_epoch"] = 0
        _emit_progress(status="training", message="Старт обучения…",
                       total_epochs=epochs, total_iterations=state["batches_per_epoch"],
                       job_id=job_id)

    def on_train_epoch_start(trainer):
        state["epoch_t0"] = time.time()
        state["batch_seen"] = 0

    def on_train_batch_end(trainer):
        state["batch_seen"] += 1
        bpe = state["batches_per_epoch"] or 1
        if state["batch_seen"] % 10 != 0 and state["batch_seen"] != 1:
            return
        display_epoch = int(getattr(trainer, "epoch", 0)) + 1
        frac = state["batch_seen"] / bpe
        pct = ((display_epoch - 1 + frac) / epochs * 100.0) if epochs > 0 else 0.0
        _emit_progress(status="training",
                       message=f"Эпоха {display_epoch}/{epochs} — итерация {state['batch_seen']}/{bpe}",
                       current_epoch=display_epoch, total_epochs=epochs,
                       current_iteration=state["batch_seen"], total_iterations=bpe,
                       percentage=pct, job_id=job_id)
        if _check_stop_requested(output_dir):
            _request_stop(trainer)

    def on_fit_epoch_end(trainer):
        display_epoch = int(getattr(trainer, "epoch", 0)) + 1
        metrics = dict(getattr(trainer, "metrics", {}) or {})

        # Prefer mask metrics for segmentation, fall back to box metrics.
        if is_seg:
            map_50 = _metric(metrics, "metrics/mAP50(M)", "metrics/mAP50(B)")
            map_50_95 = _metric(metrics, "metrics/mAP50-95(M)", "metrics/mAP50-95(B)")
            precision = _metric(metrics, "metrics/precision(M)", "metrics/precision(B)")
            recall = _metric(metrics, "metrics/recall(M)", "metrics/recall(B)")
        else:
            map_50 = _metric(metrics, "metrics/mAP50(B)")
            map_50_95 = _metric(metrics, "metrics/mAP50-95(B)")
            precision = _metric(metrics, "metrics/precision(B)")
            recall = _metric(metrics, "metrics/recall(B)")

        # Ultralytics reports mAP/precision/recall in 0..1 → percentages.
        map_50 = map_50 * 100.0 if map_50 is not None else None
        map_50_95 = map_50_95 * 100.0 if map_50_95 is not None else None
        precision = precision * 100.0 if precision is not None else None
        recall = recall * 100.0 if recall is not None else None

        if map_50 is not None and map_50 > best_metrics["best_map_50"]:
            best_metrics["best_map_50"] = map_50
            best_metrics["best_accuracy"] = map_50
        if map_50_95 is not None and map_50_95 > best_metrics["best_map_50_95"]:
            best_metrics["best_map_50_95"] = map_50_95

        # Training loss: sum of the per-component train losses for this epoch.
        loss = 0.0
        try:
            items = trainer.label_loss_items(getattr(trainer, "tloss", None), prefix="train")
            loss = float(sum(v for v in items.values() if isinstance(v, (int, float))))
        except Exception:
            try:
                loss = float(getattr(trainer, "loss", 0.0))
            except Exception:
                loss = 0.0

        # Learning rate (first param group).
        lr = learning_rate
        try:
            lrs = getattr(trainer, "lr", {}) or {}
            if lrs:
                lr = float(next(iter(lrs.values())))
        except Exception:
            pass

        # Throughput (img/s) from this epoch's wall time.
        throughput = 0.0
        if state["epoch_t0"] and state["n_train"]:
            dt = time.time() - state["epoch_t0"]
            if dt > 0:
                throughput = state["n_train"] / dt

        elapsed = time.time() - _START_TIME
        eta = (elapsed / display_epoch) * (epochs - display_epoch) if display_epoch > 0 else None
        pct = (display_epoch / epochs * 100.0) if epochs > 0 else 0.0

        fields: Dict[str, Any] = {
            "status": "training",
            "message": f"Эпоха {display_epoch}/{epochs} завершена",
            "current_epoch": display_epoch, "total_epochs": epochs,
            "loss": round(loss, 4), "learning_rate": lr, "percentage": pct,
            "throughput": round(throughput, 1), "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": eta, "epoch_record": True, "job_id": job_id,
            "epochs_reached": display_epoch, **best_metrics,
        }
        if map_50 is not None:
            fields["map_50"] = round(map_50, 2)
            fields["accuracy"] = round(map_50, 2)
        if map_50_95 is not None:
            fields["map_50_95"] = round(map_50_95, 2)
        if precision is not None:
            fields["precision"] = round(precision, 2)
        if recall is not None:
            fields["recall"] = round(recall, 2)
        _emit_progress(**fields)

        if _check_stop_requested(output_dir):
            _request_stop(trainer)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    run_name = "run"
    train_kwargs = dict(
        data=data_yaml, epochs=epochs, imgsz=image_size, batch=batch_size,
        lr0=learning_rate, device=device, project=output_dir, name=run_name,
        exist_ok=True, plots=False, verbose=False,
    )
    # Early stopping. Ultralytics stops after `patience` epochs without fitness
    # improvement. patience<=0 → disable (use a value larger than the run so it
    # never triggers); >0 → user-requested patience.
    train_kwargs["patience"] = patience if patience > 0 else (epochs + 1)
    if patience > 0:
        _log_both(f"Early stopping enabled: patience={patience} epochs")
    # Optional regularization / LR warmup — only override Ultralytics defaults
    # when the user set a value (>0); otherwise let YOLO use its own defaults.
    if warmup_epochs > 0:
        train_kwargs["warmup_epochs"] = warmup_epochs
        _log_both(f"warmup_epochs={warmup_epochs}")
    if weight_decay > 0:
        train_kwargs["weight_decay"] = weight_decay
        _log_both(f"weight_decay={weight_decay}")
    # Windows DataLoader 'spawn' workers can deadlock — run in-process.
    if sys.platform == "win32":
        train_kwargs["workers"] = 0
        _log_both("Windows detected — setting workers=0")

    _log_both(f"Calling model.train({train_kwargs})")
    model.train(**train_kwargs)
    _log_both("model.train() returned")

    best_model = os.path.join(output_dir, run_name, "weights", "best.pt")
    if not os.path.isfile(best_model):
        last_model = os.path.join(output_dir, run_name, "weights", "last.pt")
        best_model = last_model if os.path.isfile(last_model) else ""

    final_metrics = {**best_metrics}
    final_accuracy = best_metrics.get("best_accuracy") or best_metrics.get("best_map_50")

    if state["stopped"]:
        _emit_progress(status="stopped", message="Обучение остановлено пользователем",
                       total_epochs=epochs, accuracy=final_accuracy, job_id=job_id,
                       **final_metrics)
        _emit_result(status="stopped", model_path=best_model, output_dir=output_dir,
                     message="Обучение остановлено пользователем", job_id=job_id,
                     **final_metrics)
        return

    _emit_progress(status="completed", message="Обучение успешно завершено",
                   current_epoch=epochs, total_epochs=epochs, percentage=100.0,
                   accuracy=final_accuracy, job_id=job_id, **final_metrics)
    _emit_result(status="completed", model_path=best_model, output_dir=output_dir,
                 message="Обучение успешно завершено", job_id=job_id, **final_metrics)


_START_TIME = time.time()  # reset at the start of run_yolo_training()

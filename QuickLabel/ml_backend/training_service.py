"""
Training Service — single-shot process for RF-DETR model training.

Launched by QuickLabel's web server as a subprocess per training job (see
``backend/train_runtime.py``). Reads config from a JSON file, streams training
progress as JSON Lines to stdout, writes human logs to stderr (the web server
shows stderr as the "live terminal").

Adapted from VisoLabel's training_service.py. QuickLabel additions:
  * every progress message carries ``framework="rfdetr"`` so the unified
    dashboard can tell the two trainers apart,
  * a ``throughput`` (images/second) estimate is emitted during training.

Usage::

    python -m ml_backend train --config /path/to/train_config.json

Config JSON keys: model_name, task_type ("object_detection"|"instance_segmentation"),
epochs, batch_size, learning_rate, image_size, use_gpu, dataset_dir, output_dir,
classes, resume_checkpoint, job_id.

Stop: create ``{output_dir}/.stop`` — the loop polls for it and stops gracefully.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

from .protocol import (
    training_progress_message,
    training_result_message,
    training_error_message,
    write_json_line,
    log,
)

FRAMEWORK = "rfdetr"

# ── Protocol stdout ─────────────────────────────────────────
# Save the *real* stdout before anything can redirect it. All JSON Lines
# protocol messages are written to this stream; sys.stdout is redirected to
# stderr inside run_training() so print() calls from rfdetr, pycocotools, tqdm,
# numpy, etc. don't pollute the JSON protocol channel on stdout.
_protocol_stdout = sys.stdout

# ── File logger ─────────────────────────────────────────────
_log_file = None


def _init_file_logger(output_dir: str) -> None:
    """Open a log file in the output directory for persistent debugging."""
    global _log_file
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "training.log")
        _log_file = open(path, "w", buffering=1, encoding="utf-8")  # line-buffered
        _log_to_file(f"=== Training log started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
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
    """Emit a training progress message (always tagged with the framework)."""
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


RFDETR_DEFAULT_RESOLUTION = 560
RFDETR_DETECTION_RESOLUTION_SPECS = {
    # The local backend maps N/S/M detection to RFDETRBase, whose backbone
    # uses patch_size=14 and num_windows=4.
    "RF-DETR-N": (56, 560),
    "RF-DETR-S": (56, 560),
    "RF-DETR-M": (56, 560),
    # RFDETRLarge uses patch_size=16 and num_windows=2.
    "RF-DETR-L": (32, 704),
    "RF-DETR-XL": (4, 700),     # rfdetr[plus]
    "RF-DETR-2XL": (4, 880),    # rfdetr[plus]
}
RFDETR_SEGMENTATION_RESOLUTION_SPECS = {
    "RF-DETR-N": (12, 312),
    "RF-DETR-S": (24, 384),
    "RF-DETR-M": (24, 432),
    "RF-DETR-L": (24, 504),
    "RF-DETR-XL": (4, 700),
    "RF-DETR-2XL": (4, 880),
}

# QuickLabel model size token -> rfdetr class name (per task). N/S/M map to
# RFDETRBase (matches the proven VisoLabel mapping); XL/2XL need rfdetr[plus].
_RFDETR_DET_CLASS = {"N": "RFDETRBase", "S": "RFDETRBase", "M": "RFDETRBase",
                     "L": "RFDETRLarge", "XL": "RFDETRXLarge", "2XL": "RFDETR2XLarge"}
_RFDETR_SEG_CLASS = {"N": "RFDETRSegNano", "S": "RFDETRSegSmall", "M": "RFDETRSegMedium",
                     "L": "RFDETRSegLarge", "XL": "RFDETRSegXLarge", "2XL": "RFDETRSeg2XLarge"}


def _canonical_rfdetr_model_name(model_name: str) -> str:
    name = str(model_name or "RF-DETR-N")
    if name.startswith("RF-DETR-Seg-"):
        return f"RF-DETR-{name.rsplit('-', 1)[-1]}"
    return name


def _rfdetr_resolution_spec(model_name: str, task_type: str) -> tuple[int, int]:
    canonical_name = _canonical_rfdetr_model_name(model_name)
    is_segmentation = task_type == "instance_segmentation" or str(model_name or "").startswith("RF-DETR-Seg-")
    specs = RFDETR_SEGMENTATION_RESOLUTION_SPECS if is_segmentation else RFDETR_DETECTION_RESOLUTION_SPECS
    return specs.get(canonical_name, (56, RFDETR_DEFAULT_RESOLUTION))


def _normalize_rfdetr_resolution(
    image_size: int,
    model_name: str = "RF-DETR-N",
    task_type: str = "object_detection",
) -> int:
    """Normalize RF-DETR resolution to the selected backbone's divisor."""
    divisor, default_resolution = _rfdetr_resolution_spec(model_name, task_type)
    try:
        value = int(image_size)
    except (TypeError, ValueError):
        value = default_resolution
    value = max(96, value)
    normalized = int(round(value / divisor) * divisor)
    return max(96, normalized)


# ── Matcher NaN Guard ───────────────────────────────────────
# RF-DETR's matcher has a broken NaN guard: it uses C.max() to compute a
# replacement value, but torch.max() propagates NaN, so the guard is a no-op
# when NaN is present. We monkey-patch the matcher's forward with a robust
# version that replaces NaN/Inf with a large finite constant instead.
def _patch_matcher_nan_guard():
    try:
        import torch
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        from rfdetr.models.matcher import HungarianMatcher

        _original_forward = HungarianMatcher.forward

        @torch.no_grad()
        def _patched_forward(self, outputs, targets, group_detr=1):
            try:
                return _original_forward(self, outputs, targets, group_detr=group_detr)
            except ValueError as e:
                if "invalid numeric entries" not in str(e):
                    raise
                _log_both(f"[MatcherPatch] Caught '{e}', applying robust NaN guard")

                bs = outputs["pred_logits"].shape[0]
                num_queries = outputs["pred_logits"].shape[1]

                out_prob = outputs["pred_logits"].sigmoid()
                out_bbox = outputs["pred_boxes"]

                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])

                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, :, tgt_ids] - neg_cost_class[:, :, tgt_ids]

                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

                from rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox.flatten(0, 1)),
                    box_cxcywh_to_xyxy(tgt_bbox),
                ).view(bs, num_queries, -1)

                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                C = C.float().cpu()

                bad_mask = C.isinf() | C.isnan()
                if bad_mask.any():
                    finite_vals = C[~bad_mask]
                    if finite_vals.numel() > 0:
                        replacement = float(finite_vals.max().item()) * 2.0
                    else:
                        replacement = 1e6
                    C[bad_mask] = replacement
                    _log_both(f"[MatcherPatch] Replaced {bad_mask.sum().item()} NaN/Inf entries "
                              f"with {replacement:.1f}")

                sizes = [len(v["boxes"]) for v in targets]
                indices = []
                g_num_queries = num_queries // group_detr
                C_list = C.split(g_num_queries, dim=1)
                for g_i in range(group_detr):
                    C_g = C_list[g_i]
                    indices_g = [linear_sum_assignment(c[i].numpy()) for i, c in enumerate(C_g.split(sizes, -1))]
                    if g_i == 0:
                        indices = indices_g
                    else:
                        indices = [
                            (
                                np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                                np.concatenate([indice1[1], indice2[1]]),
                            )
                            for indice1, indice2 in zip(indices, indices_g)
                        ]
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        HungarianMatcher.forward = _patched_forward
        _log_both("[MatcherPatch] Successfully patched HungarianMatcher with robust NaN guard")
        return True
    except Exception as e:
        _log_both(f"[MatcherPatch] Could not patch matcher: {e}")
        return False


def _sanitize_coco_annotations(dataset_dir: str) -> None:
    """Actively remove degenerate annotations from COCO JSON files."""
    MIN_BBOX_DIM = 4.0
    MAX_ASPECT = 20.0
    MIN_AREA = MIN_BBOX_DIM ** 2

    for split in ("train", "valid", "test"):
        ann_path = os.path.join(dataset_dir, split, "_annotations.coco.json")
        if not os.path.isfile(ann_path):
            continue
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
        except Exception:
            continue

        image_map = {img["id"]: img for img in coco.get("images", [])}
        original_count = len(coco.get("annotations", []))
        clean_annotations = []
        removed = 0

        for ann in coco.get("annotations", []):
            bbox = ann.get("bbox", [])
            area = ann.get("area", 0)
            img_info = image_map.get(ann.get("image_id"), {})
            iw = float(img_info.get("width", 0) or 0)
            ih = float(img_info.get("height", 0) or 0)

            if not isinstance(bbox, list) or len(bbox) != 4:
                removed += 1
                continue
            try:
                x, y, w, h = [float(v) for v in bbox]
            except (TypeError, ValueError):
                removed += 1
                continue
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                removed += 1
                continue
            if w < MIN_BBOX_DIM or h < MIN_BBOX_DIM:
                removed += 1
                continue
            if not math.isfinite(area) or area < MIN_AREA:
                removed += 1
                continue
            aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))
            if aspect > MAX_ASPECT:
                removed += 1
                continue
            if iw > 0 and ih > 0:
                if x < -1 or y < -1 or x + w > iw + 1 or y + h > ih + 1:
                    removed += 1
                    continue

            seg = ann.get("segmentation", [])
            seg_ok = True
            if isinstance(seg, list):
                for poly in seg:
                    if isinstance(poly, list):
                        for v in poly:
                            try:
                                if not math.isfinite(float(v)):
                                    seg_ok = False
                                    break
                            except (TypeError, ValueError):
                                seg_ok = False
                                break
                    if not seg_ok:
                        break
            if not seg_ok:
                removed += 1
                continue

            clean_annotations.append(ann)

        if removed > 0:
            coco["annotations"] = clean_annotations
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(coco, f, indent=2)
            _log_both(f"[Sanitize] {split}/: removed {removed}/{original_count} degenerate annotations")
        else:
            _log_both(f"[Sanitize] {split}/: all {original_count} annotations OK")


def run_training(config_path: str) -> None:
    """Run a full RF-DETR training job from a config JSON file."""
    # Redirect sys.stdout → stderr so library prints don't pollute the JSON
    # protocol on stdout (protocol messages use _protocol_stdout, saved above).
    sys.stdout = sys.stderr

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        _emit_error(f"Failed to load config: {e}")
        sys.exit(1)

    job_id = config.get("job_id", str(uuid.uuid4()))
    model_name = config.get("model_name", "RF-DETR-N")
    task_type = config.get("task_type", "object_detection")
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 4)
    learning_rate = config.get("learning_rate", 1e-4)
    patience = int(config.get("patience") or 0)
    warmup_epochs = float(config.get("warmup_epochs") or 0)
    weight_decay = float(config.get("weight_decay") or 0)
    image_size = _normalize_rfdetr_resolution(
        config.get("image_size", None), model_name=model_name, task_type=task_type,
    )
    use_gpu = config.get("use_gpu", True)
    dataset_dir = config.get("dataset_dir", "")
    output_dir = config.get("output_dir", "training_output")
    classes = config.get("classes", [])
    resume_checkpoint = config.get("resume_checkpoint", "")

    _init_file_logger(output_dir)
    _log_both(f"Training job {job_id}: model={model_name}, task={task_type}, epochs={epochs}, "
              f"batch_size={batch_size}, image_size={image_size}, dataset={dataset_dir}")
    _log_to_file(f"Full config: {json.dumps(config, indent=2)}")

    _emit_progress(status="preparing", message="Подготовка обучения…",
                   total_epochs=epochs, job_id=job_id)

    if not dataset_dir or not os.path.isdir(dataset_dir):
        _emit_error(f"Dataset directory not found: {dataset_dir}", job_id=job_id)
        sys.exit(1)

    train_dir = os.path.join(dataset_dir, "train")
    train_ann = os.path.join(train_dir, "_annotations.coco.json")
    if not os.path.isdir(train_dir):
        _emit_error(f"train/ directory not found in {dataset_dir}", job_id=job_id)
        sys.exit(1)
    if not os.path.isfile(train_ann):
        _emit_error(f"train/_annotations.coco.json not found in {dataset_dir}", job_id=job_id)
        sys.exit(1)

    try:
        _validate_coco_dataset(dataset_dir, job_id)
    except ValueError as e:
        _emit_error(str(e), job_id=job_id)
        sys.exit(1)
    _sanitize_coco_annotations(dataset_dir)

    os.makedirs(output_dir, exist_ok=True)
    stop_file = os.path.join(output_dir, ".stop")
    if os.path.exists(stop_file):
        os.remove(stop_file)

    device = "cpu"
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                _log_both(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                _log_both("Using Apple MPS")
            else:
                _log_both("GPU requested but not available, falling back to CPU")
        except ImportError:
            _log_both("PyTorch not available, using CPU")

    _emit_progress(status="preparing", message=f"Устройство: {device}",
                   total_epochs=epochs, job_id=job_id)

    try:
        _run_rfdetr_training(
            model_name=model_name, task_type=task_type, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate, image_size=image_size,
            device=device, dataset_dir=dataset_dir, output_dir=output_dir,
            classes=classes, job_id=job_id, resume_checkpoint=resume_checkpoint,
            patience=patience, warmup_epochs=warmup_epochs, weight_decay=weight_decay,
        )
    except Exception as e:
        _log_both(f"Training failed: {traceback.format_exc()}")
        low = str(e).lower()
        ename = type(e).__name__
        if "outofmemoryerror" in ename.lower() or "out of memory" in low:
            _emit_error(
                f"Недостаточно видеопамяти GPU для «{model_name}» при batch={batch_size}, "
                f"image size={image_size}. Уменьшите Batch size (например 2–4), снизьте "
                f"Image size или выберите модель меньше (N/S вместо M/L).", job_id=job_id)
        elif "memoryerror" in ename.lower() or "memoryerror" in low:
            _emit_error(
                "Недостаточно оперативной памяти (ОЗУ) при загрузке данных. Закройте другие "
                "программы, уменьшите Image size/Batch size или число изображений за раз.",
                job_id=job_id)
        elif "device-side assert" in low or "cuda error" in low:
            _emit_error(
                "Сбой CUDA при обучении сегментации (device-side assert). На очень плотных "
                "сценах (сотни мелких объектов на кадре) RF-DETR неустойчив — он рассчитан "
                "примерно на 100 объектов на изображении. Для большого числа мелких объектов "
                "используйте YOLO-seg (n/s/m) + SAHI при распознавании, либо уменьшите batch size.",
                job_id=job_id)
        else:
            _emit_error(f"Training failed: {e}", job_id=job_id)
        sys.exit(1)


def _validate_coco_dataset(dataset_dir: str, job_id: str) -> None:
    """Validate COCO dataset structure; log diagnostics, raise on fatal issues."""
    fatal_issues = []
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(dataset_dir, split)
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.isfile(ann_path):
            _log_both(f"[Validate] {split}/ — no annotation file (ok for test)")
            continue
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
        except Exception as e:
            _log_both(f"[Validate] {split}/ — ERROR reading JSON: {e}")
            continue

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])
        _log_both(f"[Validate] {split}/: {len(images)} images, "
                  f"{len(annotations)} annotations, {len(categories)} categories")

        cat_ids = {c["id"] for c in categories}
        cat_names = {c["id"]: c["name"] for c in categories}

        img_ids = set()
        image_map = {}
        for img in images:
            img_ids.add(img["id"])
            image_map[img["id"]] = img

        cat_counts = {}
        for ann in annotations:
            cat_id = ann.get("category_id", -1)
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
        for cid in sorted(cat_counts):
            cname = cat_names.get(cid, f"?{cid}")
            _log_both(f"[Validate] {split}/: '{cname}' (id={cid}): {cat_counts[cid]} annotations")

        if split == "train" and len(images) == 0:
            fatal_issues.append("train split has 0 images — the training dataset is empty.")
        if split == "valid" and len(images) == 0:
            fatal_issues.append("valid split has 0 images — RF-DETR cannot evaluate an empty "
                                "validation set. Lower the validation fraction or add images.")

    if fatal_issues:
        raise ValueError("Dataset validation failed: " + " | ".join(fatal_issues[:5])
                         + ". See training.log for details.")
    _log_both("[Validate] Dataset validation complete")


def _run_rfdetr_training(
    model_name: str, task_type: str, epochs: int, batch_size: int,
    learning_rate: float, image_size: int, device: str, dataset_dir: str,
    output_dir: str, classes: List[str], job_id: str, resume_checkpoint: str = "",
    patience: int = 0, warmup_epochs: float = 0.0, weight_decay: float = 0.0,
) -> None:
    best_metrics: Dict[str, float] = {
        "best_accuracy": 0.0, "best_map_50": 0.0, "best_map_50_95": 0.0,
    }
    last_metrics: Dict[str, float] = {}
    # Manual early stopping (rfdetr has no built-in patience like Ultralytics):
    # track the best monitored metric and how many epochs since it improved.
    early = {"best": -1.0, "since": 0, "stopped": False}

    def _extract_ap_metrics(log_stats: dict) -> Dict[str, float]:
        metrics_out: Dict[str, float] = {}
        eval_key = None
        for key in ("test_coco_eval_bbox", "ema_test_coco_eval_bbox",
                    "test_coco_eval_masks", "ema_test_coco_eval_masks"):
            if key in log_stats:
                eval_key = key
                break
        if eval_key is None:
            return metrics_out
        coco_eval = log_stats.get(eval_key)
        if not isinstance(coco_eval, (list, tuple)):
            return metrics_out
        if len(coco_eval) > 0:
            metrics_out["map_50_95"] = float(coco_eval[0]) * 100.0
        if len(coco_eval) > 1:
            metrics_out["map_50"] = float(coco_eval[1]) * 100.0
        if len(coco_eval) > 2:
            metrics_out["map_75"] = float(coco_eval[2]) * 100.0
        return metrics_out

    _emit_progress(status="preparing", message="Загрузка RF-DETR…",
                   total_epochs=epochs, job_id=job_id)

    try:
        import rfdetr as _rf
    except ImportError:
        _emit_error("Пакет rfdetr не установлен. Запустите setup.ps1 "
                    "(или: pip install rfdetr==1.5.2).", job_id=job_id)
        sys.exit(1)

    is_segmentation = task_type == "instance_segmentation"
    size = _canonical_rfdetr_model_name(model_name).rsplit("-", 1)[-1].upper()
    class_map = _RFDETR_SEG_CLASS if is_segmentation else _RFDETR_DET_CLASS
    cls_name = class_map.get(size, "RFDETRSegNano" if is_segmentation else "RFDETRBase")
    model_kwargs = {"resolution": image_size}

    _log_both(f"Initializing model: {model_name} (class={cls_name}, kwargs={model_kwargs})")
    _emit_progress(status="preparing", message=f"Инициализация {model_name}…",
                   total_epochs=epochs, job_id=job_id)

    try:
        ModelCls = getattr(_rf, cls_name)
    except (ImportError, AttributeError) as e:
        # XL/2XL detection models live behind the rfdetr[plus] extension.
        _emit_error(f"Модель {model_name} требует платное расширение rfdetr[plus] "
                    f"(pip install rfdetr[plus]) и много видеопамяти. "
                    f"Для крупной модели без plus используйте RF-DETR-L. Детали: {e}",
                    job_id=job_id)
        sys.exit(1)
    model = ModelCls(**model_kwargs)

    _patch_matcher_nan_guard()

    _emit_progress(status="training", message="Старт обучения…",
                   total_epochs=epochs, job_id=job_id)

    start_time = time.time()

    class BatchProgressCallback:
        """Emit per-iteration progress + throughput so the UI keeps moving during
        long epochs.

        rfdetr passes a GLOBAL cumulative ``step`` (start_steps + batch index) and
        no per-epoch total, so we track the step at each epoch's start to derive a
        per-epoch ``local`` step, and learn steps-per-epoch once the epoch rolls
        over (used for the iteration bar + intra-epoch percentage from epoch 2 on)."""

        def __init__(self):
            self.epoch = 0
            self.epoch_start_step: Optional[int] = None
            self.steps_per_epoch = 0
            self.seen = 0
            self._last_emit_t: Optional[float] = None
            self._last_emit_seen = 0
            self.throughput = 0.0

        @property
        def current_epoch(self) -> int:        # kept for the eval hook below
            return self.epoch

        def __call__(self, info: dict):
            step = int(info.get("step", 0))
            epoch = int(info.get("epoch", self.epoch))
            self.seen += 1
            if self.epoch_start_step is None:
                self.epoch_start_step = step
            if epoch != self.epoch:
                # Epoch rolled over → we now know how many steps the prior epoch had.
                self.steps_per_epoch = max(self.steps_per_epoch, step - self.epoch_start_step)
                self.epoch_start_step = step
                self.epoch = epoch
            local = max(0, step - self.epoch_start_step)
            display_epoch = epoch + 1

            # Throttle: every batch on small datasets, ~2% of an epoch on big ones.
            throttle = max(1, int(self.steps_per_epoch * 0.02)) if self.steps_per_epoch else 1
            if local % throttle != 0:
                return

            now = time.time()
            if self._last_emit_t is not None:
                d_iter = self.seen - self._last_emit_seen
                dt = now - self._last_emit_t
                if dt > 0 and d_iter > 0:
                    self.throughput = batch_size * d_iter / dt
            self._last_emit_t = now
            self._last_emit_seen = self.seen

            spe = self.steps_per_epoch
            # During epoch 1 (spe unknown) let the fraction creep toward ~1 so the
            # headline % isn't frozen at 0; from epoch 2 it's exact.
            frac = (local / spe) if spe else (local / (local + 1.0))
            frac = min(1.0, frac)
            pct = ((epoch + frac) / epochs * 100.0) if epochs > 0 else 0.0
            _emit_progress(
                status="training",
                message=f"Эпоха {display_epoch}/{epochs} — итерация {local + 1}"
                        + (f"/{spe}" if spe else " (первая эпоха, метрики появятся после неё)"),
                current_epoch=display_epoch, total_epochs=epochs,
                current_iteration=local + 1, total_iterations=spe,
                percentage=round(pct, 1),
                throughput=round(self.throughput, 1),
                elapsed_seconds=round(now - start_time, 1),
                job_id=job_id,
            )
            if _check_stop_requested(output_dir):
                _log_both("Stop requested, requesting early stop")
                try:
                    model.model.request_early_stop()
                except Exception:
                    pass

    class EpochEndCallback:
        """Adapter: receives rfdetr *log_stats* dict from on_fit_epoch_end."""

        def __init__(self):
            self.current_epoch = 0

        def __call__(self, log_stats: dict):
            self.current_epoch = log_stats.get("epoch", self.current_epoch)
            display_epoch = self.current_epoch + 1
            _log_to_file(f"epoch_end: epoch={display_epoch} keys={list(log_stats.keys())}")

            loss = log_stats.get("train_loss", 0.0)
            lr = log_stats.get("train_lr", learning_rate)

            metric_values = _extract_ap_metrics(log_stats)
            if metric_values:
                last_metrics.update(metric_values)
                map_50 = metric_values.get("map_50")
                map_50_95 = metric_values.get("map_50_95")
                if map_50 is not None and map_50 > best_metrics["best_map_50"]:
                    best_metrics["best_map_50"] = map_50
                    best_metrics["best_accuracy"] = map_50
                if map_50_95 is not None and map_50_95 > best_metrics["best_map_50_95"]:
                    best_metrics["best_map_50_95"] = map_50_95
            accuracy = metric_values.get("map_50")

            # Early stopping: monitor mAP@50:95 (fallback mAP@50). Stop when it
            # hasn't improved by >0.01% for `patience` consecutive epochs.
            if patience and patience > 0:
                monitor = metric_values.get("map_50_95")
                if monitor is None:
                    monitor = metric_values.get("map_50")
                if monitor is not None:
                    if monitor > early["best"] + 1e-4:
                        early["best"] = monitor
                        early["since"] = 0
                    else:
                        early["since"] += 1
                        _log_to_file(f"early-stop: no improvement {early['since']}/{patience} "
                                     f"(monitor={monitor:.3f}, best={early['best']:.3f})")
                        if early["since"] >= patience and not early["stopped"]:
                            early["stopped"] = True
                            _log_both(f"Early stopping: метрика не улучшалась {patience} эпох — "
                                      f"останавливаем обучение")
                            try:
                                model.model.request_early_stop()
                            except Exception:
                                pass

            elapsed = time.time() - start_time
            eta = (elapsed / display_epoch) * (epochs - display_epoch) if display_epoch > 0 else None
            pct = (display_epoch / epochs * 100) if epochs > 0 else 0

            _emit_progress(
                status="training", message=f"Эпоха {display_epoch}/{epochs} завершена",
                current_epoch=display_epoch, total_epochs=epochs,
                loss=float(loss), learning_rate=float(lr), percentage=pct,
                accuracy=accuracy, eta_seconds=eta, elapsed_seconds=round(elapsed, 1),
                throughput=round(batch_cb.throughput, 1),
                epoch_record=True,        # tells train_runtime to append a history point
                job_id=job_id, epochs_reached=display_epoch,
                **metric_values, **best_metrics,
            )

    batch_cb = BatchProgressCallback()
    epoch_cb = EpochEndCallback()
    model.callbacks["on_train_batch_start"].append(batch_cb)
    model.callbacks["on_fit_epoch_end"].append(epoch_cb)

    # RF-DETR has no public evaluation-batch callback. Wrap its imported
    # evaluate() symbol so the validation phase keeps the UI moving.
    try:
        import rfdetr.main as rfdetr_main
        original_evaluate = rfdetr_main.evaluate
        trainer_model = model

        def evaluate_with_progress(eval_model, criterion, postprocess, data_loader,
                                   base_ds, device, args=None, header="Eval"):
            total = len(data_loader) if hasattr(data_loader, "__len__") else 0
            display_epoch = max(1, batch_cb.current_epoch + 1)
            phase = str(header or "Eval")

            class ProgressDataLoader:
                def __init__(self, wrapped):
                    self._wrapped = wrapped

                def __len__(self):
                    return total

                def __iter__(self):
                    for idx, batch in enumerate(self._wrapped, start=1):
                        if _check_stop_requested(output_dir):
                            try:
                                trainer_model.model.request_early_stop()
                            except Exception:
                                pass
                        eval_fraction = (idx / total) if total else 0.0
                        pct = (((display_epoch - 1) + 0.8 + (0.2 * eval_fraction)) / epochs * 100.0) if epochs > 0 else 0.0
                        _emit_progress(
                            status="evaluating",
                            message=f"{phase}: эпоха {display_epoch}/{epochs} — батч {idx}/{total or '?'}",
                            current_epoch=display_epoch, total_epochs=epochs,
                            current_iteration=idx, total_iterations=total,
                            percentage=pct, job_id=job_id, epochs_reached=display_epoch,
                            **best_metrics,
                        )
                        yield batch

            _emit_progress(status="evaluating",
                           message=f"{phase}: эпоха {display_epoch}/{epochs}",
                           current_epoch=display_epoch, total_epochs=epochs,
                           current_iteration=0, total_iterations=total,
                           percentage=(((display_epoch - 1) + 0.8) / epochs * 100.0) if epochs > 0 else 0.0,
                           job_id=job_id, epochs_reached=display_epoch, **best_metrics)
            result = original_evaluate(eval_model, criterion, postprocess,
                                       ProgressDataLoader(data_loader), base_ds, device,
                                       args=args, header=header)
            return result

        rfdetr_main.evaluate = evaluate_with_progress
    except Exception as hook_exc:
        _log_both(f"WARNING: could not install evaluation progress hook: {hook_exc}")

    extra_train_kwargs: Dict[str, Any] = {"tensorboard": True, "progress_bar": False}
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        extra_train_kwargs["resume"] = resume_checkpoint
        _log_both(f"Resuming training from checkpoint: {resume_checkpoint}")
    elif resume_checkpoint:
        _log_both(f"WARNING: resume checkpoint not found: {resume_checkpoint}")
    if classes:
        extra_train_kwargs["class_names"] = classes
    # Windows DataLoader 'spawn' workers frequently deadlock — run in-process.
    # RF-DETR's own DataLoader uses `num_workers` (default 2), NOT `workers`; on
    # Windows those worker processes deadlock and/or exhaust host RAM on large
    # images. Force in-process loading (num_workers=0) — slower but reliable.
    extra_train_kwargs["workers"] = 0
    extra_train_kwargs["num_workers"] = 0
    if sys.platform == "win32":
        _log_both("Windows detected - DataLoader workers/num_workers=0 (avoid deadlock/RAM OOM)")

    # Optional regularization / LR warmup. RF-DETR's train() merges kwargs into a
    # config object whose accepted names vary by version, so these are passed
    # "best effort": if a name is rejected at call time we retry without them
    # rather than aborting the whole run.
    optional_kwargs: Dict[str, Any] = {}
    if weight_decay > 0:
        optional_kwargs["weight_decay"] = weight_decay
    if warmup_epochs > 0:
        optional_kwargs["warmup_epochs"] = warmup_epochs
    if optional_kwargs:
        _log_both(f"Optional train kwargs (best effort): {optional_kwargs}")

    _log_both(f"Calling model.train(dataset_dir={dataset_dir}, epochs={epochs}, "
              f"batch_size={batch_size}, lr={learning_rate}, device={device})")
    # The very first batch is slow (CUDA kernel compilation / cudnn benchmark);
    # tell the UI so the early 0% period doesn't look frozen.
    _emit_progress(status="training", total_epochs=epochs, current_epoch=1, percentage=0.0,
                   message="Прогрев: компиляция CUDA и первый батч (может занять до ~минуты)…",
                   job_id=job_id)

    def _do_train(extra: Dict[str, Any]) -> None:
        model.train(dataset_dir=dataset_dir, epochs=epochs, batch_size=batch_size,
                    lr=learning_rate, output_dir=output_dir, device=device, **extra)

    def _rejected_optional(exc: Exception) -> bool:
        """A binding/validation error caused by an unsupported optional kwarg —
        raised at call time, before any real training happens, so a retry is safe."""
        msg = str(exc).lower()
        named = any(k in msg for k in optional_kwargs)
        phrase = ("unexpected keyword" in msg or "validation error" in msg
                  or "extra" in msg or "not permitted" in msg or "unexpected" in msg)
        return bool(optional_kwargs) and named and phrase

    try:
        try:
            _do_train({**extra_train_kwargs, **optional_kwargs})
        except (TypeError, ValueError) as opt_exc:
            if _rejected_optional(opt_exc):
                _log_both(f"RF-DETR не принял доп. параметры {list(optional_kwargs)}: "
                          f"{opt_exc}. Повтор обучения без них.")
                _do_train(extra_train_kwargs)
            else:
                raise
        _log_both("model.train() returned successfully")
    except KeyboardInterrupt:
        _log_both("Training interrupted (KeyboardInterrupt)")
        _finish_stopped(output_dir, epoch_cb, epochs, last_metrics, best_metrics, job_id)
        return
    except Exception as train_exc:
        _log_both(f"model.train() raised: {train_exc}\n{traceback.format_exc()}")
        raise

    epochs_reached = max(0, epoch_cb.current_epoch + 1)
    stop_requested = _check_stop_requested(output_dir)
    final_metrics: Dict[str, Any] = {**last_metrics, **best_metrics, "epochs_reached": epochs_reached}
    final_accuracy = final_metrics.get("best_accuracy") or final_metrics.get("map_50")
    best_model = _find_best_model(output_dir)

    if stop_requested and epochs_reached < epochs:
        _emit_progress(status="stopped", message="Обучение остановлено пользователем",
                       current_epoch=epochs_reached, total_epochs=epochs,
                       percentage=(epochs_reached / epochs * 100.0) if epochs > 0 else 0.0,
                       accuracy=final_accuracy, job_id=job_id, **final_metrics)
        _emit_result(status="stopped", model_path=best_model or "", output_dir=output_dir,
                     message="Обучение остановлено пользователем", job_id=job_id, **final_metrics)
        return

    done_msg = "Обучение успешно завершено"
    if early["stopped"]:
        done_msg = (f"Ранняя остановка: метрика не улучшалась {patience} эпох подряд "
                    f"(остановлено на эпохе {epochs_reached} из {epochs})")
    _emit_progress(status="completed", message=done_msg,
                   current_epoch=epochs_reached or epochs, total_epochs=epochs,
                   percentage=100.0, accuracy=final_accuracy, job_id=job_id, **final_metrics)
    _emit_result(status="completed", model_path=best_model or "", output_dir=output_dir,
                 message=done_msg, job_id=job_id, **final_metrics)


def _finish_stopped(output_dir, epoch_cb, epochs, last_metrics, best_metrics, job_id):
    epochs_reached = max(0, epoch_cb.current_epoch + 1)
    final_metrics: Dict[str, Any] = {**last_metrics, **best_metrics, "epochs_reached": epochs_reached}
    stop_accuracy = final_metrics.get("best_accuracy") or final_metrics.get("map_50")
    _emit_progress(status="stopped", message="Обучение остановлено пользователем",
                   current_epoch=epochs_reached, total_epochs=epochs,
                   accuracy=stop_accuracy, job_id=job_id, **final_metrics)
    _emit_result(status="stopped", model_path=_find_best_model(output_dir) or "",
                 output_dir=output_dir, message="Обучение остановлено пользователем",
                 job_id=job_id, **final_metrics)


def _find_best_model(output_dir: str) -> Optional[str]:
    """Find the best model checkpoint in the output directory."""
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            if f == "checkpoint_best_total.pth":
                return os.path.join(root, f)
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".pth"):
                return os.path.join(root, f)
    return None

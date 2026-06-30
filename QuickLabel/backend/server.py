"""QuickLabel FastAPI server.

Serves the single-page web UI and a small JSON API for projects, images,
classes, SAM 2 / SAM 3 assisted annotation, propagation and YOLO export.
Run with ``python -m backend.server`` using the ml_backend venv (see run.ps1).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import (WEB_DIR, HOST, PORT, QUICKLABEL_DIR, IMAGE_EXTENSIONS,
                     ensure_ml_backend_importable)
from .store import ProjectStore
from .sam_runtime import runtime
from .jobs import manager
from .train_runtime import manager as train_manager
from . import yolo_export
from . import coco_export
from . import dataset_import

app = FastAPI(title="QuickLabel")


def _store(slug: str) -> ProjectStore:
    s = ProjectStore(slug)
    if not s.exists():
        raise HTTPException(404, f"Project '{slug}' not found")
    return s


# ── request models ───────────────────────────────────────────────
class CreateProject(BaseModel):
    name: str


class ClassIn(BaseModel):
    name: str
    color: Optional[str] = None


class ClassUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None


class ImportFolder(BaseModel):
    folder: str


class ImportDataset(BaseModel):
    folder: str
    format: str = "auto"            # "auto" | "yolo" | "coco"
    copy: bool = True               # copy images into the project


class AnnotationsIn(BaseModel):
    annotations: list[dict]


class StaticRoisIn(BaseModel):
    rois: list[dict]


class Point(BaseModel):
    x: float
    y: float
    is_positive: bool = True


class PointsReq(BaseModel):
    image_id: str
    points: list[Point]


class BoxReq(BaseModel):
    image_id: str
    box: dict


class AutoReq(BaseModel):
    image_id: str
    class_id: int
    text_prompt: str
    confidence: float = 0.5
    sahi: bool = False              # slice the image into tiles + run extra passes
    slice_size: int = 512
    overlap: float = 0.2
    iou: float = 0.45               # dedup threshold when merging passes
    drop_edge: bool = True          # discard tile detections cut by a tile seam
    edge_margin: float = 2.0        # px slack for the seam test


class PropagateReq(BaseModel):
    from_image_id: str
    class_id: int
    text_prompt: str
    confidence: float = 0.5
    scope: str = "all"          # "all" | "following"


class SettingsIn(BaseModel):
    patch: dict


class ExportReq(BaseModel):
    out_dir: str
    target: str = "yolo"        # "yolo" | "coco" (RF-DETR)
    fmt: str = "detect"          # YOLO only: "detect" | "segment"
    val_split: float = 0.1
    augment: bool = False
    angles: list[float] = []
    include_suggested: bool = False


class TrainReq(BaseModel):
    framework: str = "rfdetr"               # "rfdetr" | "yolo"
    model_name: str = "RF-DETR-S"
    task_type: str = "object_detection"     # "object_detection" | "instance_segmentation"
    epochs: int = 50
    batch_size: int = 4
    image_size: int = 0                     # 0 → trainer default for the model
    learning_rate: float = 1e-4
    patience: int = 0                       # early stop: 0 = off; N = stop after N
                                            # epochs with no validation improvement
    warmup_epochs: float = 0.0              # 0 → framework default
    weight_decay: float = 0.0               # 0 → framework default
    use_gpu: bool = True
    val_split: float = 0.1
    test_split: float = 0.0
    augment: bool = False
    angles: list[float] = []
    flip_h: bool = False
    brightness: bool = False
    grayscale: bool = False
    include_suggested: bool = False


class PredictReq(BaseModel):
    run_id: str
    image_id: Optional[str] = None
    image_path: Optional[str] = None
    confidence: float = 0.25
    sahi: bool = False
    slice_size: int = 640
    overlap: float = 0.2
    iou: float = 0.45
    drop_edge: bool = True
    limit: int = 12                 # validation gallery: max images to preview


# ── projects ─────────────────────────────────────────────────────
@app.get("/api/projects")
def list_projects():
    return ProjectStore.list_projects()


@app.post("/api/projects")
def create_project(body: CreateProject):
    store = ProjectStore.create(body.name)
    return store.load()


@app.delete("/api/projects/{slug}")
def delete_project(slug: str):
    _store(slug).delete_project()
    return {"ok": True}


@app.get("/api/projects/{slug}")
def get_project(slug: str):
    return _store(slug).load()


# ── classes ──────────────────────────────────────────────────────
@app.post("/api/projects/{slug}/classes")
def add_class(slug: str, body: ClassIn):
    return _store(slug).add_class(body.name, body.color)


@app.patch("/api/projects/{slug}/classes/{cid}")
def update_class(slug: str, cid: int, body: ClassUpdate):
    try:
        return _store(slug).update_class(cid, body.name, body.color)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/projects/{slug}/classes/{cid}")
def delete_class(slug: str, cid: int):
    _store(slug).delete_class(cid)
    return {"ok": True}


# ── images ───────────────────────────────────────────────────────
@app.post("/api/projects/{slug}/import_folder")
def import_folder(slug: str, body: ImportFolder):
    try:
        added = _store(slug).import_folder(body.folder)
    except FileNotFoundError:
        raise HTTPException(400, f"Folder not found: {body.folder}")
    return {"added": added}


@app.post("/api/projects/{slug}/import_dataset")
def import_dataset_endpoint(slug: str, body: ImportDataset):
    """Import an already-exported YOLO or RF-DETR/COCO dataset back as annotations."""
    store = _store(slug)
    try:
        res = dataset_import.import_dataset(store, body.folder, fmt=body.format, copy=body.copy)
    except FileNotFoundError:
        raise HTTPException(400, f"Папка не найдена: {body.folder}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return res


@app.post("/api/projects/{slug}/upload")
async def upload_images(slug: str, files: list[UploadFile] = File(...)):
    store = _store(slug)
    added = 0
    for f in files:
        raw = await f.read()
        if store.add_uploaded(f.filename, raw):
            added += 1
    return {"added": added}


@app.get("/api/projects/{slug}/image/{image_id}")
def serve_image(slug: str, image_id: str):
    img = _store(slug).get_image(image_id)
    if not img or not Path(img["path"]).is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(img["path"])


@app.delete("/api/projects/{slug}/image/{image_id}")
def delete_image(slug: str, image_id: str):
    _store(slug).delete_image(image_id)
    return {"ok": True}


# ── annotations ──────────────────────────────────────────────────
@app.put("/api/projects/{slug}/image/{image_id}/annotations")
def set_annotations(slug: str, image_id: str, body: AnnotationsIn):
    try:
        return _store(slug).set_annotations(image_id, body.annotations)
    except KeyError as e:
        raise HTTPException(404, str(e))


# ── static ROIs (apply to every frame) ───────────────────────────
@app.put("/api/projects/{slug}/static_rois")
def set_static_rois(slug: str, body: StaticRoisIn):
    return {"static_rois": _store(slug).set_static_rois(body.rois)}


# ── helpers for building suggestions ─────────────────────────────
def _sahi_tile_boxes(w: int, h: int, slice_size: int, overlap: float) -> list[tuple]:
    """Overlapping tile boxes (x0,y0,x1,y1) covering a w×h image."""
    slice_size = max(64, int(slice_size))
    step = max(1, int(slice_size * (1.0 - max(0.0, min(0.9, overlap)))))
    xs = list(range(0, max(1, w - 1), step))
    ys = list(range(0, max(1, h - 1), step))
    out = []
    for y0 in ys:
        for x0 in xs:
            x1, y1 = min(x0 + slice_size, w), min(y0 + slice_size, h)
            out.append((max(0, x1 - slice_size), max(0, y1 - slice_size), x1, y1))
    return out


def _bbox_touches_seam(bb: dict, x0: int, y0: int, x1: int, y1: int,
                       w: int, h: int, margin: float) -> bool:
    """True if a tile-local prediction bbox hugs a tile seam that is NOT the real
    image border.

    A crystal straddling a seam is sliced into ragged halves — one per tile — and
    those halves never merge back into a single object. Because the tiles overlap
    *and* there is always a full-image pass, the whole crystal is captured cleanly
    elsewhere, so the seam-cut halves can be dropped. A bbox touching the genuine
    image edge is kept (the crystal really ends there). ``bb`` is in tile coords.
    """
    if not bb:
        return False
    lx0, ly0 = bb["x"], bb["y"]
    lx1, ly1 = lx0 + bb["width"], ly0 + bb["height"]
    tw, th = x1 - x0, y1 - y0
    if x0 > 0 and lx0 <= margin:
        return True                    # cut by the left seam (tile, not image)
    if y0 > 0 and ly0 <= margin:
        return True                    # cut by the top seam
    if x1 < w and lx1 >= tw - margin:
        return True                    # cut by the right seam
    if y1 < h and ly1 >= th - margin:
        return True                    # cut by the bottom seam
    return False


def _offset_pred(pred: dict, dx: int, dy: int) -> dict:
    """Shift a SAM prediction's bbox + polygon from tile coords to full-image coords."""
    p = dict(pred)
    bb = pred.get("bbox")
    if bb:
        p["bbox"] = {"x": bb["x"] + dx, "y": bb["y"] + dy,
                     "width": bb["width"], "height": bb["height"]}
    poly = pred.get("polygon")
    if poly:
        p["polygon"] = [{"x": pt["x"] + dx, "y": pt["y"] + dy} for pt in poly]
    return p


def _dedup_predictions(preds: list[dict], iou_thr: float) -> list[dict]:
    """Greedy bbox-IoU NMS so the same object found in overlapping passes/tiles
    isn't kept twice. Keeps the highest-confidence detection of each cluster."""
    def iou(a: dict, b: dict) -> float:
        ax1, ay1 = a["x"], a["y"]; ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
        bx1, by1 = b["x"], b["y"]; bx2, by2 = bx1 + b["width"], by1 + b["height"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = a["width"] * a["height"] + b["width"] * b["height"] - inter
        return inter / union if union > 0 else 0.0

    kept: list[dict] = []
    for p in sorted(preds, key=lambda d: d.get("confidence", 0.0), reverse=True):
        bb = p.get("bbox")
        if not bb:
            continue
        if all(iou(bb, k["bbox"]) < iou_thr for k in kept):
            kept.append(p)
    return kept


def _predictions_to_anns(preds: list[dict], class_id: int, source: str,
                         status: str) -> list[dict]:
    return [
        {
            "class_id": class_id,
            "bbox": p["bbox"],
            "polygon": p.get("polygon"),
            "confidence": p.get("confidence", 0.0),
            "source": source,
            "status": status,
        }
        for p in preds
    ]


# ── SAM 2 interactive (job-based, with progress) ─────────────────
@app.post("/api/projects/{slug}/sam2/points")
def sam2_points(slug: str, body: PointsReq):
    store = _store(slug)
    img = store.get_image(body.image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    pts = [p.model_dump() for p in body.points]

    def work(job):
        job.set_progress(40, "SAM 2: кодирование изображения…")
        set_res = runtime.set_image(img["path"], body.image_id, img["width"], img["height"])
        if set_res.get("status") != "ok":
            raise RuntimeError(set_res.get("message", "set_image failed"))
        if job.is_cancelled():
            return None
        job.set_progress(80, "SAM 2: поиск маски…")
        res = runtime.predict_points(pts, body.image_id)
        if res.get("status") != "ok":
            raise RuntimeError(res.get("message", "predict failed"))
        return {"predictions": res.get("predictions", [])}

    return {"job_id": manager.submit("sam2_points", work).id}


@app.post("/api/projects/{slug}/sam2/box")
def sam2_box(slug: str, body: BoxReq):
    store = _store(slug)
    img = store.get_image(body.image_id)
    if not img:
        raise HTTPException(404, "Image not found")

    def work(job):
        job.set_progress(40, "SAM 2: кодирование изображения…")
        set_res = runtime.set_image(img["path"], body.image_id, img["width"], img["height"])
        if set_res.get("status") != "ok":
            raise RuntimeError(set_res.get("message", "set_image failed"))
        if job.is_cancelled():
            return None
        job.set_progress(80, "SAM 2: поиск маски…")
        res = runtime.predict_box(body.box, body.image_id)
        if res.get("status") != "ok":
            raise RuntimeError(res.get("message", "predict failed"))
        return {"predictions": res.get("predictions", [])}

    return {"job_id": manager.submit("sam2_box", work).id}


# ── SAM 3 auto-segmentation on current image (job-based) ─────────
_SAM3_STEP_PERCENT = {"load_model": 10, "encode_image": 45, "text_prompt": 80}


@app.post("/api/projects/{slug}/sam3/auto")
def sam3_auto(slug: str, body: AutoReq):
    store = _store(slug)
    img = store.get_image(body.image_id)
    if not img:
        raise HTTPException(404, "Image not found")

    def work(job):
        def on_step(step, message):
            job.set_progress(_SAM3_STEP_PERCENT.get(step, job.progress["percent"]),
                             f"SAM 3: {message}")
        t0 = time.perf_counter()

        if not body.sahi:
            res = runtime.auto_segment(img["path"], body.text_prompt, body.confidence,
                                       body.image_id, progress_callback=on_step)
            if res.get("status") != "ok":
                raise RuntimeError(res.get("message", "auto-segment failed"))
            preds = res.get("predictions", [])
            elapsed = time.perf_counter() - t0
            print(f"[SAM3] auto_segment '{img['filename']}': {elapsed:.1f}s, "
                  f"{len(preds)} объектов", file=sys.stderr, flush=True)
            return {"predictions": _predictions_to_anns(preds, body.class_id, "sam3", "suggested"),
                    "elapsed": round(elapsed, 1), "passes": 1, "raw": len(preds)}

        # ── SAHI: 1 full-image pass + N tile passes, results merged/deduped ──
        tiles = _sahi_tile_boxes(img["width"], img["height"], body.slice_size, body.overlap)
        total = len(tiles) + 1
        all_preds: list[dict] = []

        # Pass 1 is over the WHOLE image: this is where a seam-spanning crystal is
        # annotated as one clean object. Its predictions are never seam-dropped.
        job.set_progress(2, f"SAM 3 (SAHI): проход 1 из {total} — основное фото…")
        res = runtime.auto_segment(img["path"], body.text_prompt, body.confidence, body.image_id)
        if res.get("status") != "ok":
            raise RuntimeError(res.get("message", "auto-segment failed"))
        all_preds.extend(res.get("predictions", []))

        W, H = img["width"], img["height"]
        edge_dropped = 0
        import tempfile as _tf
        from PIL import Image
        tile_dir = Path(_tf.mkdtemp(prefix="ql_sahi_"))
        try:
            with Image.open(img["path"]) as im:
                im = im.convert("RGB")
                for i, (x0, y0, x1, y1) in enumerate(tiles, start=1):
                    if job.is_cancelled():
                        break
                    job.set_progress(int(2 + (i / total) * 96),
                                     f"SAM 3 (SAHI): проход {i + 1} из {total} — тайл {i}/{len(tiles)}…")
                    tile_path = tile_dir / f"tile_{i}.jpg"
                    im.crop((x0, y0, x1, y1)).save(tile_path, quality=90)
                    tres = runtime.auto_segment(str(tile_path), body.text_prompt,
                                                body.confidence, f"{body.image_id}_t{i}")
                    if tres.get("status") == "ok":
                        for p in tres.get("predictions", []):
                            # Drop crystals cut by a tile seam — the full-image pass
                            # (and overlapping tiles) carry the whole object instead.
                            if body.drop_edge and _bbox_touches_seam(
                                    p.get("bbox"), x0, y0, x1, y1, W, H, body.edge_margin):
                                edge_dropped += 1
                                continue
                            all_preds.append(_offset_pred(p, x0, y0))
        finally:
            import shutil as _sh
            _sh.rmtree(tile_dir, ignore_errors=True)

        raw = len(all_preds)
        merged = _dedup_predictions(all_preds, body.iou)
        elapsed = time.perf_counter() - t0
        print(f"[SAM3-SAHI] '{img['filename']}': {total} проходов, {elapsed:.1f}s, "
              f"сырых {raw} (откинуто на швах {edge_dropped}) → "
              f"после дедупликации {len(merged)} объектов",
              file=sys.stderr, flush=True)
        return {"predictions": _predictions_to_anns(merged, body.class_id, "sam3", "suggested"),
                "elapsed": round(elapsed, 1), "passes": total, "raw": raw,
                "edge_dropped": edge_dropped, "cancelled": job.is_cancelled()}

    return {"job_id": manager.submit("sam3_auto", work).id}


# ── SAM 3 propagation across images (job-based, cancellable) ──────
@app.post("/api/projects/{slug}/sam3/propagate")
def sam3_propagate(slug: str, body: PropagateReq):
    """Run the SAM 3 text prompt over many images and store the results as
    confirm/correct suggestions for the chosen class. Cancellable per image."""
    store = _store(slug)
    data = store.load()
    store.upsert_rule(body.class_id, body.text_prompt, body.confidence)

    images = data["images"]
    if body.scope == "following":
        ids = [im["id"] for im in images]
        try:
            images = images[ids.index(body.from_image_id) + 1:]
        except ValueError:
            pass

    def work(job):
        total = len(images)
        total_added = 0
        processed = 0
        t0 = time.perf_counter()
        for i, im in enumerate(images, start=1):
            if job.is_cancelled():
                break
            job.set_progress(int((i - 1) / max(1, total) * 100),
                             f"SAM 3: изображение {i} из {total}…")
            ti = time.perf_counter()
            res = runtime.auto_segment(im["path"], body.text_prompt,
                                       body.confidence, im["id"])
            dt = time.perf_counter() - ti
            processed += 1
            if res.get("status") == "ok":
                anns = _predictions_to_anns(res.get("predictions", []),
                                            body.class_id, "sam3", "suggested")
                total_added += store.add_suggestions(im["id"], anns)
                print(f"[SAM3] propagate '{im['filename']}': {dt:.1f}s, "
                      f"{len(anns)} объектов", file=sys.stderr, flush=True)
        elapsed = time.perf_counter() - t0
        avg = round(elapsed / processed, 1) if processed else 0.0
        print(f"[SAM3] propagate готово: {processed} изображений за {elapsed:.1f}s "
              f"(~{avg}s/фото)", file=sys.stderr, flush=True)
        return {"total_added": total_added, "images": total,
                "cancelled": job.is_cancelled(),
                "elapsed": round(elapsed, 1), "avg": avg, "processed": processed}

    return {"job_id": manager.submit("sam3_propagate", work).id}


# ── job status / cancel ──────────────────────────────────────────
@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    if not manager.cancel(job_id):
        raise HTTPException(404, "Job not found")
    return {"ok": True}


# ── settings & export ────────────────────────────────────────────
@app.patch("/api/projects/{slug}/settings")
def update_settings(slug: str, body: SettingsIn):
    return _store(slug).update_settings(body.patch)


@app.post("/api/projects/{slug}/export")
def export(slug: str, body: ExportReq):
    data = _store(slug).load()
    common = dict(val_split=body.val_split, augment=body.augment,
                  angles=body.angles, include_suggested=body.include_suggested)
    if body.target == "coco":
        counts = coco_export.export_project(
            data, Path(body.out_dir), project_name=data.get("name", "dataset"),
            **common,
        )
    else:
        counts = yolo_export.export_project(data, Path(body.out_dir), fmt=body.fmt, **common)
    return {"out_dir": str(Path(body.out_dir).resolve()), "target": body.target, **counts}


# ── training (local RF-DETR / YOLO) ──────────────────────────────
@app.get("/api/train/check")
def train_check():
    """Which training frameworks are installed + the torch device."""
    return train_manager.check_deps()


@app.post("/api/projects/{slug}/train")
def train_start(slug: str, body: TrainReq):
    _store(slug)  # 404 if the project is missing
    try:
        return train_manager.start(slug, body.model_dump())
    except RuntimeError as e:
        # 409: a run is already active; 400: bad dataset / config.
        code = 409 if "уже выполняется" in str(e) else 400
        raise HTTPException(code, str(e))


@app.get("/api/train/status")
def train_status():
    return train_manager.status()


@app.post("/api/train/stop")
def train_stop():
    return {"ok": train_manager.stop()}


@app.get("/api/projects/{slug}/trained_models")
def trained_models(slug: str):
    return {"models": _store(slug).list_trained_models()}


@app.delete("/api/projects/{slug}/trained_models/{run_id}")
def delete_trained_model(slug: str, run_id: str):
    _store(slug).delete_trained_model(run_id)
    return {"ok": True}


def _run_predict_subprocess(cfg: dict) -> dict:
    """Run the one-shot inference service and return its JSON result."""
    ensure_ml_backend_importable()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(QUICKLABEL_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
        cfg_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ml_backend", "predict", "--config", cfg_path],
            cwd=str(QUICKLABEL_DIR), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=900,
        )
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass
    result = None
    for line in proc.stdout.splitlines():            # last JSON line is the result
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                pass
    if result is None:
        tail = (proc.stderr or "")[-400:]
        raise HTTPException(500, f"Инференс не вернул результат. {tail}")
    if result.get("status") == "error":
        raise HTTPException(400, result.get("message", "Ошибка инференса"))
    return result


@app.post("/api/projects/{slug}/predict")
def predict_model(slug: str, body: PredictReq):
    store = _store(slug)
    rec = next((m for m in store.list_trained_models() if m.get("run_id") == body.run_id), None)
    if not rec:
        raise HTTPException(404, "Обученная модель не найдена")
    model_path = rec.get("model_path") or ""
    if not model_path or not Path(model_path).is_file():
        raise HTTPException(400, "Файл модели не найден (обучение могло не сохранить чекпойнт).")

    if body.image_id:
        img = store.get_image(body.image_id)
        if not img:
            raise HTTPException(404, "Изображение не найдено")
        image_path = img["path"]
    elif body.image_path:
        image_path = body.image_path.strip().strip('"')
    else:
        raise HTTPException(400, "Не выбрано изображение")
    if not Path(image_path).is_file():
        raise HTTPException(400, f"Файл изображения не найден: {image_path}")

    cfg = {
        "framework": rec.get("framework"), "task_type": rec.get("task"),
        "model_name": rec.get("model_name"), "model_path": model_path,
        "image_path": image_path, "image_size": rec.get("image_size"),
        "classes": rec.get("classes") or [], "class_colors": rec.get("class_colors") or [],
        "confidence": body.confidence, "sahi": body.sahi,
        "slice_size": body.slice_size, "overlap": body.overlap, "iou": body.iou,
        "drop_edge": body.drop_edge,
    }
    return _run_predict_subprocess(cfg)


def _val_images_for_run(store: ProjectStore, run_id: str) -> list[str]:
    """Validation images of a finished run (YOLO images/val or COCO valid/)."""
    base = store.run_dir(run_id) / "dataset"
    for d in (base / "images" / "val", base / "valid", base / "val"):
        if d.is_dir():
            imgs = sorted(str(p) for p in d.iterdir()
                          if p.suffix.lower() in IMAGE_EXTENSIONS)
            if imgs:
                return imgs
    return []


@app.get("/api/projects/{slug}/trained_models/{run_id}/has_val")
def trained_model_has_val(slug: str, run_id: str):
    """Whether the run still has its validation images on disk (for the UI button)."""
    return {"count": len(_val_images_for_run(_store(slug), run_id))}


@app.post("/api/projects/{slug}/trained_models/{run_id}/validate")
def validate_model(slug: str, run_id: str, body: PredictReq):
    """Run the trained model over its OWN validation set and return annotated
    previews — lets you eyeball quality after training, image by image."""
    store = _store(slug)
    rec = next((m for m in store.list_trained_models() if m.get("run_id") == run_id), None)
    if not rec:
        raise HTTPException(404, "Обученная модель не найдена")
    model_path = rec.get("model_path") or ""
    if not model_path or not Path(model_path).is_file():
        raise HTTPException(400, "Файл модели не найден (обучение могло не сохранить чекпойнт).")
    val_imgs = _val_images_for_run(store, run_id)
    if not val_imgs:
        raise HTTPException(400, "Валидационные изображения не найдены — датасет обучения "
                                 "(_train/<run>/dataset) был удалён. Переобучите модель, чтобы "
                                 "проверить её на валидации, или используйте «Проверить» на любом фото.")
    limit = max(1, min(int(body.limit or 12), 48))
    cfg = {
        "framework": rec.get("framework"), "task_type": rec.get("task"),
        "model_name": rec.get("model_name"), "model_path": model_path,
        "image_paths": val_imgs[:limit], "image_size": rec.get("image_size"),
        "classes": rec.get("classes") or [], "class_colors": rec.get("class_colors") or [],
        "confidence": body.confidence, "sahi": body.sahi,
        "slice_size": body.slice_size, "overlap": body.overlap, "iou": body.iou,
        "drop_edge": body.drop_edge,
    }
    res = _run_predict_subprocess(cfg)
    res["total_val"] = len(val_imgs)
    res["shown"] = min(limit, len(val_imgs))
    return res


@app.get("/api/health")
def health():
    return runtime.health()


# ── static web UI (mounted last so /api wins) ────────────────────
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main():
    import uvicorn
    import webbrowser
    import threading

    url = f"http://{HOST}:{PORT}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"QuickLabel running at {url}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()

"""QuickLabel FastAPI server.

Serves the single-page web UI and a small JSON API for projects, images,
classes, SAM 2 / SAM 3 assisted annotation, propagation and YOLO export.
Run with ``python -m backend.server`` using the ml_backend venv (see run.ps1).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import WEB_DIR, HOST, PORT
from .store import ProjectStore
from .sam_runtime import runtime
from .jobs import manager
from . import yolo_export
from . import coco_export

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
        res = runtime.auto_segment(img["path"], body.text_prompt, body.confidence,
                                   body.image_id, progress_callback=on_step)
        elapsed = time.perf_counter() - t0
        if res.get("status") != "ok":
            raise RuntimeError(res.get("message", "auto-segment failed"))
        preds = res.get("predictions", [])
        # Stopwatch: how long SAM 3 took on this single image.
        print(f"[SAM3] auto_segment '{img['filename']}': {elapsed:.1f}s, "
              f"{len(preds)} объектов", file=sys.stderr, flush=True)
        return {"predictions": _predictions_to_anns(preds, body.class_id, "sam3", "suggested"),
                "elapsed": round(elapsed, 1)}

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

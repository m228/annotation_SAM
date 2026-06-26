"""
Inference / test service — run a trained YOLO or RF-DETR model on ONE image and
return an annotated preview + detection counts.

Launched one-shot by the web server: ``python -m ml_backend predict --config <json>``.
Reads config JSON, writes a single JSON-line result on stdout, logs to stderr.

Optional **SAHI-style tiling** (slicing aided inference): for high-resolution
images with tiny objects (e.g. 2046×2046 with small crystals), downscaling to the
model resolution loses them. With ``sahi: true`` the image is cut into overlapping
tiles, each is run through the model at near-native scale, detections are offset
back to full-image coordinates and merged with per-class NMS. Implemented manually
(no sahi dependency) so it works identically for YOLO and RF-DETR.

Objects sliced by a tile seam are detected as ragged half-shapes in each tile. With
``drop_edge`` (default on) any detection hugging a tile seam — but not the real
image border — is discarded; the overlap guarantees the object is captured whole in
a neighbouring tile, so only the clean full-object detection survives.

Config JSON keys: framework ("yolo"|"rfdetr"), task_type, model_name, model_path,
image_path, image_size, classes (names), class_colors (hex), confidence,
sahi (bool), slice_size, overlap, iou, drop_edge (bool), edge_margin (px),
max_side (downscale guard for non-sahi).

Result JSON: {status, count, per_class:{name:n}, width, height, image_b64, sahi,
tiles, edge_dropped}.
"""
from __future__ import annotations

import base64
import json
import sys
import traceback
from typing import Any, Dict, List, Optional

from .protocol import write_json_line, log

_stdout = sys.stdout


def _result(obj: dict) -> None:
    write_json_line(obj, stream=_stdout)


def run_predict(config_path: str) -> None:
    # Library prints → stderr; the JSON result uses the saved real stdout.
    sys.stdout = sys.stderr
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        _result({"status": "error", "message": f"Failed to load config: {e}"})
        sys.exit(1)
    try:
        if cfg.get("image_paths"):
            _result({"status": "ok", **_predict_batch(cfg)})
        else:
            _result({"status": "ok", **_predict(cfg)})
    except Exception as e:
        log(traceback.format_exc())
        _result({"status": "error", "message": str(e)})
        sys.exit(1)


# ── geometry / merge helpers ─────────────────────────────────────
def _tiles(w: int, h: int, slice_size: int, overlap: float):
    """Yield (x0, y0, x1, y1) tile boxes covering the image with overlap."""
    slice_size = max(64, int(slice_size))
    step = max(1, int(slice_size * (1.0 - max(0.0, min(0.9, overlap)))))
    xs = list(range(0, max(1, w - 1), step))
    ys = list(range(0, max(1, h - 1), step))
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + slice_size, w)
            y1 = min(y0 + slice_size, h)
            x0a = max(0, x1 - slice_size)
            y0a = max(0, y1 - slice_size)
            yield x0a, y0a, x1, y1


def _touches_inner_seam(box, x0, y0, x1, y1, w, h, margin: float) -> bool:
    """True if a tile-local detection box hugs a tile seam that is NOT the real
    image border.

    When SAHI cuts the image into tiles, an object straddling a seam is sliced in
    two and each half becomes a ragged, partial mask. Because tiles overlap, that
    same object is captured *whole* inside a neighbouring tile, so the partial
    halves can be dropped safely. A box touching the genuine image edge is kept —
    there the object really does end there. ``margin`` is the slack in px.
    """
    lx0, ly0, lx1, ly1 = box           # coordinates are relative to the tile
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


def _nms_merge(dets: List[dict], iou: float) -> List[dict]:
    """Per-class NMS over full-image detections."""
    if not dets:
        return []
    import torch
    from torchvision.ops import nms
    keep: List[dict] = []
    by_cls: Dict[int, List[int]] = {}
    for i, d in enumerate(dets):
        by_cls.setdefault(d["cls"], []).append(i)
    for _cls, idxs in by_cls.items():
        boxes = torch.tensor([dets[i]["box"] for i in idxs], dtype=torch.float32)
        scores = torch.tensor([dets[i]["score"] for i in idxs], dtype=torch.float32)
        k = nms(boxes, scores, float(iou))
        keep.extend(dets[idxs[j]] for j in k.tolist())
    keep.sort(key=lambda d: d["score"], reverse=True)
    return keep


# ── model loaders → a unified predict_tile(bgr_array) -> [det] ────
def _load_predictor(cfg: dict):
    framework = cfg["framework"]
    model_path = cfg["model_path"]
    conf = float(cfg.get("confidence", 0.25))
    classes = cfg.get("classes", [])

    if framework == "yolo":
        from ultralytics import YOLO
        model = YOLO(model_path)
        names = model.names if isinstance(getattr(model, "names", None), dict) else \
            {i: n for i, n in enumerate(classes)}
        imgsz = int(cfg.get("image_size") or 640) or 640

        def predict_tile(bgr):
            res = model.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
            out = []
            boxes = res.boxes
            masks = getattr(res, "masks", None)
            n = 0 if boxes is None else len(boxes)
            for i in range(n):
                xyxy = boxes.xyxy[i].tolist()
                poly = None
                if masks is not None and getattr(masks, "xy", None) is not None and i < len(masks.xy):
                    poly = [[float(x), float(y)] for x, y in masks.xy[i].tolist()]
                out.append({"box": [float(v) for v in xyxy],
                            "score": float(boxes.conf[i]), "cls": int(boxes.cls[i]),
                            "poly": poly})
            return out
        return predict_tile, names

    if framework == "rfdetr":
        import rfdetr as _rf
        from PIL import Image
        import cv2
        from .training_service import (_RFDETR_DET_CLASS, _RFDETR_SEG_CLASS,
                                        _canonical_rfdetr_model_name,
                                        _normalize_rfdetr_resolution)
        is_seg = cfg.get("task_type") == "instance_segmentation"
        size = _canonical_rfdetr_model_name(cfg.get("model_name", "RF-DETR-N")).rsplit("-", 1)[-1].upper()
        cls_name = (_RFDETR_SEG_CLASS if is_seg else _RFDETR_DET_CLASS).get(
            size, "RFDETRSegNano" if is_seg else "RFDETRBase")
        res = _normalize_rfdetr_resolution(cfg.get("image_size"),
                                           model_name=cfg.get("model_name", "RF-DETR-N"),
                                           task_type=cfg.get("task_type", "object_detection"))
        ModelCls = getattr(_rf, cls_name)
        model = ModelCls(pretrain_weights=model_path, resolution=res,
                         num_classes=len(classes) + 1)
        names = {i: n for i, n in enumerate(classes)}

        def predict_tile(bgr):
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            d = model.predict(pil, threshold=conf)
            out = []
            xyxy = getattr(d, "xyxy", None)
            n = 0 if xyxy is None else len(xyxy)
            for i in range(n):
                poly = None
                mask = getattr(d, "mask", None)
                if mask is not None and i < len(mask):
                    poly = _mask_to_poly(mask[i])
                out.append({"box": [float(v) for v in xyxy[i]],
                            "score": float(d.confidence[i]) if d.confidence is not None else 1.0,
                            "cls": int(d.class_id[i]) if d.class_id is not None else 0,
                            "poly": poly})
            return out
        return predict_tile, names

    raise ValueError(f"Unknown framework: {framework}")


def _mask_to_poly(mask) -> Optional[list]:
    import numpy as np
    import cv2
    m = (np.asarray(mask).astype("uint8")) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(c) < 3:
        return None
    return [[float(x), float(y)] for x, y in c]


# ── main inference ───────────────────────────────────────────────
def _predict(cfg: dict) -> dict:
    import cv2

    image_path = cfg.get("image_path", "")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось открыть изображение: {image_path}")
    predict_tile, names = _load_predictor(cfg)
    return _infer_and_draw(img, predict_tile, names, cfg)


def _predict_batch(cfg: dict) -> dict:
    """Run the model over many images with a SINGLE model load (used by the
    post-training validation gallery). Returns one result per image."""
    import os
    import cv2
    predict_tile, names = _load_predictor(cfg)
    results: List[dict] = []
    for p in cfg.get("image_paths", []):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            results.append({"name": os.path.basename(p), "error": "не удалось открыть"})
            continue
        r = _infer_and_draw(img, predict_tile, names, cfg)
        r["name"] = os.path.basename(p)
        results.append(r)
    return {"results": results, "count_images": len(results)}


def _infer_and_draw(img, predict_tile, names, cfg: dict) -> dict:
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    use_sahi = bool(cfg.get("sahi", False))
    # Drop detections sliced by a tile seam (kept whole in an overlapping tile),
    # so SAHI stops producing ragged half-objects at the edges. On by default.
    drop_edge = bool(cfg.get("drop_edge", True))
    edge_margin = float(cfg.get("edge_margin", 2.0))
    all_dets: List[dict] = []
    n_tiles = 0
    n_edge_dropped = 0
    if use_sahi:
        slice_size = int(cfg.get("slice_size", 640))
        overlap = float(cfg.get("overlap", 0.2))
        for (x0, y0, x1, y1) in _tiles(w, h, slice_size, overlap):
            tile = img[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            n_tiles += 1
            for d in predict_tile(tile):
                bx = d["box"]
                if drop_edge and _touches_inner_seam(bx, x0, y0, x1, y1, w, h, edge_margin):
                    n_edge_dropped += 1
                    continue
                d["box"] = [bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0]
                if d.get("poly"):
                    d["poly"] = [[px + x0, py + y0] for px, py in d["poly"]]
                all_dets.append(d)
        all_dets = _nms_merge(all_dets, float(cfg.get("iou", 0.45)))
    else:
        n_tiles = 1
        all_dets = predict_tile(img)

    # ── draw ──
    colors = cfg.get("class_colors", []) or []
    def color_for(cls: int):
        if 0 <= cls < len(colors):
            return _hex_to_bgr(colors[cls])
        palette = [(255, 90, 60), (60, 200, 90), (60, 140, 255), (0, 200, 230), (200, 60, 230)]
        return palette[cls % len(palette)]

    per_class: Dict[str, int] = {}
    line = max(2, int(round(max(w, h) / 900)))
    font_scale = max(0.5, max(w, h) / 2200.0)
    for d in all_dets:
        cls = d["cls"]
        name = names.get(cls, f"class {cls}") if isinstance(names, dict) else \
            (names[cls] if 0 <= cls < len(names) else f"class {cls}")
        per_class[name] = per_class.get(name, 0) + 1
        col = color_for(cls)
        x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
        if d.get("poly") and len(d["poly"]) >= 3:
            pts = np.array([[int(px), int(py)] for px, py in d["poly"]], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], col)
            cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
            cv2.polylines(img, [pts], True, col, line)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), col, line)
        label = f"{name} {d['score']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, label, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (20, 20, 20), max(1, line - 1), cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
    return {"count": len(all_dets), "per_class": per_class, "width": w, "height": h,
            "image_b64": image_b64, "sahi": use_sahi, "tiles": n_tiles,
            "edge_dropped": n_edge_dropped}


def _hex_to_bgr(hx: str):
    try:
        hx = hx.lstrip("#")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return (b, g, r)
    except Exception:
        return (60, 140, 255)

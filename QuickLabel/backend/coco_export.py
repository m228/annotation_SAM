"""COCO-JSON export for RF-DETR (Roboflow-style layout, with segmentation).

Output::

    <export>/
        train/_annotations.coco.json   train/*.jpg
        valid/_annotations.coco.json   valid/*.jpg

This matches the COCO format RF-DETR (Roboflow) trains on: each split folder holds
its images plus an ``_annotations.coco.json``. Categories include a placeholder
supercategory at id 0 (Roboflow convention) with real classes from id 1, and
every annotation carries both ``bbox`` and ``segmentation`` (polygon), so the
same dataset works for RF-DETR detection and RF-DETR segmentation.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import export_common as ec


def _clip_polygon(pts: np.ndarray, w: int, h: int) -> list[float]:
    flat = []
    for px, py in pts:
        flat.append(round(float(min(max(px, 0), w - 1)), 2))
        flat.append(round(float(min(max(py, 0), h - 1)), 2))
    return flat


def _bbox_and_area(pts: np.ndarray, w: int, h: int):
    xs = np.clip(pts[:, 0], 0, w - 1)
    ys = np.clip(pts[:, 1], 0, h - 1)
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    bw, bh = x1 - x0, y1 - y0
    if bw < 1 or bh < 1:
        return None, None, None
    # Shoelace polygon area (falls back to bbox area for degenerate inputs).
    area = 0.0
    n = len(pts)
    for i in range(n):
        x_i, y_i = pts[i]
        x_j, y_j = pts[(i + 1) % n]
        area += x_i * y_j - x_j * y_i
    area = abs(area) / 2.0 or (bw * bh)
    return [round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)], round(area, 2), (x1, y1)


def export_project(data: dict, out_dir: Path, *,
                   val_split: float = 0.1, augment: bool = False,
                   angles: Optional[list[float]] = None,
                   include_suggested: bool = False, seed: int = 42,
                   project_name: str = "dataset", **_ignored) -> dict:
    out_dir = Path(out_dir)
    classes, id_map = ec.class_id_map(data)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Roboflow-style categories: placeholder supercategory at id 0, classes 1..N.
    super_name = (project_name or "objects").strip() or "objects"
    categories = [{"id": 0, "name": super_name, "supercategory": "none"}]
    for i, c in enumerate(classes):
        categories.append({"id": i + 1, "name": c["name"], "supercategory": super_name})

    splits = {
        s: {"dir": out_dir / s, "images": [], "annotations": [],
            "img_id": 0, "ann_id": 0}
        for s in ("train", "valid")
    }
    for st in splits.values():
        st["dir"].mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "instances": 0, "augmented": 0}
    for s in ec.iter_samples(data, val_split=val_split, augment=augment, angles=angles,
                             include_suggested=include_suggested, seed=seed):
        st = splits["valid" if s.split == "val" else "train"]
        file_name = f"{s.stem}{'.jpg' if s.is_aug else s.ext}"
        if not s.write_image(st["dir"] / file_name):
            continue
        image_id = st["img_id"]
        st["img_id"] += 1
        st["images"].append({
            "id": image_id, "license": 1, "file_name": file_name,
            "height": s.height, "width": s.width, "date_captured": "",
        })
        n_inst = 0
        for ann, pts in zip(s.anns, s.pts_list):
            cid = id_map.get(ann.get("class_id"))
            if cid is None:
                continue
            bbox, area, _ = _bbox_and_area(pts, s.width, s.height)
            if bbox is None:
                continue
            st["annotations"].append({
                "id": st["ann_id"], "image_id": image_id,
                "category_id": cid + 1,                 # real classes start at 1
                "bbox": bbox, "area": area,
                "segmentation": [_clip_polygon(pts, s.width, s.height)],
                "iscrowd": 0,
            })
            st["ann_id"] += 1
            n_inst += 1
        counts["train" if s.split == "train" else "val"] += 1
        counts["instances"] += n_inst
        if s.is_aug:
            counts["augmented"] += 1

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for name, st in splits.items():
        doc = {
            "info": {"description": f"QuickLabel export ({project_name})",
                     "version": "1.0", "date_created": now},
            "licenses": [{"id": 1, "name": "", "url": ""}],
            "categories": categories,
            "images": st["images"],
            "annotations": st["annotations"],
        }
        (st["dir"] / "_annotations.coco.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    counts["format"] = "coco-seg"
    counts["classes"] = len(classes)
    return counts

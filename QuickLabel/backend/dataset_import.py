"""Import a previously exported dataset (YOLO or RF-DETR / COCO) back into a project.

Use case: part of a dataset was already labelled and exported; this reads those
labels back so annotation can continue inside QuickLabel instead of starting over.
Images are copied into the project and every box/polygon becomes a **confirmed**
annotation. The format is auto-detected:

  * ``_annotations.coco.json`` anywhere under the folder → COCO (RF-DETR export).
  * ``data.yaml`` + ``labels/`` → YOLO (detection or segmentation).

All splits found (train/val(id)/test) are merged into the single project. This is
the inverse of ``yolo_export.py`` / ``coco_export.py``.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from .config import IMAGE_EXTENSIONS
from .store import ProjectStore


# ── format detection ─────────────────────────────────────────────
def detect_format(root: Path) -> str:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    if list(root.rglob("_annotations.coco.json")):
        return "coco"
    if list(root.rglob("data.yaml")) or (root / "labels").is_dir():
        return "yolo"
    raise ValueError("Не удалось определить формат датасета: нет ни "
                     "_annotations.coco.json (RF-DETR/COCO), ни data.yaml (YOLO).")


def _img_size(path: Path) -> Optional[tuple[int, int]]:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return im.size           # (w, h)
    except Exception:
        return None


# ── YOLO ─────────────────────────────────────────────────────────
def _parse_yolo_names(yaml_path: Path) -> dict[int, str]:
    """Read the ``names:`` mapping from a data.yaml without requiring PyYAML.

    Supports both the dict form (``  0: crystal``) and the list forms
    (``names: [a, b]`` or a block of ``  - a`` items)."""
    names: dict[int, str] = {}
    try:
        lines = yaml_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return names
    in_block = False
    list_idx = 0
    for raw in lines:
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("names:"):
            rest = stripped[len("names:"):].strip()
            if rest.startswith("[") or rest.startswith("{"):
                try:
                    val = ast.literal_eval(rest)
                    if isinstance(val, dict):
                        return {int(k): str(v) for k, v in val.items()}
                    if isinstance(val, (list, tuple)):
                        return {i: str(v) for i, v in enumerate(val)}
                except Exception:
                    pass
            in_block = True
            list_idx = 0
            continue
        if in_block:
            # Leave the block once a non-indented key appears.
            if not raw.startswith((" ", "\t")):
                break
            item = stripped
            if item.startswith("- "):
                names[list_idx] = item[2:].strip().strip("'\"")
                list_idx += 1
            elif ":" in item:
                k, _, v = item.partition(":")
                try:
                    names[int(k.strip())] = v.strip().strip("'\"")
                except ValueError:
                    continue
    return names


def _yolo_label_for_image(img_path: Path) -> Optional[Path]:
    """labels/<split>/<stem>.txt next to images/<split>/<stem>.<ext>."""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            cand = Path(*parts).with_suffix(".txt")
            return cand
    # Flat layout: a sibling labels/ folder.
    cand = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    return cand if cand.is_file() else img_path.with_suffix(".txt")


def _parse_yolo_line(parts: list[str], w: int, h: int) -> Optional[dict]:
    try:
        nums = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    if len(nums) == 4:                       # cx cy bw bh  (detection)
        cx, cy, bw, bh = nums
        x = (cx - bw / 2.0) * w
        y = (cy - bh / 2.0) * h
        return {"bbox": _bbox(x, y, bw * w, bh * h, w, h), "polygon": None}
    if len(nums) >= 6 and len(nums) % 2 == 0:  # x1 y1 x2 y2 …  (segmentation)
        poly = [{"x": round(nums[i] * w, 2), "y": round(nums[i + 1] * h, 2)}
                for i in range(0, len(nums), 2)]
        xs = [p["x"] for p in poly]
        ys = [p["y"] for p in poly]
        return {"bbox": _bbox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), w, h),
                "polygon": poly}
    return None


def _bbox(x: float, y: float, bw: float, bh: float, w: int, h: int) -> dict:
    x = max(0.0, min(x, w - 1))
    y = max(0.0, min(y, h - 1))
    bw = max(1.0, min(bw, w - x))
    bh = max(1.0, min(bh, h - y))
    return {"x": round(x, 2), "y": round(y, 2), "width": round(bw, 2), "height": round(bh, 2)}


def load_yolo(root: Path) -> list[dict]:
    root = Path(root)
    yaml_path = next(iter(root.rglob("data.yaml")), None)
    names = _parse_yolo_names(yaml_path) if yaml_path else {}
    samples: list[dict] = []
    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "labels" in img_path.parts:        # skip anything under labels/
            continue
        label_path = _yolo_label_for_image(img_path)
        if not label_path or not label_path.is_file():
            continue
        size = _img_size(img_path)
        if not size:
            continue
        w, h = size
        anns = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            geom = _parse_yolo_line(parts, w, h)
            if not geom:
                continue
            anns.append({"class_name": names.get(cls, f"class_{cls}"), **geom})
        samples.append({"path": str(img_path), "annotations": anns})
    return samples


# ── COCO (RF-DETR) ───────────────────────────────────────────────
def _seg_to_polygon(seg) -> Optional[list[dict]]:
    if not isinstance(seg, list) or not seg:
        return None
    poly = seg[0]
    if not isinstance(poly, list) or len(poly) < 6:
        return None
    return [{"x": round(float(poly[i]), 2), "y": round(float(poly[i + 1]), 2)}
            for i in range(0, len(poly) - 1, 2)]


def load_coco(root: Path) -> list[dict]:
    root = Path(root)
    samples: list[dict] = []
    for ann_file in sorted(root.rglob("_annotations.coco.json")):
        try:
            doc = json.loads(ann_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        img_dir = ann_file.parent
        # Real classes; skip the Roboflow placeholder supercategory (id 0/"none").
        cat_name: dict[int, str] = {}
        for c in doc.get("categories", []):
            if c.get("id") == 0 and str(c.get("supercategory", "")).lower() in ("none", ""):
                continue
            cat_name[c["id"]] = c.get("name", f"class_{c['id']}")
        by_image: dict[int, list[dict]] = {}
        for a in doc.get("annotations", []):
            by_image.setdefault(a.get("image_id"), []).append(a)
        for img in doc.get("images", []):
            img_path = img_dir / img.get("file_name", "")
            if not img_path.is_file():
                continue
            w = int(img.get("width") or 0)
            h = int(img.get("height") or 0)
            if not (w and h):
                size = _img_size(img_path)
                if not size:
                    continue
                w, h = size
            anns = []
            for a in by_image.get(img.get("id"), []):
                bb = a.get("bbox")
                if not (isinstance(bb, list) and len(bb) == 4):
                    continue
                anns.append({
                    "class_name": cat_name.get(a.get("category_id"), "object"),
                    "bbox": _bbox(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), w, h),
                    "polygon": _seg_to_polygon(a.get("segmentation")),
                })
            samples.append({"path": str(img_path), "annotations": anns})
    return samples


# ── entry point ──────────────────────────────────────────────────
def import_dataset(store: ProjectStore, folder: str, fmt: str = "auto",
                   copy: bool = True) -> dict:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(folder)
    if fmt == "auto":
        fmt = detect_format(root)
    if fmt == "yolo":
        samples = load_yolo(root)
    elif fmt == "coco":
        samples = load_coco(root)
    else:
        raise ValueError(f"Неизвестный формат: {fmt}")
    if not samples:
        raise ValueError("В папке не найдено изображений с разметкой для импорта.")
    result = store.import_samples(samples, copy=copy)
    result["format"] = fmt
    return result

"""Shared dataset-export pipeline used by both YOLO and COCO (RF-DETR) writers.

Centralises the parts that must behave identically across formats:
  * which annotations count (status / geometry filtering),
  * static ROIs (normalised → pixel, per-frame exceptions),
  * train/val split per source image,
  * rotation augmentation (image + annotation geometry recomputed together).

``iter_samples`` yields one record per output image (originals + augmented),
already carrying pixel-space annotation points; format writers just serialise.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


# ── geometry helpers ─────────────────────────────────────────────
def ann_points(ann: dict) -> Optional[np.ndarray]:
    """Polygon points if present, else the bbox as a 4-corner polygon."""
    poly = ann.get("polygon")
    if poly and len(poly) >= 3:
        return np.array([[p["x"], p["y"]] for p in poly], dtype=np.float64)
    bb = ann.get("bbox")
    if bb:
        x, y, bw, bh = bb["x"], bb["y"], bb["width"], bb["height"]
        return np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float64)
    return None


def rotation_matrix(w: int, h: int, angle_deg: float) -> tuple[np.ndarray, int, int]:
    """cv2 rotation matrix about the image centre with an expanded canvas."""
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    m[0, 2] += new_w / 2.0 - w / 2.0
    m[1, 2] += new_h / 2.0 - h / 2.0
    return m, new_w, new_h


def apply_matrix(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return np.hstack([pts.astype(np.float64), ones]) @ m.T


def class_id_map(data: dict) -> tuple[list[dict], dict]:
    """Sorted classes + map original class_id → contiguous 0..N-1 index."""
    classes = sorted(data["classes"], key=lambda c: c["id"])
    return classes, {c["id"]: i for i, c in enumerate(classes)}


def _static_for(data: dict, img: dict, wanted) -> list[dict]:
    """Static ROIs applicable to ``img`` (skip exceptions; norm → pixel)."""
    out = []
    w, h = img["width"], img["height"]
    for r in data.get("static_rois", []):
        if r.get("class_id") is None or img["id"] in (r.get("exceptions") or []):
            continue
        ex = dict(r)
        if r.get("polygon_norm"):
            ex["polygon"] = [{"x": int(round(p["x"] * w)), "y": int(round(p["y"] * h))}
                             for p in r["polygon_norm"]]
        if r.get("bbox_norm"):
            n = r["bbox_norm"]
            ex["bbox"] = {"x": int(round(n["x"] * w)), "y": int(round(n["y"] * h)),
                          "width": int(round(n["width"] * w)), "height": int(round(n["height"] * h))}
        if wanted(ex):
            out.append(ex)
    return out


class Sample:
    """One output image plus its pixel-space annotations."""
    __slots__ = ("split", "stem", "ext", "width", "height", "anns", "pts_list",
                 "src", "array", "is_aug")

    def __init__(self, split, stem, ext, width, height, anns, pts_list, src, array, is_aug):
        self.split, self.stem, self.ext = split, stem, ext
        self.width, self.height = width, height
        self.anns, self.pts_list = anns, pts_list
        self.src, self.array, self.is_aug = src, array, is_aug

    def write_image(self, dest: Path) -> bool:
        """Copy the original or write the rotated array. False if unreadable."""
        import shutil
        if self.array is not None:
            return bool(cv2.imwrite(str(dest), self.array))
        try:
            shutil.copy2(self.src, dest)
            return True
        except Exception:
            return False


def iter_samples(data: dict, *, val_split: float, augment: bool,
                 angles: Optional[list[float]], include_suggested: bool,
                 seed: int = 42) -> Iterator[Sample]:
    angles = angles or []

    def wanted(ann: dict) -> bool:
        if ann.get("status") == "suggested" and not include_suggested:
            return False
        return ann_points(ann) is not None

    images = [
        img for img in data["images"]
        if _static_for(data, img, wanted) or any(wanted(a) for a in img["annotations"])
    ]
    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = int(round(len(images) * val_split))
    val_ids = {img["id"] for img in images[:n_val]}

    for img in images:
        split = "val" if img["id"] in val_ids else "train"
        src = Path(img["path"])
        anns = [a for a in img["annotations"] if wanted(a)] + _static_for(data, img, wanted)
        if not anns:
            continue
        w, h = img["width"], img["height"]
        stem = f"{Path(img['filename']).stem}_{img['id']}"
        pts_list = [ann_points(a) for a in anns]
        ext = src.suffix.lower() if src.suffix else ".jpg"

        yield Sample(split, stem, ext, w, h, anns, pts_list, src, None, False)

        # Rotation augmentation — train split only; val stays clean.
        if augment and split == "train" and angles:
            mat = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if mat is None:
                continue
            for ang in angles:
                m, nw, nh = rotation_matrix(w, h, float(ang))
                rotated = cv2.warpAffine(mat, m, (nw, nh), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
                rot_pts = [apply_matrix(m, p) for p in pts_list]
                yield Sample(split, f"{stem}_rot{int(round(ang))}", ".jpg",
                             nw, nh, anns, rot_pts, None, rotated, True)

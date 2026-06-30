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


# ── tiling helpers ───────────────────────────────────────────────
# Big frames (e.g. 2048×2048) lose tiny objects when a trainer downscales them
# to its input size. Tiling slices each source frame into overlapping
# ``tile_size`` windows so small crystals survive at near-native resolution.
def _tile_starts(length: int, size: int, overlap: float) -> list[int]:
    """Window start coordinates covering ``length`` with the given overlap.

    Overlap keeps objects on a tile seam intact in at least one neighbour. The
    final window is snapped to the edge so the whole frame is covered.
    """
    if length <= size:
        return [0]
    step = max(1, int(round(size * (1.0 - max(0.0, min(0.9, overlap))))))
    starts = list(range(0, length - size + 1, step))
    if not starts or starts[-1] != length - size:
        starts.append(length - size)
    return starts


def _poly_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_poly_rect(pts: np.ndarray, left: float, top: float,
                    right: float, bottom: float) -> np.ndarray:
    """Sutherland–Hodgman clip of a polygon to an axis-aligned rectangle."""
    poly = [(float(p[0]), float(p[1])) for p in pts]
    # (coordinate axis, boundary value, keep-side sign)
    for axis, val, sign in ((0, left, 1), (0, right, -1), (1, top, 1), (1, bottom, -1)):
        if not poly:
            break
        out: list[tuple[float, float]] = []
        for i in range(len(poly)):
            cur, prev = poly[i], poly[i - 1]
            cur_in = sign * (cur[axis] - val) >= 0
            prev_in = sign * (prev[axis] - val) >= 0
            if cur_in != prev_in:
                denom = cur[axis] - prev[axis]
                t = (val - prev[axis]) / denom if denom else 0.0
                out.append((prev[0] + t * (cur[0] - prev[0]),
                            prev[1] + t * (cur[1] - prev[1])))
            if cur_in:
                out.append(cur)
        poly = out
    return np.array(poly, dtype=np.float64) if poly else np.empty((0, 2), dtype=np.float64)


def _iter_tiles(mat: np.ndarray, w: int, h: int, anns: list, pts_list: list,
                stem: str, *, size: int, overlap: float, empty_ratio: float,
                min_visibility: float, rng: random.Random) -> Iterator["Sample"]:
    """Yield one train ``Sample`` per tile of a source frame.

    Each annotation polygon is clipped to the tile and kept only if at least
    ``min_visibility`` of its area survives (so border slivers are dropped — the
    object stays whole in the overlapping neighbour). Tiles with no objects are
    background: only a ``empty_ratio`` fraction is kept as negatives.
    """
    for ty in _tile_starts(h, size, overlap):
        for tx in _tile_starts(w, size, overlap):
            tw, th = min(size, w - tx), min(size, h - ty)
            keep_anns, keep_pts = [], []
            for ann, p in zip(anns, pts_list):
                clipped = _clip_poly_rect(p, tx, ty, tx + tw, ty + th)
                if len(clipped) < 3:
                    continue
                orig = _poly_area(p)
                if orig > 0 and _poly_area(clipped) / orig < min_visibility:
                    continue
                keep_anns.append(ann)
                keep_pts.append(clipped - np.array([tx, ty], dtype=np.float64))
            if not keep_anns and rng.random() >= empty_ratio:
                continue                                # drop most empty tiles
            crop = mat[ty:ty + th, tx:tx + tw].copy()
            yield Sample("train", f"{stem}_t{tx}_{ty}", ".jpg", tw, th,
                         keep_anns, keep_pts, None, crop, True, is_tile=True)


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
                 "src", "array", "is_aug", "is_tile")

    def __init__(self, split, stem, ext, width, height, anns, pts_list, src, array,
                 is_aug, is_tile=False):
        self.split, self.stem, self.ext = split, stem, ext
        self.width, self.height = width, height
        self.anns, self.pts_list = anns, pts_list
        self.src, self.array, self.is_aug, self.is_tile = src, array, is_aug, is_tile

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
                 test_split: float = 0.0, flip_h: bool = False,
                 brightness: bool = False, grayscale: bool = False,
                 tile: bool = False, tile_size: int = 640, tile_overlap: float = 0.2,
                 tile_max_images: int = 0, tile_empty_ratio: float = 0.15,
                 tile_min_visibility: float = 0.3,
                 seed: int = 42) -> Iterator[Sample]:
    """Yield one Sample per output image. Split is "train" | "val" | "test".

    ``test_split`` defaults to 0.0 → only train/val are produced (the original
    two-way behaviour is preserved exactly). Augmentations apply to the train
    split only and each adds one extra copy per image:
      * ``angles``      — one rotated copy per angle (geometry recomputed),
      * ``flip_h``      — horizontal mirror (geometry mirrored),
      * ``brightness``  — random brightness/contrast jitter (pixels only),
      * ``grayscale``   — desaturated copy (pixels only).
    All default off → original behaviour preserved.

    Tiling (``tile``) is a separate pass on the train split only (so no val/test
    leakage): each selected source frame is sliced into overlapping
    ``tile_size`` windows, yielded *in addition to* the full frame.
    ``tile_max_images`` limits how many train frames are sliced (0 = all).
    """
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
    n = len(images)
    n_val = int(round(n * val_split))
    n_test = int(round(n * max(0.0, test_split)))
    # Guard against over-allocating when fractions sum past 1 on tiny datasets.
    n_test = min(n_test, max(0, n - n_val))
    val_ids = {img["id"] for img in images[:n_val]}
    test_ids = {img["id"] for img in images[n_val:n_val + n_test]}

    # Which train frames get sliced into tiles (0 = all). Selection follows the
    # shuffled order so it is deterministic for a given seed.
    tile_ids: set = set()
    if tile:
        train_order = [img["id"] for img in images
                       if img["id"] not in val_ids and img["id"] not in test_ids]
        tile_ids = set(train_order[:tile_max_images] if tile_max_images and tile_max_images > 0
                       else train_order)

    for img in images:
        if img["id"] in val_ids:
            split = "val"
        elif img["id"] in test_ids:
            split = "test"
        else:
            split = "train"
        src = Path(img["path"])
        anns = [a for a in img["annotations"] if wanted(a)] + _static_for(data, img, wanted)
        if not anns:
            continue
        w, h = img["width"], img["height"]
        stem = f"{Path(img['filename']).stem}_{img['id']}"
        pts_list = [ann_points(a) for a in anns]
        ext = src.suffix.lower() if src.suffix else ".jpg"

        yield Sample(split, stem, ext, w, h, anns, pts_list, src, None, False)

        # Augmentation & tiling — train split only; val/test stay clean.
        need_aug = augment and split == "train" and (angles or flip_h or brightness or grayscale)
        need_tile = tile and split == "train" and img["id"] in tile_ids
        if need_aug or need_tile:
            mat = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if mat is None:
                continue
        if need_aug:
            # Rotations (geometry recomputed via the rotation matrix).
            for ang in angles:
                m, nw, nh = rotation_matrix(w, h, float(ang))
                rotated = cv2.warpAffine(mat, m, (nw, nh), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
                rot_pts = [apply_matrix(m, p) for p in pts_list]
                yield Sample(split, f"{stem}_rot{int(round(ang))}", ".jpg",
                             nw, nh, anns, rot_pts, None, rotated, True)
            # Horizontal flip (mirror x; dimensions unchanged).
            if flip_h:
                flipped = cv2.flip(mat, 1)
                fl_pts = [np.column_stack([(w - 1) - p[:, 0], p[:, 1]]) for p in pts_list]
                yield Sample(split, f"{stem}_flip", ".jpg", w, h, anns, fl_pts, None, flipped, True)
            # Brightness / contrast jitter (pixels only; geometry unchanged).
            if brightness:
                alpha = rng.uniform(0.75, 1.3)      # contrast
                beta = rng.uniform(-30, 30)         # brightness offset
                bright = cv2.convertScaleAbs(mat, alpha=alpha, beta=beta)
                yield Sample(split, f"{stem}_bright", ".jpg", w, h, anns, pts_list, None, bright, True)
            # Grayscale (pixels only; kept 3-channel so exporters stay uniform).
            if grayscale:
                gray = cv2.cvtColor(cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                yield Sample(split, f"{stem}_gray", ".jpg", w, h, anns, pts_list, None, gray, True)

        # Tiling pass — extra small windows of this frame (train split only).
        if need_tile:
            yield from _iter_tiles(mat, w, h, anns, pts_list, stem,
                                   size=tile_size, overlap=tile_overlap,
                                   empty_ratio=tile_empty_ratio,
                                   min_visibility=tile_min_visibility, rng=rng)

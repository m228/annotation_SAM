"""Project persistence.

A project is a folder under ``PROJECTS_DIR`` containing a single
``project.json``. Uploaded images are copied into ``<project>/images``;
folder-imported images are referenced by their absolute path. This keeps the
data model trivial to inspect, back up, and extend.
"""
from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps

from .config import PROJECTS_DIR, IMAGE_EXTENSIONS


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-").lower()
    return slug or "project"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


DEFAULT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
]


class ProjectStore:
    """CRUD + image management for a single project on disk."""

    def __init__(self, slug: str):
        self.slug = slug
        self.dir = PROJECTS_DIR / slug
        self.images_dir = self.dir / "images"
        self.json_path = self.dir / "project.json"

    # ── lifecycle ────────────────────────────────────────────────
    @classmethod
    def create(cls, name: str) -> "ProjectStore":
        slug = _slug(name)
        candidate = slug
        i = 2
        while (PROJECTS_DIR / candidate).exists():
            candidate = f"{slug}-{i}"
            i += 1
        store = cls(candidate)
        store.images_dir.mkdir(parents=True, exist_ok=True)
        store._write({
            "name": name,
            "slug": candidate,
            "created_at": _now(),
            "updated_at": _now(),
            "classes": [],
            "images": [],
            "propagation_rules": [],
            "trained_models": [],   # records of local training runs (see add_trained_model)
            "static_rois": [],   # ROIs reused on every frame (see set_static_rois)
            "settings": {
                "export_format": "detect",   # "detect" (bbox) | "segment" (polygon)
                "val_split": 0.1,
                "augment": {"enabled": False, "angles": [90, 180, 270]},
                "sam3_confidence": 0.5,
            },
        })
        return store

    @classmethod
    def list_projects(cls) -> list[dict]:
        out = []
        for d in sorted(PROJECTS_DIR.iterdir()):
            pj = d / "project.json"
            if pj.is_file():
                try:
                    data = json.loads(pj.read_text(encoding="utf-8"))
                except Exception:
                    continue
                out.append({
                    "slug": d.name,
                    "name": data.get("name", d.name),
                    "images": len(data.get("images", [])),
                    "classes": len(data.get("classes", [])),
                    "updated_at": data.get("updated_at", ""),
                })
        return out

    def exists(self) -> bool:
        return self.json_path.is_file()

    def delete_project(self) -> None:
        """Remove the whole project folder (json + copied images)."""
        if self.dir.is_dir():
            shutil.rmtree(self.dir, ignore_errors=True)

    # ── raw read/write ───────────────────────────────────────────
    def load(self) -> dict:
        data = json.loads(self.json_path.read_text(encoding="utf-8"))
        data.setdefault("static_rois", [])      # migrate older projects
        data.setdefault("trained_models", [])
        return data

    def _write(self, data: dict) -> None:
        data["updated_at"] = _now()
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.json_path)

    def save(self, data: dict) -> dict:
        self._write(data)
        return data

    # ── classes ──────────────────────────────────────────────────
    def add_class(self, name: str, color: Optional[str] = None) -> dict:
        data = self.load()
        classes = data["classes"]
        cid = (max((c["id"] for c in classes), default=-1) + 1)
        color = color or DEFAULT_COLORS[cid % len(DEFAULT_COLORS)]
        cls = {"id": cid, "name": name, "color": color}
        classes.append(cls)
        self._write(data)
        return cls

    def update_class(self, cid: int, name: Optional[str], color: Optional[str]) -> dict:
        data = self.load()
        for c in data["classes"]:
            if c["id"] == cid:
                if name is not None:
                    c["name"] = name
                if color is not None:
                    c["color"] = color
                self._write(data)
                return c
        raise KeyError(f"class {cid} not found")

    def delete_class(self, cid: int) -> None:
        data = self.load()
        data["classes"] = [c for c in data["classes"] if c["id"] != cid]
        for img in data["images"]:
            img["annotations"] = [a for a in img["annotations"] if a.get("class_id") != cid]
        data["propagation_rules"] = [
            r for r in data["propagation_rules"] if r.get("class_id") != cid
        ]
        data["static_rois"] = [
            r for r in data.get("static_rois", []) if r.get("class_id") != cid
        ]
        self._write(data)

    # ── static ROIs (reused on every frame) ──────────────────────
    def set_static_rois(self, rois: list[dict]) -> list[dict]:
        """Replace the project-wide ROIs that apply to all images."""
        data = self.load()
        clean = []
        for r in rois:
            r = dict(r)
            r.setdefault("id", new_id())
            r.setdefault("source", "manual")
            r["status"] = "confirmed"
            r["static"] = True
            clean.append(r)
        data["static_rois"] = clean
        self._write(data)
        return clean

    # ── images ───────────────────────────────────────────────────
    def _image_record(self, path: Path, copied: bool) -> Optional[dict]:
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                w, h = im.size
        except Exception:
            return None
        return {
            "id": new_id(),
            "filename": path.name,
            "path": str(path.resolve()),
            "copied": copied,
            "width": w,
            "height": h,
            "annotations": [],
        }

    def _existing_paths(self, data: dict) -> set[str]:
        return {img["path"] for img in data["images"]}

    def import_folder(self, folder: str) -> int:
        """Reference every image found in ``folder`` (non-recursive copy-free)."""
        data = self.load()
        existing = self._existing_paths(data)
        added = 0
        base = Path(folder)
        if not base.is_dir():
            raise FileNotFoundError(folder)
        for p in sorted(base.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if str(p.resolve()) in existing:
                continue
            rec = self._image_record(p, copied=False)
            if rec:
                data["images"].append(rec)
                added += 1
        self._write(data)
        return added

    def add_uploaded(self, filename: str, raw: bytes) -> Optional[dict]:
        """Save uploaded bytes into the project's images folder and register it."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        dest = self.images_dir / filename
        stem, suffix = dest.stem, dest.suffix
        i = 2
        while dest.exists():
            dest = self.images_dir / f"{stem}-{i}{suffix}"
            i += 1
        dest.write_bytes(raw)
        rec = self._image_record(dest, copied=True)
        if not rec:
            dest.unlink(missing_ok=True)
            return None
        data = self.load()
        data["images"].append(rec)
        self._write(data)
        return rec

    def import_samples(self, samples: list[dict], copy: bool = True) -> dict:
        """Bulk-add images + annotations from an imported dataset.

        Each sample = ``{"path": <src image>, "annotations": [{"class_name",
        "bbox", "polygon", "confidence"?}, …]}``. Classes are created on demand,
        matched by name (case-insensitive). Images are copied into the project by
        default. Written to disk once at the end (not per-image)."""
        data = self.load()
        name_to_id = {c["name"].strip().lower(): c["id"] for c in data["classes"]}
        classes_added = 0

        def _ensure_class(name: str) -> int:
            nonlocal classes_added
            key = (name or "object").strip().lower()
            if key in name_to_id:
                return name_to_id[key]
            cid = max((c["id"] for c in data["classes"]), default=-1) + 1
            data["classes"].append(
                {"id": cid, "name": name, "color": DEFAULT_COLORS[cid % len(DEFAULT_COLORS)]})
            name_to_id[key] = cid
            classes_added += 1
            return cid

        self.images_dir.mkdir(parents=True, exist_ok=True)
        images_added = anns_added = 0
        for s in samples:
            src = Path(s.get("path", ""))
            if not src.is_file():
                continue
            if copy:
                dest = self.images_dir / src.name
                stem, suffix, i = dest.stem, dest.suffix, 2
                while dest.exists():
                    dest = self.images_dir / f"{stem}-{i}{suffix}"
                    i += 1
                try:
                    dest.write_bytes(src.read_bytes())
                except Exception:
                    continue
                rec = self._image_record(dest, copied=True)
            else:
                rec = self._image_record(src, copied=False)
            if not rec:
                continue
            anns = []
            for a in s.get("annotations", []):
                anns.append({
                    "id": new_id(),
                    "class_id": _ensure_class(a.get("class_name", "object")),
                    "bbox": a.get("bbox"),
                    "polygon": a.get("polygon"),
                    "confidence": a.get("confidence", 1.0),
                    "source": "imported",
                    "status": "confirmed",
                })
            rec["annotations"] = anns
            data["images"].append(rec)
            images_added += 1
            anns_added += len(anns)
        self._write(data)
        return {"images": images_added, "annotations": anns_added, "classes": classes_added}

    def get_image(self, image_id: str) -> Optional[dict]:
        for img in self.load()["images"]:
            if img["id"] == image_id:
                return img
        return None

    def delete_image(self, image_id: str) -> None:
        data = self.load()
        target = next((i for i in data["images"] if i["id"] == image_id), None)
        if target and target.get("copied"):
            Path(target["path"]).unlink(missing_ok=True)
        data["images"] = [i for i in data["images"] if i["id"] != image_id]
        self._write(data)

    # ── annotations ──────────────────────────────────────────────
    def set_annotations(self, image_id: str, annotations: list[dict]) -> dict:
        """Replace the full annotation list for an image (UI is source of truth)."""
        data = self.load()
        for img in data["images"]:
            if img["id"] == image_id:
                clean = []
                for a in annotations:
                    a = dict(a)
                    a.setdefault("id", new_id())
                    a.setdefault("source", "manual")
                    a.setdefault("status", "confirmed")
                    clean.append(a)
                img["annotations"] = clean
                self._write(data)
                return img
        raise KeyError(f"image {image_id} not found")

    def add_suggestions(self, image_id: str, suggestions: list[dict]) -> int:
        """Append SAM-generated suggestions, skipping images already annotated."""
        data = self.load()
        for img in data["images"]:
            if img["id"] == image_id:
                # Don't overwrite confirmed work the user already did.
                if any(a.get("status") == "confirmed" for a in img["annotations"]):
                    return 0
                # Drop previous suggestions so re-running is idempotent.
                img["annotations"] = [
                    a for a in img["annotations"] if a.get("status") != "suggested"
                ]
                for s in suggestions:
                    s = dict(s)
                    s.setdefault("id", new_id())
                    s["status"] = "suggested"
                    img["annotations"].append(s)
                self._write(data)
                return len(suggestions)
        return 0

    # ── propagation rules ────────────────────────────────────────
    def upsert_rule(self, class_id: int, text_prompt: str, confidence: float) -> dict:
        data = self.load()
        rules = data["propagation_rules"]
        for r in rules:
            if r["class_id"] == class_id:
                r["text_prompt"] = text_prompt
                r["confidence"] = confidence
                self._write(data)
                return r
        rule = {"class_id": class_id, "text_prompt": text_prompt, "confidence": confidence}
        rules.append(rule)
        self._write(data)
        return rule

    def update_settings(self, patch: dict) -> dict:
        data = self.load()
        data["settings"].update(patch)
        self._write(data)
        return data["settings"]

    # ── training runs (local model training) ─────────────────────
    @property
    def train_dir(self) -> Path:
        """Working directory for training runs (datasets + checkpoints)."""
        return self.dir / "_train"

    def run_dir(self, run_id: str) -> Path:
        return self.train_dir / run_id

    def add_trained_model(self, record: dict) -> dict:
        """Insert or update a trained-model record (keyed by run_id)."""
        data = self.load()
        models = data.setdefault("trained_models", [])
        rid = record.get("run_id")
        for i, m in enumerate(models):
            if m.get("run_id") == rid:
                models[i] = {**m, **record}
                self._write(data)
                return models[i]
        record.setdefault("created_at", _now())
        models.insert(0, record)        # newest first
        self._write(data)
        return record

    def list_trained_models(self) -> list[dict]:
        return self.load().get("trained_models", [])

    def delete_trained_model(self, run_id: str) -> None:
        data = self.load()
        data["trained_models"] = [
            m for m in data.get("trained_models", []) if m.get("run_id") != run_id
        ]
        self._write(data)
        # Remove the run's dataset + checkpoints from disk.
        run = self.run_dir(run_id)
        if run.is_dir():
            shutil.rmtree(run, ignore_errors=True)

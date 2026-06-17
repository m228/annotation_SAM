"""
JSON Lines Protocol - Shared request/response definitions for CLI IPC

All communication between the main app and ml_backend uses JSON Lines
(one JSON object per line) over stdin/stdout. stderr is reserved for
human-readable logs.

SAM Service Protocol (long-running, bidirectional):
  Request  → one JSON line on stdin
  Response → one JSON line on stdout

Training Service Protocol (single-shot, output-only):
  Config   → CLI args or config file path
  Progress → JSON lines streamed on stdout
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional


# ────────────────────────────────────────────────────────────
# SAM Protocol
# ────────────────────────────────────────────────────────────

# --- Requests (main app → SAM service via stdin) ---

class SAMCommands:
    LOAD = "load"
    SET_IMAGE = "set_image"
    PREDICT_POINTS = "predict_points"
    PREDICT_BOX = "predict_box"
    AUTO_SEGMENT = "auto_segment"
    GET_MODELS = "get_models"
    HEALTH = "health"
    SHUTDOWN = "shutdown"


def sam_load_request(model: str = "sam2-local") -> dict:
    return {"cmd": SAMCommands.LOAD, "model": model}


def sam_set_image_request(image_path: str, image_id: str = "", model: str = "") -> dict:
    req = {"cmd": SAMCommands.SET_IMAGE, "image_path": image_path, "image_id": image_id}
    if model:
        req["model"] = model
    return req


def sam_predict_points_request(
    points: List[Dict[str, Any]],
    image_id: str = "",
    multimask: bool = True,
    decode_mask: bool = True,
) -> dict:
    return {
        "cmd": SAMCommands.PREDICT_POINTS,
        "points": points,
        "image_id": image_id,
        "multimask": multimask,
        "decode_mask": decode_mask,
    }


def sam_predict_box_request(
    box: Dict[str, int],
    image_id: str = "",
    multimask: bool = True,
    decode_mask: bool = True,
) -> dict:
    return {
        "cmd": SAMCommands.PREDICT_BOX,
        "box": box,
        "image_id": image_id,
        "multimask": multimask,
        "decode_mask": decode_mask,
    }


def sam_auto_segment_request(
    image_path: str,
    image_id: str = "",
    model: str = "sam3-local",
    text_prompt: str = "",
    confidence_threshold: float = 0.5,
    cpu_max_side: int = 0,
    cpu_threads: int = 0,
) -> dict:
    req = {
        "cmd": SAMCommands.AUTO_SEGMENT,
        "image_path": image_path,
        "image_id": image_id,
        "text_prompt": text_prompt,
        "confidence_threshold": confidence_threshold,
    }
    if cpu_max_side:
        req["cpu_max_side"] = cpu_max_side
    if cpu_threads:
        req["cpu_threads"] = cpu_threads
    if model:
        req["model"] = model
    return req


def sam_get_models_request() -> dict:
    return {"cmd": SAMCommands.GET_MODELS}


def sam_health_request() -> dict:
    return {"cmd": SAMCommands.HEALTH}


def sam_shutdown_request() -> dict:
    return {"cmd": SAMCommands.SHUTDOWN}


# --- Responses (SAM service → main app via stdout) ---

def ok_response(response_type: str, **data) -> dict:
    return {"status": "ok", "type": response_type, **data}


def error_response(message: str, response_type: str = "error") -> dict:
    return {"status": "error", "type": response_type, "message": message}


def progress_response(message: str, step: str = "", response_type: str = "progress") -> dict:
    return {"status": "progress", "type": response_type, "message": message, "step": step}


# ────────────────────────────────────────────────────────────
# Training Protocol
# ────────────────────────────────────────────────────────────

class TrainingMessageTypes:
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    LOG = "log"


def training_progress_message(
    status: str = "idle",
    message: str = "",
    current_epoch: int = 0,
    total_epochs: int = 0,
    current_step: int = 0,
    total_steps: int = 0,
    loss: float = 0.0,
    learning_rate: float = 0.0,
    percentage: float = 0.0,
    accuracy: Optional[float] = None,
    eta_seconds: Optional[float] = None,
    job_id: str = "",
    **extra,
) -> dict:
    msg = {
        "type": TrainingMessageTypes.PROGRESS,
        "status": status,
        "message": message,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "current_step": current_step,
        "total_steps": total_steps,
        "loss": loss,
        "learning_rate": learning_rate,
        "percentage": percentage,
        "job_id": job_id,
    }
    if accuracy is not None:
        msg["accuracy"] = accuracy
    if eta_seconds is not None:
        msg["eta_seconds"] = eta_seconds
    msg.update(extra)
    return msg


def training_result_message(
    status: str = "completed",
    model_path: str = "",
    output_dir: str = "",
    message: str = "",
    job_id: str = "",
    **metrics,
) -> dict:
    return {
        "type": TrainingMessageTypes.RESULT,
        "status": status,
        "model_path": model_path,
        "output_dir": output_dir,
        "message": message,
        "job_id": job_id,
        **metrics,
    }


def training_error_message(message: str, job_id: str = "") -> dict:
    return {
        "type": TrainingMessageTypes.ERROR,
        "status": "error",
        "message": message,
        "job_id": job_id,
    }


# ────────────────────────────────────────────────────────────
# Wire helpers
# ────────────────────────────────────────────────────────────

def write_json_line(obj: dict, stream=None) -> None:
    """Write a single JSON line to a stream (default: stdout)."""
    if stream is None:
        stream = sys.stdout
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    stream.write(line + "\n")
    stream.flush()


def read_json_line(stream=None) -> Optional[dict]:
    """Read a single JSON line from a stream (default: stdin).
    Returns None on EOF or empty line.
    """
    if stream is None:
        stream = sys.stdin
    line = stream.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def log(message: str) -> None:
    """Write a log message to stderr (never interferes with protocol on stdout)."""
    print(message, file=sys.stderr, flush=True)


"""
SAM Service - Long-running process for SAM segmentation

Launched by the main app via subprocess. Reads JSON commands from stdin
and writes JSON responses to stdout. Primarily serves SAM3 auto-segmentation
requests in a long-running process.

Usage:
    python -m ml_backend sam [--model sam3-local]

Protocol:
    stdin  → JSON Lines requests  (see protocol.py)
    stdout → JSON Lines responses (see protocol.py)
    stderr → human-readable logs
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from .protocol import (
    SAMCommands,
    ok_response,
    error_response,
    progress_response,
    write_json_line,
    read_json_line,
    log,
)
from .sam3_compat import install_sam3_edt_fallback


def _is_cuda_device(value: Any) -> bool:
    """Return True for torch CUDA device values without importing torch globally."""
    if isinstance(value, str):
        return value.lower().startswith("cuda")
    return getattr(value, "type", None) == "cuda"


def _with_cpu_device_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if _is_cuda_device(kwargs.get("device")):
        kwargs = dict(kwargs)
        kwargs["device"] = "cpu"
    return kwargs


def _cpu_addmm_act(torch_module, activation, linear, mat1):
    if torch_module.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")

    weight = linear.weight
    bias = linear.bias
    if mat1.dtype != weight.dtype:
        mat1 = mat1.to(weight.dtype)

    y = torch_module.nn.functional.linear(mat1, weight, bias)
    if activation in [torch_module.nn.functional.relu, torch_module.nn.ReLU]:
        return torch_module.nn.functional.relu(y)
    if activation in [torch_module.nn.functional.gelu, torch_module.nn.GELU]:
        return torch_module.nn.functional.gelu(y)
    raise ValueError(f"Unexpected activation {activation}")


@contextmanager
def _sam3_cpu_compatibility(torch_module, enabled: bool):
    """Redirect hardcoded SAM3 CUDA calls to CPU on CPU-only installs.

    The current SAM3 package still has a few construction/inference paths with
    literal ``device="cuda"``, ``.cuda()``, and CPU tensor ``.pin_memory()``
    calls. On a CPU-only PyTorch build those fail before our requested
    ``device="cpu"`` can take effect.
    """
    if not enabled:
        yield
        return

    factory_names = (
        "arange",
        "as_tensor",
        "empty",
        "eye",
        "full",
        "linspace",
        "ones",
        "rand",
        "randint",
        "randn",
        "tensor",
        "zeros",
    )

    originals = {}
    for name in factory_names:
        if hasattr(torch_module, name):
            original = getattr(torch_module, name)
            originals[("torch", name)] = original

            def _wrap_factory(func):
                def _wrapped(*args, **kwargs):
                    return func(*args, **_with_cpu_device_kwargs(kwargs))

                return _wrapped

            setattr(torch_module, name, _wrap_factory(original))

    originals[("tensor", "cuda")] = torch_module.Tensor.cuda
    originals[("tensor", "pin_memory")] = torch_module.Tensor.pin_memory
    originals[("module", "cuda")] = torch_module.nn.Module.cuda
    torch_module.Tensor.cuda = lambda self, *args, **kwargs: self.to("cpu")
    torch_module.Tensor.pin_memory = lambda self, *args, **kwargs: self
    torch_module.nn.Module.cuda = lambda self, *args, **kwargs: self.to("cpu")

    def _wrap_autocast(func):
        def _wrapped(*args, **kwargs):
            if args and _is_cuda_device(args[0]):
                args = ("cpu",) + args[1:]
            if _is_cuda_device(kwargs.get("device_type")):
                kwargs = dict(kwargs)
                kwargs["device_type"] = "cpu"
            return func(*args, **kwargs)

        return _wrapped

    originals[("torch", "autocast")] = torch_module.autocast
    torch_module.autocast = _wrap_autocast(torch_module.autocast)
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "autocast"):
        originals[("amp", "autocast")] = torch_module.amp.autocast
        torch_module.amp.autocast = _wrap_autocast(torch_module.amp.autocast)

    try:
        import sam3.model.vitdet as sam3_vitdet
        import sam3.perflib.fused as sam3_fused

        cpu_addmm_act = lambda activation, linear, mat1: _cpu_addmm_act(
            torch_module,
            activation,
            linear,
            mat1,
        )
        originals[("sam3_fused", "addmm_act")] = sam3_fused.addmm_act
        originals[("sam3_vitdet", "addmm_act")] = sam3_vitdet.addmm_act
        sam3_fused.addmm_act = cpu_addmm_act
        sam3_vitdet.addmm_act = cpu_addmm_act
    except Exception as exc:
        log(f"Could not patch SAM3 CPU fused MLP path: {exc}")

    try:
        yield
    finally:
        for (owner, name), original in originals.items():
            if owner == "torch":
                setattr(torch_module, name, original)
            elif owner == "amp":
                setattr(torch_module.amp, name, original)
            elif owner == "tensor":
                setattr(torch_module.Tensor, name, original)
            elif owner == "module":
                torch_module.nn.Module.cuda = original
            elif owner == "sam3_fused":
                import sam3.perflib.fused as sam3_fused

                setattr(sam3_fused, name, original)
            elif owner == "sam3_vitdet":
                import sam3.model.vitdet as sam3_vitdet

                setattr(sam3_vitdet, name, original)


class SAMService:
    """Long-running SAM segmentation service."""

    INTERACTIVE_MODEL = "sam2-local"
    INTERACTIVE_CHECKPOINT_CANDIDATES = (
        "sam2.1_hiera_large.pt",
        "sam2_hiera_large.pt",
        "sam2.pt",
    )
    INTERACTIVE_CONFIG_DEFAULT = "sam2.1_hiera_l.yaml"
    INTERACTIVE_CONFIG_ALIASES = (
        "sam2.1_hiera_l.yaml",
        "sam2_hiera_l.yaml",
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "configs/sam2/sam2_hiera_l.yaml",
    )

    AUTO_SEGMENT_MODEL = "sam3-local"
    AUTO_SEGMENT_CHECKPOINT = "sam3.pt"
    AUTO_SEGMENT_CPU_MAX_SIDE = 1008
    AUTO_SEGMENT_CPU_THREADS = 0
    SUPPORTED_MODELS = [INTERACTIVE_MODEL, AUTO_SEGMENT_MODEL]
    DEFAULT_MODEL = INTERACTIVE_MODEL

    def __init__(self):
        self._model_name: Optional[str] = None
        self._interactive_predictor = None
        self._interactive_model_name: Optional[str] = None
        self._auto_segment_model = None
        self._auto_segment_model_name: Optional[str] = None
        self._cached_image_id: Optional[str] = None
        self._interactive_image_size: Optional[tuple[int, int]] = None
        self._device: str = "cpu"
        self._cpu_thread_count: Optional[int] = None
        self._cpu_interop_thread_count: Optional[int] = None

    @classmethod
    def _normalize_model_name(cls, model_name: str = "") -> str:
        normalized = (model_name or "").strip().lower()
        if normalized in {"", "sam2", cls.INTERACTIVE_MODEL}:
            return cls.INTERACTIVE_MODEL
        if normalized in {"sam3", cls.AUTO_SEGMENT_MODEL}:
            return cls.AUTO_SEGMENT_MODEL
        return model_name

    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import torch
            log(
                "Torch device check: "
                f"version={getattr(torch, '__version__', 'unknown')}, "
                f"cuda_available={torch.cuda.is_available()}, "
                f"cuda_version={getattr(getattr(torch, 'version', None), 'cuda', None)}"
            )
            if torch.cuda.is_available():
                try:
                    log(f"CUDA device: {torch.cuda.get_device_name(0)}")
                except Exception:
                    pass
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError as exc:
            log(f"Torch import failed during device detection: {exc}")
        except Exception as exc:
            log(f"Torch device detection failed: {exc}")
        return "cpu"

    def load_model(self, model_name: str = "") -> dict:
        """Load or switch the SAM model used for segmentation."""
        model_name = self._normalize_model_name(model_name or self.DEFAULT_MODEL)

        if model_name == self._model_name:
            return ok_response("load", model=self._model_name)

        if model_name not in self.SUPPORTED_MODELS:
            return error_response(
                f"Unknown model: {model_name}. Supported: {self.SUPPORTED_MODELS}",
                response_type="load",
            )

        try:
            self._device = self._detect_device()
            log(f"Loading SAM model '{model_name}' on device '{self._device}'...")

            if model_name == self.INTERACTIVE_MODEL:
                predictor, load_error = self._load_interactive_model(model_name)
                if predictor is None:
                    return error_response(load_error or "Could not load SAM2 model", response_type="load")
            else:
                processor, load_error = self._load_auto_segment_model(model_name)
                if processor is None:
                    return error_response(load_error or "Could not load SAM3 model", response_type="load")

            self._model_name = model_name
            self._cached_image_id = None
            self._interactive_image_size = None

            log(f"SAM model '{model_name}' loaded successfully on {self._device}")
            return ok_response("load", model=model_name, device=self._device)

        except Exception as e:
            log(f"Error loading model: {traceback.format_exc()}")
            return error_response(str(e), response_type="load")

    def _find_sam3_checkpoint(self, model: str = "") -> Optional[str]:
        """Resolve the local SAM3 checkpoint path used for auto-annotation."""
        if model and model not in {self.AUTO_SEGMENT_MODEL, "sam3"}:
            candidate = Path(model)
            if candidate.exists():
                return str(candidate.resolve())

        search_paths = []

        env_path = os.environ.get("ML_BACKEND_SAM3_PATH")
        if env_path:
            search_paths.append(Path(env_path))

        project_root = Path(__file__).resolve().parent.parent
        search_paths.extend([
            project_root / "models" / self.AUTO_SEGMENT_CHECKPOINT,
            Path.cwd() / "models" / self.AUTO_SEGMENT_CHECKPOINT,
        ])

        for candidate in search_paths:
            if candidate.exists():
                log(f"SAM3 checkpoint found: {candidate}")
                return str(candidate.resolve())

        log("SAM3 checkpoint not found. Searched: " + " | ".join(str(path) for path in search_paths))
        return None

    def _find_sam2_checkpoint(self, model: str = "") -> Optional[str]:
        """Resolve local SAM2 checkpoint path for interactive segmentation."""
        if model and model not in {self.INTERACTIVE_MODEL, "sam2"}:
            candidate = Path(model)
            if candidate.exists():
                return str(candidate.resolve())

        search_paths = []

        env_path = os.environ.get("ML_BACKEND_SAM2_PATH")
        if env_path:
            search_paths.append(Path(env_path))

        project_root = Path(__file__).resolve().parent.parent
        for checkpoint_name in self.INTERACTIVE_CHECKPOINT_CANDIDATES:
            search_paths.append(project_root / "models" / checkpoint_name)
            search_paths.append(Path.cwd() / "models" / checkpoint_name)

        for candidate in search_paths:
            if candidate.exists():
                log(f"SAM2 checkpoint found: {candidate}")
                return str(candidate.resolve())

        log("SAM2 checkpoint not found. Searched: " + " | ".join(str(path) for path in search_paths))
        return None

    def _resolve_sam2_config_candidates(self) -> list[str]:
        """Build an ordered list of SAM2 config path/identifier candidates."""
        config_hint = os.environ.get("ML_BACKEND_SAM2_CONFIG", self.INTERACTIVE_CONFIG_DEFAULT).strip()
        if not config_hint:
            config_hint = self.INTERACTIVE_CONFIG_DEFAULT

        candidates: list[str] = []
        project_root = Path(__file__).resolve().parent.parent

        def _append(value: str) -> None:
            v = (value or "").strip()
            if v and v not in candidates:
                candidates.append(v)

        def _append_if_exists(path_value: Path) -> None:
            try:
                if path_value.exists():
                    _append(str(path_value.resolve()))
            except Exception:
                pass

        _append(config_hint)
        for alias in self.INTERACTIVE_CONFIG_ALIASES:
            _append(alias)

        base_name = Path(config_hint).name
        if base_name:
            _append(base_name)
            if base_name == "sam2_hiera_l.yaml":
                _append("sam2.1_hiera_l.yaml")
                _append("configs/sam2/sam2_hiera_l.yaml")
            elif base_name == "sam2.1_hiera_l.yaml":
                _append("sam2_hiera_l.yaml")
                _append("configs/sam2.1/sam2.1_hiera_l.yaml")

        for candidate in list(candidates):
            candidate_path = Path(candidate)
            if candidate_path.is_absolute():
                _append_if_exists(candidate_path)
                continue

            _append_if_exists(project_root / candidate)
            _append_if_exists(project_root / "models" / candidate)
            _append_if_exists(Path.cwd() / candidate)
            _append_if_exists(Path.cwd() / "models" / candidate)

        return candidates

    def _build_sam2_compatible(self, build_sam2, config_value: str, checkpoint_path: str):
        """Call build_sam2 across different SAM2 package signatures."""
        attempted = []

        call_specs = [
            lambda: build_sam2(config_file=config_value, ckpt_path=checkpoint_path, device=self._device),
            lambda: build_sam2(config_file=config_value, checkpoint=checkpoint_path, device=self._device),
            lambda: build_sam2(model_cfg=config_value, checkpoint=checkpoint_path, device=self._device),
            lambda: build_sam2(config_value, checkpoint_path, device=self._device),
            lambda: build_sam2(config_value, checkpoint_path, self._device),
        ]

        last_exc: Exception | None = None
        for fn in call_specs:
            try:
                return fn()
            except TypeError as exc:
                attempted.append(str(exc))
                last_exc = exc
                continue

        raise TypeError("; ".join(attempted)) from last_exc

    def _load_interactive_model(self, model: str = ""):
        """Load SAM2 interactive predictor."""
        requested_model = model or self.INTERACTIVE_MODEL

        if requested_model == self._interactive_model_name and self._interactive_predictor is not None:
            return self._interactive_predictor, None

        checkpoint_path = self._find_sam2_checkpoint(requested_model)
        if not checkpoint_path:
            return None, (
                "SAM2 checkpoint not found. Expected one of "
                f"{list(self.INTERACTIVE_CHECKPOINT_CANDIDATES)} in 'models/' or set ML_BACKEND_SAM2_PATH."
            )

        self._device = self._detect_device()

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            return None, (
                "SAM2 is not installed in ml_backend/.venv. "
                "Install it in the backend venv and set ML_BACKEND_SAM2_PATH. "
                f"Error: {exc}"
            )

        try:
            config_candidates = self._resolve_sam2_config_candidates()
            sam2_model = None
            load_errors: list[str] = []

            for sam2_config in config_candidates:
                log(
                    f"Loading SAM2 checkpoint '{checkpoint_path}' with config '{sam2_config}' on '{self._device}'"
                )
                try:
                    sam2_model = self._build_sam2_compatible(build_sam2, sam2_config, checkpoint_path)
                    break
                except Exception as cfg_exc:
                    load_errors.append(f"{sam2_config}: {cfg_exc}")

            if sam2_model is None:
                raise RuntimeError(
                    "Unable to build SAM2 model. Tried configs: " + " | ".join(load_errors)
                )

            predictor = SAM2ImagePredictor(sam2_model)
            self._interactive_predictor = predictor
            self._interactive_model_name = requested_model
            return self._interactive_predictor, None
        except Exception as exc:
            log(f"Error loading SAM2: {traceback.format_exc()}")
            return None, f"Could not load SAM2 checkpoint '{checkpoint_path}': {exc}"

    def _load_auto_segment_model(self, model: str = ""):
        """Load the local SAM3 checkpoint for automatic segmentation using the SAM3 library."""
        requested_model = model or self.AUTO_SEGMENT_MODEL

        if requested_model == self._auto_segment_model_name and self._auto_segment_model is not None:
            return self._auto_segment_model, None

        checkpoint_path = self._find_sam3_checkpoint(requested_model)
        if not checkpoint_path:
            return None, (
                "SAM3 checkpoint not found. Expected 'models/sam3.pt' in the project root "
                "or set ML_BACKEND_SAM3_PATH."
            )

        self._device = self._detect_device()
        if self._device == "mps":
            log("SAM3 does not support the macOS MPS backend reliably; using CPU.")
            self._device = "cpu"

        try:
            install_sam3_edt_fallback(log)
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            log(f"SAM3 import failed: {traceback.format_exc()}")
            return None, (
                "SAM3 could not be imported in ml_backend/.venv. "
                "Run the ML backend installer to install SAM3 and its native dependencies. "
                f"Error: {exc}"
            )

        try:
            import torch

            log(
                f"Loading SAM3 checkpoint '{checkpoint_path}' on '{self._device}'"
            )
            cpu_compat = self._device == "cpu" and not torch.cuda.is_available()
            if cpu_compat:
                log("SAM3 is loading on CPU because no CUDA/MPS device was detected.")
                log("Applying SAM3 CPU compatibility patch for hardcoded CUDA operations.")
            with _sam3_cpu_compatibility(torch, enabled=cpu_compat):
                sam3_model = build_sam3_image_model(
                    checkpoint_path=checkpoint_path,
                    device=self._device,
                    eval_mode=True,
                    load_from_HF=False,
                    compile=False,
                    enable_segmentation=True,
                    enable_inst_interactivity=False,
                )
                processor = Sam3Processor(
                    sam3_model,
                    resolution=1008,
                    device=self._device,
                    confidence_threshold=0.5,
                )
            self._auto_segment_model = processor
            self._auto_segment_model_name = requested_model
            return self._auto_segment_model, None
        except Exception as exc:
            log(f"Error loading SAM3: {traceback.format_exc()}")
            return None, (
                f"Could not load SAM3 checkpoint '{checkpoint_path}': {exc}"
            )

    @staticmethod
    def _normalize_expected_size(expected_width: int = 0, expected_height: int = 0) -> Optional[tuple[int, int]]:
        width = int(expected_width or 0)
        height = int(expected_height or 0)
        if width > 0 and height > 0:
            return width, height
        return None

    def _load_interactive_image(
        self,
        image_path: str,
        expected_width: int = 0,
        expected_height: int = 0,
    ):
        """Load the image frame that matches what the Qt canvas is displaying."""
        from PIL import ImageOps

        expected_size = self._normalize_expected_size(expected_width, expected_height)
        raw_image = Image.open(image_path)
        transformed = ImageOps.exif_transpose(raw_image).convert("RGB")

        if expected_size is None or transformed.size == expected_size:
            return transformed

        raw_rgb = raw_image.convert("RGB")
        if raw_rgb.size == expected_size:
            log(
                "Interactive SAM frame mismatch after EXIF transpose; "
                f"using raw image frame to match UI size {expected_size} for '{image_path}'"
            )
            return raw_rgb

        raise ValueError(
            "Interactive SAM image frame mismatch. "
            f"UI expects {expected_size}, but backend loaded {transformed.size} "
            f"(raw: {raw_rgb.size}) for '{image_path}'."
        )

    def set_image(
        self,
        image_path: str,
        image_id: str = "",
        model: str = "",
        expected_width: int = 0,
        expected_height: int = 0,
        **_ignored,
    ) -> dict:
        """Set image context for SAM2 interactive predictions."""
        model_name = self._normalize_model_name(model or self.INTERACTIVE_MODEL)
        if model_name != self.INTERACTIVE_MODEL:
            return error_response(
                "set_image is only supported for SAM2 interactive mode.",
                response_type="set_image",
            )

        predictor, load_error = self._load_interactive_model(model_name)
        if predictor is None:
            return error_response(load_error or "Could not load SAM2 model", response_type="set_image")

        if not os.path.isfile(image_path):
            return error_response(f"Image file not found: {image_path}", response_type="set_image")

        try:
            frame = self._load_interactive_image(
                image_path,
                expected_width=expected_width,
                expected_height=expected_height,
            )
            frame_np = np.array(frame)
            predictor.set_image(frame_np)

            self._model_name = model_name
            self._cached_image_id = image_id or image_path
            self._interactive_image_size = tuple(frame.size)

            return ok_response(
                "set_image",
                model=model_name,
                image_id=self._cached_image_id,
                frame_width=int(frame.size[0]),
                frame_height=int(frame.size[1]),
            )
        except Exception as e:
            log(f"Error setting interactive SAM2 image: {traceback.format_exc()}")
            self._cached_image_id = None
            self._interactive_image_size = None
            return error_response(str(e), response_type="set_image")

    def predict_points(
        self,
        points: List[Dict],
        image_id: str = "",
        model: str = "",
        multimask: bool = True,
        decode_mask: bool = True,
    ) -> dict:
        """Predict polygon candidates from point prompts with SAM2."""
        model_name = self._normalize_model_name(model or self.INTERACTIVE_MODEL)
        if model_name != self.INTERACTIVE_MODEL:
            return error_response("predict_points only supports SAM2 interactive mode.", response_type="prediction")

        predictor, load_error = self._load_interactive_model(model_name)
        if predictor is None:
            return error_response(load_error or "Could not load SAM2 model", response_type="prediction")

        if not points:
            return error_response("No points provided", response_type="prediction")

        if image_id and self._cached_image_id and image_id != self._cached_image_id:
            return error_response(
                "Image context mismatch. Call set_image before predict_points.",
                response_type="prediction",
            )

        try:
            point_coords = np.array(
                [[float(p.get("x", 0)), float(p.get("y", 0))] for p in points],
                dtype=np.float32,
            )
            point_labels = np.array(
                [1 if bool(p.get("is_positive", True)) else 0 for p in points],
                dtype=np.int32,
            )

            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=bool(multimask),
            )

            masks_np = np.asarray(masks)
            scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
            if masks_np.ndim == 4:
                masks_np = masks_np[:, 0, :, :]
            elif masks_np.ndim == 2:
                masks_np = np.expand_dims(masks_np, axis=0)

            preferred_point = None
            for p in points:
                if bool(p.get("is_positive", True)):
                    preferred_point = (float(p.get("x", 0)), float(p.get("y", 0)))
                    break
            if preferred_point is None:
                p = points[0]
                preferred_point = (float(p.get("x", 0)), float(p.get("y", 0)))

            predictions = self._process_masks(
                masks_np,
                scores_np,
                decode_mask=decode_mask,
                preferred_point=preferred_point,
            )
            return ok_response("prediction", predictions=predictions)
        except Exception as e:
            log(f"Error during interactive SAM2 point prediction: {traceback.format_exc()}")
            return error_response(str(e), response_type="prediction")

    def predict_box(
        self,
        box: Dict[str, int],
        image_id: str = "",
        model: str = "",
        multimask: bool = True,
        decode_mask: bool = True,
    ) -> dict:
        """Predict polygon candidates from a box prompt with SAM2."""
        model_name = self._normalize_model_name(model or self.INTERACTIVE_MODEL)
        if model_name != self.INTERACTIVE_MODEL:
            return error_response("predict_box only supports SAM2 interactive mode.", response_type="prediction")

        predictor, load_error = self._load_interactive_model(model_name)
        if predictor is None:
            return error_response(load_error or "Could not load SAM2 model", response_type="prediction")

        if image_id and self._cached_image_id and image_id != self._cached_image_id:
            return error_response(
                "Image context mismatch. Call set_image before predict_box.",
                response_type="prediction",
            )

        try:
            x = float(box.get("x", 0))
            y = float(box.get("y", 0))
            width = float(box.get("width", 0))
            height = float(box.get("height", 0))
            sam_box = np.array([x, y, x + width, y + height], dtype=np.float32)

            masks, scores, _ = predictor.predict(
                box=sam_box,
                multimask_output=bool(multimask),
            )

            masks_np = np.asarray(masks)
            scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
            if masks_np.ndim == 4:
                masks_np = masks_np[:, 0, :, :]
            elif masks_np.ndim == 2:
                masks_np = np.expand_dims(masks_np, axis=0)

            predictions = self._process_masks(
                masks_np,
                scores_np,
                decode_mask=decode_mask,
                preferred_point=(x + (width / 2.0), y + (height / 2.0)),
            )
            return ok_response("prediction", predictions=predictions)
        except Exception as e:
            log(f"Error during interactive SAM2 box prediction: {traceback.format_exc()}")
            return error_response(str(e), response_type="prediction")


    def _process_masks(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        decode_mask: bool,
        preferred_point: Optional[tuple[float, float]] = None,
    ) -> List[Dict]:
        """Convert SAM masks to protocol format."""
        import cv2

        predictions: List[Dict[str, object]] = []
        for i in range(len(scores)):
            mask = masks[i]
            score = float(scores[i])
            mask_height, mask_width = mask.shape[:2]

            # Mask → polygon
            binary = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            # Prefer the contour containing the prompt point when available.
            selected_contour = None
            if preferred_point is not None:
                px, py = preferred_point
                containing = [
                    c for c in contours if cv2.pointPolygonTest(c, (px, py), False) >= 0
                ]
                if containing:
                    selected_contour = max(containing, key=cv2.contourArea)

            if selected_contour is None:
                selected_contour = max(contours, key=cv2.contourArea)

            epsilon = 0.002 * cv2.arcLength(selected_contour, True)
            simplified = cv2.approxPolyDP(selected_contour, epsilon, True)
            polygon = []
            for p in simplified:
                x = min(mask_width - 1, max(0, int(p[0][0])))
                y = min(mask_height - 1, max(0, int(p[0][1])))
                if not polygon or polygon[-1] != {"x": x, "y": y}:
                    polygon.append({"x": x, "y": y})
            if len(polygon) >= 2 and polygon[0] == polygon[-1]:
                polygon.pop()
            if len(polygon) < 3:
                continue

            # Bounding box
            x_coords = [p["x"] for p in polygon]
            y_coords = [p["y"] for p in polygon]
            bbox = {
                "x": min(x_coords),
                "y": min(y_coords),
                "width": max(x_coords) - min(x_coords),
                "height": max(y_coords) - min(y_coords),
            }

            pred: Dict[str, object] = {
                "polygon": polygon,
                "bbox": bbox,
                "confidence": score,
                "metadata": {"frame_width": mask_width, "frame_height": mask_height},
            }

            # Optionally encode mask as RLE or base64 PNG
            if decode_mask:
                import base64
                import io
                mask_img = Image.fromarray(binary)
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                pred["mask"] = base64.b64encode(buf.getvalue()).decode("ascii")

            predictions.append(pred)

        # Sort by confidence descending
        predictions.sort(key=lambda p: p["confidence"], reverse=True)
        return predictions

    @staticmethod
    def _positive_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return default

    @classmethod
    def _auto_segment_cpu_max_side(cls, requested_max_side: Any = None) -> int:
        requested = cls._positive_int(requested_max_side, default=0)
        if requested > 0:
            return requested

        raw = os.environ.get("VISOLABEL_SAM3_CPU_MAX_SIDE", "").strip()
        if not raw:
            return cls.AUTO_SEGMENT_CPU_MAX_SIDE
        return cls._positive_int(raw, default=cls.AUTO_SEGMENT_CPU_MAX_SIDE)

    @classmethod
    def _auto_segment_cpu_threads(cls, requested_threads: Any = None) -> int:
        requested = cls._positive_int(requested_threads, default=0)
        if requested > 0:
            return requested

        raw = os.environ.get("VISOLABEL_SAM3_CPU_THREADS", "").strip()
        if raw:
            return cls._positive_int(raw, default=cls.AUTO_SEGMENT_CPU_THREADS)

        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count)

    def _configure_torch_cpu_performance(self, torch_module, requested_threads: Any = None) -> None:
        if torch_module.cuda.is_available():
            return

        threads = self._auto_segment_cpu_threads(requested_threads)
        if threads <= 0:
            return

        try:
            if self._cpu_thread_count != threads:
                torch_module.set_num_threads(threads)
                self._cpu_thread_count = threads
                log(f"SAM3 CPU PyTorch threads set to {threads}")
        except Exception as exc:
            log(f"Could not set SAM3 CPU PyTorch threads to {threads}: {exc}")

        interop_threads = 1
        try:
            if self._cpu_interop_thread_count != interop_threads:
                torch_module.set_num_interop_threads(interop_threads)
                self._cpu_interop_thread_count = interop_threads
                log(f"SAM3 CPU PyTorch interop threads set to {interop_threads}")
        except Exception as exc:
            log(f"Could not set SAM3 CPU PyTorch interop threads to {interop_threads}: {exc}")

        try:
            if hasattr(torch_module.backends, "mkldnn"):
                torch_module.backends.mkldnn.enabled = True
        except Exception:
            pass

        try:
            torch_module.set_flush_denormal(True)
        except Exception:
            pass

    @staticmethod
    def _resize_for_auto_segment(image, device_type: str, cpu_max_side: Any = None):
        """Return an inference image plus coordinate scales back to original size."""
        original_width, original_height = image.size
        if original_width <= 0 or original_height <= 0:
            return image, 1.0, 1.0, (original_width, original_height), image.size

        max_side = 0
        if device_type == "cpu":
            max_side = SAMService._auto_segment_cpu_max_side(cpu_max_side)

        longest_side = max(original_width, original_height)
        if max_side <= 0 or longest_side <= max_side:
            return image, 1.0, 1.0, (original_width, original_height), image.size

        scale = max_side / float(longest_side)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resampling = getattr(Image, "Resampling", Image)
        resample = getattr(resampling, "BILINEAR", Image.BILINEAR)
        resized = image.resize((resized_width, resized_height), resample)
        scale_x = (
            (original_width - 1) / (resized_width - 1)
            if original_width > 1 and resized_width > 1
            else original_width / resized_width
        )
        scale_y = (
            (original_height - 1) / (resized_height - 1)
            if original_height > 1 and resized_height > 1
            else original_height / resized_height
        )
        return resized, scale_x, scale_y, (original_width, original_height), resized.size

    @staticmethod
    def _scale_auto_segment_polygon(
        polygon: list[dict],
        scale_x: float,
        scale_y: float,
        original_width: int,
        original_height: int,
    ) -> list[dict]:
        scaled = []
        for point in polygon:
            x = int(round(float(point["x"]) * scale_x))
            y = int(round(float(point["y"]) * scale_y))
            if original_width > 0:
                x = min(original_width - 1, max(0, x))
            if original_height > 0:
                y = min(original_height - 1, max(0, y))
            if not scaled or scaled[-1] != {"x": x, "y": y}:
                scaled.append({"x": x, "y": y})
        if len(scaled) >= 2 and scaled[0] == scaled[-1]:
            scaled.pop()
        return scaled if len(scaled) >= 3 else []

    def auto_segment(
        self,
        image_path: str,
        image_id: str = "",
        model: str = "",
        text_prompt: str = "",
        confidence_threshold: float = 0.5,
        min_mask_region_area: int = 100,
        cpu_max_side: int = 0,
        cpu_threads: int = 0,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        **kwargs,
    ) -> dict:
        """Automatically segment objects in the image using SAM3 with a text prompt."""
        def progress(step: str, message: str) -> None:
            if progress_callback is not None:
                progress_callback(step, message)

        log(
            "Auto-segment request: "
            f"image_path='{image_path}', model='{model or self.AUTO_SEGMENT_MODEL}', "
            f"text_prompt='{text_prompt}', confidence_threshold={confidence_threshold}, "
            f"min_mask_region_area={min_mask_region_area}, "
            f"cpu_max_side={cpu_max_side or self._auto_segment_cpu_max_side()}, "
            f"cpu_threads={cpu_threads or self._auto_segment_cpu_threads()}"
        )
        try:
            import torch
            self._configure_torch_cpu_performance(torch, requested_threads=cpu_threads)
        except ImportError:
            pass

        progress("load_model", "Loading SAM3 model")
        processor, load_error = self._load_auto_segment_model(model)
        if processor is None:
            return error_response(load_error or "Could not load SAM3 model", response_type="auto_segment")

        try:
            import cv2
            import torch
            from PIL import Image as PILImage

            if not os.path.isfile(image_path):
                return error_response(f"Image file not found: {image_path}", response_type="auto_segment")

            image = PILImage.open(image_path).convert("RGB")
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)

            # Set confidence threshold
            processor.confidence_threshold = confidence_threshold

            # Match the fast official PyTorch CPU path: use autocast on CUDA only.
            device_type = "cuda" if self._device.startswith("cuda") else "cpu"
            if device_type == "cpu":
                try:
                    cv2.setNumThreads(max(1, min(8, self._auto_segment_cpu_threads(cpu_threads))))
                except Exception:
                    pass
            image, scale_x, scale_y, original_size, inference_size = self._resize_for_auto_segment(
                image,
                device_type,
                cpu_max_side,
            )
            original_width, original_height = original_size
            inference_width, inference_height = inference_size
            if inference_size != original_size:
                log(
                    "Resized SAM3 CPU inference image "
                    f"from {original_width}x{original_height} to {inference_width}x{inference_height}"
                )
            log(f"Running SAM3 inference on device_type='{device_type}'")
            cpu_compat = device_type == "cpu" and not torch.cuda.is_available()
            # On pre-Ampere GPUs (e.g. Turing / RTX 20xx, compute capability < 8.0)
            # bfloat16 has no tensor-core support and runs extremely slowly. fp16
            # uses the fast fp16 tensor cores there, so pick the AMP dtype by
            # the actual GPU capability.
            amp_dtype = torch.bfloat16
            if device_type == "cuda":
                try:
                    if torch.cuda.get_device_capability()[0] < 8:
                        amp_dtype = torch.float16
                        log("Using float16 autocast (GPU is pre-Ampere; bfloat16 is slow here).")
                except Exception:
                    pass
            autocast_context = (
                torch.autocast(device_type=device_type, dtype=amp_dtype)
                if device_type == "cuda"
                else nullcontext()
            )
            with _sam3_cpu_compatibility(torch, enabled=cpu_compat):
                with torch.inference_mode():
                    with autocast_context:
                        progress("encode_image", "Encoding image")
                        state = processor.set_image(image)
                        prompt = text_prompt or "object"
                        progress("text_prompt", f'Looking for "{prompt}"')
                        state = processor.set_text_prompt(prompt, state)

            predictions = []
            masks = state.get("masks", None)
            boxes = state.get("boxes", None)
            scores = state.get("scores", None)

            if masks is not None and len(masks) > 0:
                masks_np = masks.squeeze(1).cpu().float().numpy()  # (N, H, W)
                boxes_np = boxes.cpu().float().numpy() if boxes is not None else None
                scores_np = scores.cpu().float().numpy() if scores is not None else None

                for i in range(len(masks_np)):
                    binary = (masks_np[i] > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    largest = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) < min_mask_region_area:
                        continue

                    epsilon = 0.002 * cv2.arcLength(largest, True)
                    simplified = cv2.approxPolyDP(largest, epsilon, True)
                    if len(simplified) < 3:
                        continue

                    inference_polygon = [{"x": int(p[0][0]), "y": int(p[0][1])} for p in simplified]
                    polygon = self._scale_auto_segment_polygon(
                        inference_polygon,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        original_width=original_width,
                        original_height=original_height,
                    )
                    if len(polygon) < 3:
                        continue
                    x_coords = [p["x"] for p in polygon]
                    y_coords = [p["y"] for p in polygon]
                    bbox = {
                        "x": min(x_coords),
                        "y": min(y_coords),
                        "width": max(x_coords) - min(x_coords),
                        "height": max(y_coords) - min(y_coords),
                    }

                    score = float(scores_np[i]) if scores_np is not None else 1.0
                    predictions.append({
                        "polygon": polygon,
                        "bbox": bbox,
                        "confidence": score,
                        "metadata": {
                            "frame_width": original_width,
                            "frame_height": original_height,
                            "inference_width": inference_width,
                            "inference_height": inference_height,
                        },
                    })

            predictions.sort(key=lambda p: p["confidence"], reverse=True)
            log(f"Auto-segment found {len(predictions)} objects for prompt '{text_prompt}'")
            return ok_response("auto_segment", predictions=predictions)

        except ImportError as e:
            log(f"Auto-segment import error: {e}")
            return error_response(
                f"SAM3 dependency missing: {e}",
                response_type="auto_segment",
            )
        except Exception as e:
            log(f"Error during auto-segment: {traceback.format_exc()}")
            return error_response(str(e), response_type="auto_segment")

    def get_models(self) -> dict:
        """Return available models."""
        return ok_response(
            "models",
            models=self.SUPPORTED_MODELS,
            current_model=self._model_name,
        )

    def health(self) -> dict:
        """Health check."""
        model_loaded = bool(self._interactive_predictor is not None or self._auto_segment_model is not None)
        return ok_response(
            "health",
            model_loaded=model_loaded,
            current_model=self._model_name,
            interactive_model_loaded=self._interactive_predictor is not None,
            auto_segment_model_loaded=self._auto_segment_model is not None,
            device=self._device,
        )

    def shutdown(self) -> dict:
        """Clean up and prepare to exit."""
        if self._interactive_predictor is not None:
            del self._interactive_predictor
            self._interactive_predictor = None
        if self._auto_segment_model is not None:
            del self._auto_segment_model
            self._auto_segment_model = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._model_name = None
        self._interactive_model_name = None
        self._auto_segment_model_name = None
        self._cached_image_id = None
        self._interactive_image_size = None
        log("SAM service shutdown complete")
        return ok_response("shutdown")


def run_sam_service(initial_model: str = "") -> None:
    """Main loop for the SAM service."""
    service = SAMService()

    # Optionally pre-load a model
    if initial_model:
        result = service.load_model(initial_model)
        write_json_line(result)
        if result["status"] != "ok":
            log(f"Warning: Failed to pre-load model '{initial_model}'")

    log("SAM service ready, waiting for commands on stdin...")

    while True:
        try:
            request = read_json_line()
            if request is None:
                # EOF - parent process closed stdin
                log("stdin closed, shutting down")
                service.shutdown()
                break

            cmd = request.get("cmd", "")

            if cmd == SAMCommands.LOAD:
                response = service.load_model(request.get("model", ""))
            elif cmd == SAMCommands.SET_IMAGE:
                response = service.set_image(
                    image_path=request.get("image_path", ""),
                    image_id=request.get("image_id", ""),
                    model=request.get("model", ""),
                    expected_width=request.get("expected_width", 0),
                    expected_height=request.get("expected_height", 0),
                )
            elif cmd == SAMCommands.PREDICT_POINTS:
                response = service.predict_points(
                    points=request.get("points", []),
                    image_id=request.get("image_id", ""),
                    model=request.get("model", ""),
                    multimask=request.get("multimask", True),
                    decode_mask=request.get("decode_mask", True),
                )
            elif cmd == SAMCommands.PREDICT_BOX:
                response = service.predict_box(
                    box=request.get("box", {}),
                    image_id=request.get("image_id", ""),
                    model=request.get("model", ""),
                    multimask=request.get("multimask", True),
                    decode_mask=request.get("decode_mask", True),
                )
            elif cmd == SAMCommands.AUTO_SEGMENT:
                response = service.auto_segment(
                    image_path=request.get("image_path", ""),
                    image_id=request.get("image_id", ""),
                    model=request.get("model", ""),
                    text_prompt=request.get("text_prompt", ""),
                    confidence_threshold=request.get("confidence_threshold", 0.5),
                    min_mask_region_area=request.get("min_mask_region_area", 100),
                    cpu_max_side=request.get("cpu_max_side", 0),
                    cpu_threads=request.get("cpu_threads", 0),
                    progress_callback=lambda step, message: write_json_line(progress_response(message, step=step)),
                )
            elif cmd == SAMCommands.GET_MODELS:
                response = service.get_models()
            elif cmd == SAMCommands.HEALTH:
                response = service.health()
            elif cmd == SAMCommands.SHUTDOWN:
                response = service.shutdown()
                write_json_line(response)
                break
            else:
                response = error_response(f"Unknown command: {cmd}")

            write_json_line(response)

        except json.JSONDecodeError as e:
            write_json_line(error_response(f"Invalid JSON: {e}"))
        except Exception as e:
            log(f"Unhandled error: {traceback.format_exc()}")
            write_json_line(error_response(f"Internal error: {e}"))


# Allow importing json at module level for the except clause
import json

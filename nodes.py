# -*- coding: utf-8 -*-
"""h4 Toolkit custom nodes for ComfyUI.

This module implements the Plot node (end-to-end sampler + grid generator with
progress previews) and the Debug-a-tron-3000 adaptive router/inspector. Both
nodes are instrumented with extremely verbose logging so that every action is
visible in the Stability Matrix / ComfyUI console during development.

The implementation leans on existing ComfyUI building blocks (samplers, VAE
utilities, CLIP encoders, etc.) to ensure forward compatibility while keeping
resource usage as low as practical. All helper utilities in this file are kept
lightweight and are re-used by both nodes to honour the Node policy.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
import os
import textwrap
import time
from dataclasses import dataclass, field
import copy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import hashlib
from datetime import datetime

import numpy as np
import torch

from colorama import Fore, Style

try:  # pragma: no cover - optional preview module
    import latent_preview
except ImportError:  # pragma: no cover - preview is optional
    latent_preview = None

try:  # pragma: no cover - optional annotation dependency
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None

import comfy.model_management as model_management
import comfy.samplers as comfy_samplers
import comfy.sd as comfy_sd
import comfy.utils as comfy_utils
import folder_paths
import nodes as comfy_nodes

try:  # pragma: no cover - optional runtime dependency
    from comfy.cli_args import args as comfy_args  # type: ignore[import]
except Exception:  # pragma: no cover - args are optional in tests
    comfy_args = None

try:  # pragma: no cover - optional execution context info
    from comfy_execution.utils import get_executing_context  # type: ignore[import]
except Exception:  # pragma: no cover - execution context disabled
    get_executing_context = None


TOOLKIT_VERSION = "1.3.0"
PLOT_NODE_VERSION = "1.3.0"
DEBUG_NODE_VERSION = "2.1.0"


TRACE_ENABLED = os.getenv("H4_TOOLKIT_TRACE", "0") not in {"0", "false", "False", ""}


def _discover_sampler_choices() -> List[str]:
    try:
        raw = getattr(comfy_samplers.KSampler, "SAMPLERS", None) or []
    except Exception:  # pragma: no cover - environment specific
        raw = []

    names: List[str] = []
    if isinstance(raw, dict):
        names = list(raw.keys())
    else:
        for item in raw:
            if isinstance(item, (list, tuple)) and item:
                names.append(str(item[0]))
            elif isinstance(item, str):
                names.append(item)

    if not names:
        names = [
            "euler",
            "euler_ancestral",
            "lms",
            "heun",
            "dpmpp_2m",
            "dpmpp_2m_sde",
            "dpmpp_sde",
            "ddim",
            "ddpm",
        ]

    seen = set()
    ordered: List[str] = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _discover_scheduler_choices() -> List[str]:
    try:
        raw = getattr(comfy_samplers, "SCHEDULERS", None) or []
    except Exception:  # pragma: no cover - environment specific
        raw = []

    names: List[str] = []
    if isinstance(raw, dict):
        names = list(raw.keys())
    else:
        for item in raw:
            if isinstance(item, (list, tuple)) and item:
                names.append(str(item[0]))
            elif isinstance(item, str):
                names.append(item)

    if not names:
        names = [
            "normal",
            "simple",
            "karras",
            "exponential",
            "sgm_uniform",
            "lognormal",
        ]

    seen = set()
    ordered: List[str] = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


SAMPLER_CHOICES = tuple(_discover_sampler_choices())
SAMPLER_LOOKUP = {
    key: choice
    for choice in SAMPLER_CHOICES
    for key in {
        choice.lower(),
        choice.replace(" ", "_").lower(),
        choice.replace("_", "").lower(),
    }
}


SCHEDULER_CHOICES = tuple(_discover_scheduler_choices())
SCHEDULER_LOOKUP = {
    key: choice
    for choice in SCHEDULER_CHOICES
    for key in {
        choice.lower(),
        choice.replace(" ", "_").lower(),
        choice.replace("_", "").lower(),
    }
}


def _invoke_sampler_with_compatibility(
    logger: ToolkitLogger, sampler_kwargs: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    sampler_fn = getattr(comfy_samplers, "sample", None)
    if sampler_fn is None:
        raise RuntimeError("ComfyUI sampler entrypoint is unavailable")

    try:
        signature = inspect.signature(sampler_fn)
    except (TypeError, ValueError):
        signature = None

    rename_priority = {
        "steps": ("num_steps", "step_count", "steps"),
        "cfg": ("cfg", "cfg_scale", "guidance_scale"),
        "sampler_name": ("sampler_name", "sampler"),
        "scheduler_name": ("scheduler_name", "scheduler"),
        "denoise": ("denoise", "denoise_strength"),
        "positive": ("positive", "positive_conditioning"),
        "negative": ("negative", "negative_conditioning"),
        "latent": ("latent_image", "latent"),
        "callback": ("callback", "step_callback"),
        "device": ("device", "model_device"),
        "sigmas": ("sigmas", "sigma_schedule"),
        "model_sampling": ("model_sampling", "model_sampler"),
    }

    if signature is None:
        filtered_direct = {key: value for key, value in sampler_kwargs.items() if value is not None}
        try:
            return sampler_fn(**filtered_direct)
        except TypeError as exc:  # pragma: no cover - best effort fallback
            logger.warn(f"Sampler call signature mismatch; retrying with raw kwargs: {exc}")
            return sampler_fn(**sampler_kwargs)

    params = signature.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    filtered_kwargs: Dict[str, Any] = {}
    renamed: Dict[str, str] = {}
    dropped: List[str] = []

    for key, value in sampler_kwargs.items():
        aliases = rename_priority.get(key, (key,))
        chosen: Optional[str] = None
        for alias in aliases:
            if alias in params:
                chosen = alias
                break
        if chosen is None and key in params:
            chosen = key
        if chosen is not None:
            filtered_kwargs[chosen] = value
            if chosen != key:
                renamed[key] = chosen
        elif accepts_kwargs:
            filtered_kwargs[key] = value
        else:
            dropped.append(key)

    missing_required = [
        name
        for name, param in params.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and param.default is inspect._empty
        and name not in filtered_kwargs
    ]

    if dropped:
        logger.trace(
            "Sampler compatibility shim dropped unsupported kwargs: " + ", ".join(sorted(dropped))
        )
    if renamed:
        details = ", ".join(f"{orig}->{alias}" for orig, alias in sorted(renamed.items()))
        logger.trace(f"Sampler compatibility shim remapped kwargs: {details}")
    if missing_required:
        logger.warn(
            "Sampler compatibility shim missing required parameters after filtering: "
            + ", ".join(missing_required)
        )
        # fall back to the raw call so the original error bubbles with context
        return sampler_fn(**sampler_kwargs)

    try:
        return sampler_fn(**filtered_kwargs)
    except TypeError as exc:
        logger.warn(f"Sampler compatibility shim retry failed: {exc}")
        return sampler_fn(**sampler_kwargs)


AXIS_PRESETS = (
    "prompt",
    "checkpoint",
    "sampler",
    "scheduler",
    "cfg",
    "steps",
    "denoise",
    "seed",
    "lora",
    "none",
)


PLOT_WIDGET_TOOLTIPS = {
    "seed": "Base seed for the first run; axis overrides can provide per-cell values.",
    "steps": "Default sampler steps; supply override lines on either axis to vary.",
    "cfg": "Classifier-free guidance weight applied unless overridden.",
    "denoise": "Amount of denoising to apply. 1.0 = full denoise, lower to blend latent inputs.",
    "clip_skip": "How many CLIP layers to skip. Higher values can emphasise structure over detail.",
    "sampler_name": "Sampler selection for baseline runs. Axis entries can swap samplers per cell.",
    "scheduler_name": "Scheduler curve used to distribute steps.",
    "width": "Pixel width for new latent creation when no input latent/image is supplied.",
    "height": "Pixel height for new latent creation when no input latent/image is supplied.",
    "checkpoint": "Starting checkpoint. Axis entries using the checkpoint preset can override per cell.",
    "vae_name": "Override VAE to decode with. Leave on <checkpoint> to use the base bundle.",
    "clip_text_name": "Override the text encoder CLIP. Uses checkpoint CLIP when left on <checkpoint>.",
    "clip_vision_name": "Optional vision CLIP for dual models. Leave on <checkpoint> when unused.",
    "positive_prompt": "Base positive prompt shared by all runs unless suffixed by axis instructions.",
    "negative_prompt": "Negative prompt applied when CFG > 1.0.",
    "x_axis_mode": "How to interpret each line listed in X axis values.",
    "x_axis_values": "One entry per line. Accepts comma separated lists for convenience.",
    "y_axis_mode": "How to interpret each line listed in Y axis values.",
    "y_axis_values": "One entry per line. Accepts comma separated lists for convenience.",
}


DEBUG_WIDGET_TOOLTIPS = {
    "mode": "Monitor keeps signals internal. Passthrough relays inputs to outputs while logging.",
    "go_ultra": "Ask the router to GO PLUS ULTRA?! and unlock a diagnostics panel packed with snapshots, previews, JSON logs, and anomaly detectors.",
}

ULTRA_CONTROL_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "ultra_capture_first_step",
        "type": "BOOLEAN",
        "default": True,
        "label": "Snapshot: first step",
        "tooltip": "Capture analytics for the earliest latent frame observed (or current latent when history is unavailable).",
    },
    {
        "name": "ultra_capture_mid_step",
        "type": "BOOLEAN",
        "default": False,
        "label": "Snapshot: midpoint",
        "tooltip": "When latent history is provided, also analyse the midpoint frame for drift checks.",
    },
    {
        "name": "ultra_capture_last_step",
        "type": "BOOLEAN",
        "default": True,
        "label": "Snapshot: final step",
        "tooltip": "Capture analytics for the most recent latent frame to spot end-of-run anomalies.",
    },
    {
        "name": "ultra_preview_images",
        "type": "BOOLEAN",
        "default": True,
        "label": "Generate preview image",
        "tooltip": "Attempt to decode the analysed latent into a quick preview image when a VAE is connected.",
    },
    {
        "name": "ultra_json_log",
        "type": "BOOLEAN",
        "default": True,
        "label": "Write JSON session log",
        "tooltip": "Serialise the captured diagnostics to a JSON file under ComfyUI's temp directory for later comparison.",
    },
    {
        "name": "ultra_highlight_missing_conditioning",
        "type": "BOOLEAN",
        "default": True,
        "label": "Highlight missing conditioning",
        "tooltip": "Warn when positive/negative or auxiliary conditioning branches are absent to prevent silent misconfigurations.",
    },
    {
        "name": "ultra_token_preview",
        "type": "BOOLEAN",
        "default": False,
        "label": "Preview conditioning tokens",
        "tooltip": "Attempt to echo prompt fragments and key embedding stats for the first few conditioning entries.",
    },
    {
        "name": "ultra_latent_anomaly_checks",
        "type": "BOOLEAN",
        "default": True,
        "label": "Check for latent anomalies",
        "tooltip": "Scan tensors for NaN/Inf values and suspicious variance spikes to surface stability issues early.",
    },
    {
        "name": "ultra_model_diff_tracking",
        "type": "BOOLEAN",
        "default": False,
        "label": "Fingerprint attached models",
        "tooltip": "Hash model and LoRA parameter signatures so you can verify which weights were active this run.",
    },
    {
        "name": "ultra_watch_expression",
        "type": "STRING",
        "default": "",
        "label": "Custom watch expression",
        "tooltip": "Evaluate a Python expression against the inbound payloads (available globals: torch, np, slot names). Leave blank to disable.",
    },
    {
        "name": "ultra_cache_artifacts",
        "type": "BOOLEAN",
        "default": True,
        "label": "Persist preview artifacts",
        "tooltip": "Keep generated preview images and JSON logs on disk instead of treating them as temporary diagnostics.",
    },
)

ULTRA_CONTROL_DEFAULTS: Dict[str, Any] = {
    control["name"]: control["default"] for control in ULTRA_CONTROL_DEFINITIONS
}


DEBUG_SLOT_DEFINITIONS: Tuple[Tuple[str, str, str], ...] = (
    ("model_in", "Model", "MODEL"),
    ("clip_in", "CLIP", "CLIP"),
    ("clip_vision_in", "CLIP_Vision", "CLIP_VISION"),
    ("vae_in", "VAE", "VAE"),
    ("conditioning_in", "Conditioning", "CONDITIONING"),
    ("conditioning_positive_in", "Positive", "CONDITIONING"),
    ("conditioning_negative_in", "Negative", "CONDITIONING"),
    ("latent_in", "Latent", "LATENT"),
    ("image_in", "Image", "IMAGE"),
    ("mask_in", "Mask", "MASK"),
)

DEBUG_SLOT_TOOLTIPS: Dict[str, str] = {
    "model_in": "Connect any diffusion model blob to inspect or forward downstream.",
    "clip_in": "Attach a CLIP text encoder payload for logging and pass-through.",
    "clip_vision_in": "Optional CLIP vision encoder payload.",
    "vae_in": "Supply a VAE instance for inspection or passthrough.",
    "conditioning_in": "Generic conditioning list or bundle.",
    "conditioning_positive_in": "Positive conditioning branch when separated upstream.",
    "conditioning_negative_in": "Negative conditioning branch when separated upstream.",
    "latent_in": "Latent dictionary (samples / noise etc.).",
    "image_in": "Images or batches headed into the debugger.",
    "mask_in": "Mask tensor payloads to monitor or route.",
}

ROUTER_BRANCH_SUFFIXES: Tuple[str, ...] = ("_a", "_b")
ROUTER_BRANCH_DISPLAY_NAMES: Dict[str, str] = {
    "_a": "Branch A",
    "_b": "Branch B",
}

ROUTER_SLOT_DEFINITIONS: Tuple[Tuple[str, str, str], ...] = tuple(
    (
        f"{slot_name}{suffix}",
        f"{ROUTER_BRANCH_DISPLAY_NAMES[suffix]} · {display_name}",
        slot_type,
    )
    for suffix in ROUTER_BRANCH_SUFFIXES
    for slot_name, display_name, slot_type in DEBUG_SLOT_DEFINITIONS
)

ROUTER_SLOT_TOOLTIPS: Dict[str, str] = {}
for slot_name, display_name, _slot_type in DEBUG_SLOT_DEFINITIONS:
    base_tooltip = DEBUG_SLOT_TOOLTIPS.get(slot_name)
    for suffix in ROUTER_BRANCH_SUFFIXES:
        branch_label = ROUTER_BRANCH_DISPLAY_NAMES[suffix]
        tooltip = base_tooltip
        if tooltip:
            tooltip = f"{tooltip} ({branch_label})"
        else:
            tooltip = f"{branch_label}: {display_name} slot"
        ROUTER_SLOT_TOOLTIPS[f"{slot_name}{suffix}"] = tooltip

ROUTER_VAE_MODEL_FALLBACKS: Dict[str, str] = {
    f"vae_in{suffix}": f"model_in{suffix}"
    for suffix in ROUTER_BRANCH_SUFFIXES
}

ROUTER_SLOT_RETURN_TYPES: Tuple[str, ...] = tuple(slot[2] for slot in ROUTER_SLOT_DEFINITIONS)
ROUTER_SLOT_RETURN_NAMES: Tuple[str, ...] = tuple(slot[1] for slot in ROUTER_SLOT_DEFINITIONS)

DEBUG_SLOT_RETURN_TYPES: Tuple[str, ...] = tuple(slot[2] for slot in DEBUG_SLOT_DEFINITIONS)
DEBUG_SLOT_RETURN_NAMES: Tuple[str, ...] = tuple(slot[1] for slot in DEBUG_SLOT_DEFINITIONS)

CONDITIONING_SLOT_LABELS: Dict[str, str] = {
    "conditioning_in": "Generic conditioning",
    "conditioning_positive_in": "Positive conditioning",
    "conditioning_negative_in": "Negative conditioning",
}


class ToolkitLogger:
    def __init__(self, namespace: str, *, enable_colour: bool = True) -> None:
        self.namespace = namespace
        self.enable_colour = enable_colour

    def _emit(self, level: str, colour: str, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        prefix = f"[{timestamp}] [h4 Toolkit::{self.namespace}] {level:>5}"
        if self.enable_colour:
            print(f"{colour}{prefix} :: {message}{Style.RESET_ALL}")
        else:
            print(f"{prefix} :: {message}")

    def trace(self, message: str) -> None:
        if TRACE_ENABLED:
            self._emit("TRACE", Style.DIM, message)

    def info(self, message: str) -> None:
        self._emit("INFO", Style.BRIGHT, message)

    def warn(self, message: str) -> None:
        self._emit("WARN", Fore.YELLOW, message)

    def error(self, message: str) -> None:
        self._emit("ERROR", Fore.RED, message)


GLOBAL_LOGGER = ToolkitLogger("h4_ToolKit")


@dataclass
class AxisDescriptor:
    """Represents a user-specified modifier from the X/Y axis panels."""

    source_label: str
    checkpoint: Optional[str] = None
    prompt_suffix: str = ""
    loras: List[Tuple[str, float]] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        pieces: List[str] = []
        if self.checkpoint:
            pieces.append(f"checkpoint={self.checkpoint}")
        if self.prompt_suffix:
            pieces.append(f"prompt+={self.prompt_suffix}")
        for name, strength in self.loras:
            pieces.append(f"lora={name}@{strength:.2f}")
        for key, value in self.overrides.items():
            pieces.append(f"{key}={value}")
        return " | ".join(pieces) if pieces else "identity"


@dataclass
class GenerationPlan:
    """Concrete execution plan for a single plot cell."""

    label: str
    checkpoint_name: Optional[str]
    prompt_suffix: str
    loras: List[Tuple[str, float]] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "checkpoint": self.checkpoint_name,
            "prompt_suffix": self.prompt_suffix,
            "loras": [
                {"name": name, "strength": strength} for name, strength in self.loras
            ],
            "overrides": self.overrides,
        }


def _split_lines(value: str) -> List[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _normalise_token(token: str) -> str:
    return token.replace("\\", "/").lower().strip()


def _resolve_asset_path(category: str, identifier: str) -> Optional[str]:
    if not identifier:
        return None
    candidate = identifier.strip()
    if os.path.isabs(candidate):
        return candidate if os.path.exists(candidate) else None
    try:
        return folder_paths.get_full_path(category, candidate)
    except Exception:
        return None


def resolve_checkpoint_name(token: str) -> Optional[str]:
    candidates_raw = folder_paths.get_filename_list("checkpoints")
    candidates = list(candidates_raw or [])
    if not token:
        return None
    if os.path.isabs(token) and os.path.exists(token):
        return token
    token_norm = _normalise_token(token)
    direct_map = {_normalise_token(item): item for item in candidates}
    if token_norm in direct_map:
        return direct_map[token_norm]
    stripped = token_norm
    for suffix in (".safetensors", ".ckpt", ".pt"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break
    best_match: Optional[str] = None
    for item in candidates:
        normalised = _normalise_token(item)
        name_without_ext = normalised.rsplit(".", 1)[0]
        if normalised.endswith(token_norm) or name_without_ext.endswith(stripped):
            best_match = item
            break
    return best_match


def resolve_lora_name(token: str) -> Optional[str]:
    candidates_raw = folder_paths.get_filename_list("loras")
    candidates = list(candidates_raw or [])
    if not token:
        return None
    if os.path.isabs(token) and os.path.exists(token):
        return token
    token_norm = _normalise_token(token)
    direct_map = {_normalise_token(item): item for item in candidates}
    if token_norm in direct_map:
        return direct_map[token_norm]
    stripped = token_norm.rsplit("/", 1)[-1]
    for item in candidates:
        if _normalise_token(item).endswith(stripped):
            return item
    return None


def _callable_or_value(candidate: Any) -> Any:
    torch_nn = getattr(torch, "nn", None)
    module_cls = getattr(torch_nn, "Module", None) if torch_nn is not None else None
    if module_cls is not None and isinstance(candidate, module_cls):
        return candidate
    if callable(candidate):
        call_signature: Optional[inspect.Signature]
        try:
            call_signature = inspect.signature(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            call_signature = None
        if call_signature is not None:
            requires_arguments = any(
                param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                and param.default is inspect._empty
                for param in call_signature.parameters.values()
            )
            has_var_args = any(
                param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
                for param in call_signature.parameters.values()
            )
            if requires_arguments:
                return candidate
            if has_var_args:
                return candidate
        try:
            return candidate()  # type: ignore[misc]
        except TypeError:
            return None
    return candidate


def _extract_model_sampling(model: Any) -> Optional[Any]:
    if model is None:
        return None

    attribute_chains = (
        ("model_sampling",),
        ("model", "model_sampling"),
        ("model", "model", "model_sampling"),
        ("model", "inner_model", "model_sampling"),
        ("inner_model", "model_sampling"),
        ("clip", "model_sampling"),
        ("latent_format", "model_sampling"),
        ("model", "latent_format", "model_sampling"),
        ("model", "model", "latent_format", "model_sampling"),
        ("diffusion_model", "model_sampling"),
        ("diffusion_model", "latent_format", "model_sampling"),
        ("inner_model", "latent_format", "model_sampling"),
    )

    for chain in attribute_chains:
        current = model
        for attr in chain:
            if current is None:
                break
            current = getattr(current, attr, None)
        else:
            value = _callable_or_value(current)
            if value is not None:
                return value

    for helper_name in ("get_model_sampling", "model_sampling_func"):
        helper = getattr(model, helper_name, None)
        value = _callable_or_value(helper)
        if value is not None:
            return value

    fallback = getattr(model, "model_sampling", None)
    return _callable_or_value(fallback)


def _unwrap_sampler_candidate(candidate: Any) -> Optional[Any]:
    if candidate is None:
        return None
    if hasattr(candidate, "sample") and callable(getattr(candidate, "sample")):
        return candidate
    if isinstance(candidate, (list, tuple, set)):
        for item in candidate:
            resolved = _unwrap_sampler_candidate(item)
            if resolved is not None:
                return resolved
        return None
    if isinstance(candidate, dict):
        for key in ("sampler", "sampler_obj", "sampler_object", "sampler_instance"):
            if key in candidate:
                resolved = _unwrap_sampler_candidate(candidate[key])
                if resolved is not None:
                    return resolved
        return None
    return None


def _call_with_available_arguments(target: Any, available: Dict[str, Any]) -> Optional[Any]:
    if not callable(target):
        return None
    try:
        signature = inspect.signature(target)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        try:
            return target()  # type: ignore[misc]
        except TypeError:
            return None

    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.kind == param.VAR_POSITIONAL:
            continue
        if param.kind == param.VAR_KEYWORD:
            kwargs.update({key: value for key, value in available.items() if value is not None})
            continue
        value = available.get(name)
        if value is None:
            if param.default is inspect._empty:
                return None
            continue
        if param.kind == param.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value

    try:
        return target(*args, **kwargs)  # type: ignore[misc]
    except TypeError:
        return None


def _resolve_sampler_object(
    logger: ToolkitLogger,
    sampler_name: str,
    scheduler_name: str,
    model: Any,
    *,
    device: Optional[torch.device] = None,
    model_sampling: Optional[Any] = None,
    model_sampling_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    resolved_available: Dict[str, Any] = {
        "sampler_name": sampler_name,
        "sampler": sampler_name,
        "name": sampler_name,
        "scheduler_name": scheduler_name,
        "scheduler": scheduler_name,
        "schedule_name": scheduler_name,
        "model": model,
        "model_sampling": model_sampling,
        "model_type": (model_sampling_meta or {}).get("model_type"),
        "sigma_min": (model_sampling_meta or {}).get("sigma_min"),
        "sigma_max": (model_sampling_meta or {}).get("sigma_max"),
        "sigma_data": (model_sampling_meta or {}).get("sigma_data"),
        "sigma_schedule": (model_sampling_meta or {}).get("sigma_schedule"),
        "schedule": (model_sampling_meta or {}).get("schedule"),
        "device": device,
    }
    resolved_available = {key: value for key, value in resolved_available.items() if value is not None}

    def attempt_target(target: Any) -> Optional[Any]:
        candidate = _call_with_available_arguments(target, resolved_available)
        sampler_object = _unwrap_sampler_candidate(candidate)
        return sampler_object

    # Direct module-level factory functions
    factory_candidates: List[Any] = []
    preferred_names = {
        "create_sampler",
        "build_sampler",
        "make_sampler",
        "sampler_from_config",
    }
    for name in preferred_names:
        attr = getattr(comfy_samplers, name, None)
        if callable(attr):
            factory_candidates.append(attr)

    # Objects providing factory methods
    for attr_name in dir(comfy_samplers):
        lower_name = attr_name.lower()
        if "sampler" not in lower_name:
            continue
        if lower_name in {"sample", "sample_inner", "sampler"}:
            continue
        attr = getattr(comfy_samplers, attr_name)
        if callable(attr):
            factory_candidates.append(attr)
            continue
        for method_name in ("create_sampler", "build_sampler", "make_sampler"):
            method = getattr(attr, method_name, None)
            if callable(method):
                factory_candidates.append(method)

    seen_factories: List[Any] = []
    for factory in factory_candidates:
        if factory in seen_factories:
            continue
        seen_factories.append(factory)
        try:
            sampler_object = attempt_target(factory)
        except Exception as exc:  # pragma: no cover - environment specific
            logger.trace(
                f"Sampler factory '{getattr(factory, '__name__', repr(factory))}' failed: {exc}"
            )
            continue
        if sampler_object is not None:
            return sampler_object

    # Model-provided helpers
    for method_name in (
        "create_sampler",
        "make_sampler",
        "build_sampler",
        "get_sampler",
        "sampler",
        "sampler_factory",
    ):
        method = getattr(model, method_name, None)
        if method is None or not callable(method):
            continue
        try:
            sampler_object = attempt_target(method)
        except Exception as exc:  # pragma: no cover - environment specific
            logger.trace(
                f"Model sampler helper '{method_name}' failed: {exc}"
            )
            continue
        if sampler_object is not None:
            return sampler_object

    return None


def parse_axis_entries(mode: str, raw_text: str) -> List[AxisDescriptor]:
    """Parses axis definitions into typed descriptors according to the selected preset."""

    entries: List[AxisDescriptor] = []
    lines = _split_lines(raw_text)
    if lines:
        GLOBAL_LOGGER.trace(
            f"Parsing axis entries for mode={mode} ({len(lines)} entries)"
        )

    def _append_descriptor(descriptor: AxisDescriptor) -> None:
        entries.append(descriptor)
        GLOBAL_LOGGER.trace(f"Axis parsed -> {descriptor.describe()}")

    if not lines:
        return entries

    tokenizing_modes = {
        "checkpoint",
        "sampler",
        "scheduler",
        "cfg",
        "steps",
        "denoise",
        "seed",
        "lora",
    }

    for raw_line in lines:
        segments = [raw_line]
        if mode in tokenizing_modes:
            exploded = [segment.strip() for segment in raw_line.split(",") if segment.strip()]
            if exploded:
                segments = exploded

        for entry in segments:
            lowered = entry.lower()
            if lowered in ("none", "base", "default", "-"):
                _append_descriptor(AxisDescriptor(source_label=entry))
                continue

            if mode == "prompt":
                _append_descriptor(AxisDescriptor(source_label=entry, prompt_suffix=entry))
                continue

            if mode == "checkpoint":
                token = entry
                if lowered.startswith("checkpoint:"):
                    token = entry.split(":", 1)[1].strip()
                resolved = resolve_checkpoint_name(token)
                if resolved:
                    _append_descriptor(AxisDescriptor(source_label=entry, checkpoint=resolved))
                else:
                    GLOBAL_LOGGER.warn(f"Checkpoint '{entry}' not found; skipping")
                continue

            if mode == "cfg":
                try:
                    value = round(float(entry), 3)
                    _append_descriptor(
                        AxisDescriptor(
                            source_label=entry,
                            overrides={"cfg": value},
                        )
                    )
                except ValueError:
                    GLOBAL_LOGGER.warn(f"Invalid CFG value '{entry}'")
                continue

            if mode == "steps":
                try:
                    value = max(1, int(float(entry)))
                    _append_descriptor(
                        AxisDescriptor(
                            source_label=entry,
                            overrides={"steps": value},
                        )
                    )
                except ValueError:
                    GLOBAL_LOGGER.warn(f"Invalid steps value '{entry}'")
                continue

            if mode == "sampler":
                candidate = entry.strip()
                lookup = SAMPLER_LOOKUP.get(candidate.lower())
                if lookup:
                    _append_descriptor(AxisDescriptor(source_label=entry, overrides={"sampler": lookup}))
                else:
                    GLOBAL_LOGGER.warn(f"Sampler '{entry}' is not recognised; skipping")
                continue

            if mode == "scheduler":
                candidate = entry.strip()
                lookup = SCHEDULER_LOOKUP.get(candidate.lower())
                if lookup:
                    _append_descriptor(
                        AxisDescriptor(source_label=entry, overrides={"scheduler": lookup})
                    )
                else:
                    GLOBAL_LOGGER.warn(f"Scheduler '{entry}' is not recognised; skipping")
                continue

            if mode == "denoise":
                try:
                    value = min(1.0, max(0.0, float(entry)))
                    _append_descriptor(
                        AxisDescriptor(source_label=entry, overrides={"denoise": round(value, 3)})
                    )
                except ValueError:
                    GLOBAL_LOGGER.warn(f"Invalid denoise value '{entry}'")
                continue

            if mode == "seed":
                try:
                    value = int(float(entry))
                    _append_descriptor(
                        AxisDescriptor(source_label=entry, overrides={"seed": value})
                    )
                except ValueError:
                    GLOBAL_LOGGER.warn(f"Invalid seed '{entry}'")
                continue

            if mode == "lora":
                payload = entry
                if payload.lower().startswith("lora:"):
                    payload = payload.split(":", 1)[1]
                strength = 1.0
                name_part = payload
                if "@" in payload:
                    name_part, strength_part = payload.split("@", 1)
                    try:
                        strength = float(strength_part)
                    except ValueError:
                        strength = 1.0
                elif "|" in payload:
                    name_part, strength_part = payload.split("|", 1)
                    try:
                        strength = float(strength_part)
                    except ValueError:
                        strength = 1.0
                resolved = resolve_lora_name(name_part.strip())
                if resolved:
                    _append_descriptor(
                        AxisDescriptor(
                            source_label=entry,
                            loras=[(resolved, strength)],
                        )
                    )
                else:
                    GLOBAL_LOGGER.warn(f"LoRA '{entry}' not found; skipping")
                continue

            if mode == "none":
                _append_descriptor(AxisDescriptor(source_label=entry))
                continue

            # Fallback to prompt behaviour when mode is unknown
            _append_descriptor(AxisDescriptor(source_label=entry, prompt_suffix=entry))

    return entries


def build_generation_matrix(
    base_checkpoint: Optional[str],
    x_descriptors: List[AxisDescriptor],
    y_descriptors: List[AxisDescriptor],
) -> List[GenerationPlan]:
    """Computes the cartesian product of axis descriptors into concrete plans."""

    def expand(axis_desc: List[AxisDescriptor]) -> List[List[AxisDescriptor]]:
        return [[desc] for desc in axis_desc] or [[AxisDescriptor(source_label="∅")]]

    runs: List[GenerationPlan] = []
    for y_bundle in expand(y_descriptors):
        for x_bundle in expand(x_descriptors):
            combined = x_bundle + y_bundle
            checkpoint_name = base_checkpoint
            suffixes: List[str] = []
            loras: List[Tuple[str, float]] = []
            label_parts: List[str] = []
            overrides: Dict[str, Any] = {}
            for descriptor in combined:
                label_parts.append(descriptor.describe())
                if descriptor.checkpoint:
                    checkpoint_name = descriptor.checkpoint
                if descriptor.prompt_suffix:
                    suffixes.append(descriptor.prompt_suffix)
                if descriptor.loras:
                    loras.extend(descriptor.loras)
                if descriptor.overrides:
                    overrides.update(descriptor.overrides)
            label = " | ".join([part for part in label_parts if part != "identity"]) or "base"
            plan = GenerationPlan(
                label=label,
                checkpoint_name=checkpoint_name,
                prompt_suffix="\n".join(suffixes) if suffixes else "",
                loras=loras,
                overrides=overrides,
            )
            runs.append(plan)
            GLOBAL_LOGGER.trace(
                f"Planned run: checkpoint={plan.checkpoint_name}, prompt+={plan.prompt_suffix!r}, loras={plan.loras}, overrides={plan.overrides}"
            )
    return runs


def auto_square_layout(total_images: int) -> Tuple[int, int]:
    """Find a grid layout that is as square as possible."""

    if total_images <= 0:
        return (1, 1)
    root = int(math.sqrt(total_images))
    for columns in range(root, 0, -1):
        if total_images % columns == 0:
            rows = total_images // columns
            return (max(1, rows), max(1, columns))
    columns = max(1, root)
    rows = math.ceil(total_images / columns)
    return (rows, columns)


def compose_image_grid(
    images: List[torch.Tensor],
    rows_hint: Optional[int] = None,
    cols_hint: Optional[int] = None,
) -> torch.Tensor:
    """Stack a list of decoded image tensors into a single grid image."""

    if not images:
        raise ValueError("compose_image_grid called with no images")
    def to_bhwc(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            # Attempt to detect BCHW vs BHWC based on channel dimension size.
            if tensor.shape[-1] <= 4:
                return tensor
            return tensor.permute(0, 2, 3, 1)
        if tensor.ndim == 3:
            if tensor.shape[-1] <= 4:
                return tensor.unsqueeze(0)
            if tensor.shape[0] <= 4:
                return tensor.permute(1, 2, 0).unsqueeze(0)
        raise ValueError(f"Unsupported tensor shape for grid composition: {tuple(tensor.shape)}")

    batched: List[torch.Tensor] = []
    for tensor in images:
        normalised = to_bhwc(tensor)
        for item in normalised:
            batched.append(item)

    total = len(batched)
    if rows_hint is not None and cols_hint is not None:
        rows, cols = rows_hint, cols_hint
    else:
        rows, cols = auto_square_layout(total)
    GLOBAL_LOGGER.trace(
        f"Composing grid with {total} tiles -> layout {rows}x{cols}"
    )
    sample_shape = batched[0].shape
    height, width, channels = sample_shape[0], sample_shape[1], sample_shape[2]
    grid = torch.zeros(
        (rows * height, cols * width, channels),
        dtype=batched[0].dtype,
        device=batched[0].device,
    )
    for idx, tile in enumerate(batched):
        if tile.shape != sample_shape:
            raise ValueError(
                f"Grid tile shape mismatch: expected {sample_shape}, received {tuple(tile.shape)}"
            )
        r = idx // cols
        c = idx % cols
        grid[
            r * height : (r + 1) * height,
            c * width : (c + 1) * width,
            :,
        ] = tile
    return grid.unsqueeze(0)


def annotate_grid_image(
    grid_tensor: torch.Tensor,
    plans: List[GenerationPlan],
    rows: int,
    cols: int,
    logger: ToolkitLogger,
) -> torch.Tensor:
    if Image is None or ImageDraw is None or ImageFont is None:
        logger.trace("Skipping grid annotation; Pillow not available")
        return grid_tensor
    if not plans:
        return grid_tensor

    try:
        device = grid_tensor.device
        grid_cpu = grid_tensor.detach().to("cpu")
        base = grid_cpu[0].float().clamp(0.0, 1.0)
        image_array = (base.numpy() * 255.0).astype(np.uint8)
        image = Image.fromarray(image_array)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warn(f"Unable to prepare grid for annotation: {exc}")
        return grid_tensor

    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    cell_width = image.width // max(1, cols)
    cell_height = image.height // max(1, rows)
    max_chars = max(8, cell_width // 9)

    for index, plan in enumerate(plans):
        label_raw = plan.label or f"Run {index + 1}"
        label_wrapped = textwrap.wrap(label_raw, width=max_chars)
        if not label_wrapped:
            label_wrapped = [label_raw]
        label_text = "\n".join(label_wrapped[:3])

        text_region_width = cell_width - 12
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), label_text, font=font, spacing=2)
            text_width = min(text_region_width, bbox[2] - bbox[0])
            text_height = bbox[3] - bbox[1]
        else:  # pragma: no cover - compatibility fallback
            text_width, text_height = draw.textsize(label_text, font=font, spacing=2)

        padding_x = 8
        padding_y = 8
        r = index // max(1, cols)
        c = index % max(1, cols)
        origin_x = c * cell_width + padding_x
        origin_y = r * cell_height + padding_y
        background_rect = (
            origin_x - 4,
            origin_y - 4,
            origin_x + text_width + 8,
            origin_y + text_height + 8,
        )

        try:
            draw.rectangle(background_rect, fill=(0, 0, 0, 160))
            draw.text((origin_x, origin_y), label_text, fill=(255, 255, 255, 255), font=font, spacing=2)
        except Exception as exc:  # pragma: no cover - drawing fallback
            logger.warn(f"Failed to annotate grid cell {index + 1}: {exc}")
            break

    annotated_array = np.asarray(image, dtype=np.float32) / 255.0
    annotated_tensor = torch.from_numpy(annotated_array).to(grid_tensor.dtype).unsqueeze(0)
    return annotated_tensor.to(device)


def _ensure_latent_dict(latent: Any) -> Dict[str, torch.Tensor]:
    if isinstance(latent, dict):
        return latent  # type: ignore[return-value]
    if isinstance(latent, (list, tuple)):
        if not latent:
            raise ValueError("Received empty latent container")
        return _ensure_latent_dict(latent[0])
    raise TypeError(f"Unsupported latent payload type: {type(latent)!r}")


def clone_latent(latent: Any) -> Dict[str, torch.Tensor]:
    source = _ensure_latent_dict(latent)
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in source.items()}


def clone_conditioning_payload(conditioning: Any) -> Any:
    if conditioning is None:
        return None
    if isinstance(conditioning, torch.Tensor):
        return conditioning.clone()
    if isinstance(conditioning, list):
        return [clone_conditioning_payload(item) for item in conditioning]
    if isinstance(conditioning, tuple):
        return tuple(clone_conditioning_payload(item) for item in conditioning)
    if isinstance(conditioning, dict):
        return {key: clone_conditioning_payload(value) for key, value in conditioning.items()}
    try:
        return copy.deepcopy(conditioning)
    except Exception:
        return conditioning


def _clone_generic_payload(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_generic_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_generic_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_generic_payload(item) for item in value)
    if isinstance(value, set):
        return {_clone_generic_payload(item) for item in value}
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


class h4_PlotXY:
    """All-in-one checkpoint loader, sampler, scheduler, and grid plotter powering h4 : The Engine."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802 - ComfyUI naming convention
        def list_safe(category: str) -> List[str]:
            try:
                return folder_paths.get_filename_list(category)
            except Exception:  # pragma: no cover - external environment
                GLOBAL_LOGGER.warn(f"Unable to enumerate assets for category '{category}'")
                return []

        checkpoint_list = list_safe("checkpoints")
        vae_list = list_safe("vae")
        clip_list = list_safe("clip")
        clip_vision_list = list_safe("clip_vision")

        default_checkpoint = checkpoint_list[0] if checkpoint_list else ""
        vae_choices = ("<checkpoint>",) + tuple(vae_list)
        clip_choices = ("<checkpoint>",) + tuple(clip_list)
        clip_vision_choices = ("<checkpoint>",) + tuple(clip_vision_list)

        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 30.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "clip_skip": ("INT", {"default": 2, "min": 1, "max": 12}),
                "sampler_name": (SAMPLER_CHOICES, {"default": SAMPLER_CHOICES[0]}),
                "scheduler_name": (SCHEDULER_CHOICES, {"default": SCHEDULER_CHOICES[0]}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "checkpoint": (checkpoint_list, {"default": default_checkpoint} if checkpoint_list else {"default": ""}),
                "vae_name": (vae_choices, {"default": vae_choices[0]}),
                "clip_text_name": (clip_choices, {"default": clip_choices[0]}),
                "clip_vision_name": (clip_vision_choices, {"default": clip_vision_choices[0]}),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "x_axis_mode": (AXIS_PRESETS, {"default": "prompt"}),
                "x_axis_values": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "y_axis_mode": (AXIS_PRESETS, {"default": "none"}),
                "y_axis_values": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "bypass_engine": ("BOOLEAN", {"default": False, "label": "Bypass (passthrough mode)"}),
            },
            "optional": {
                "model_in": ("MODEL",),
                "clip_in": ("CLIP",),
                "vae_in": ("VAE",),
                "conditioning_in": ("CONDITIONING",),
                "conditioning_positive_in": ("CONDITIONING",),
                "conditioning_negative_in": ("CONDITIONING",),
                "latent_in": ("LATENT",),
                "image_in": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = (
        "Images",
        "Grid_Image",
        "Model",
        "CLIP",
        "VAE",
        "Latent",
        "Positive",
        "Negative",
    )
    FUNCTION = "run_pipeline"
    CATEGORY = "h4 Toolkit/Generation"

    def __init__(self) -> None:
        self.logger = ToolkitLogger("h4_Engine")
        self.clip_encoder = comfy_nodes.CLIPTextEncode()
        self.vae_decode = comfy_nodes.VAEDecode()
        vae_encoder_cls = getattr(comfy_nodes, "VAELatentEncode", None) or getattr(
            comfy_nodes, "VAEEncode", None
        )
        self.vae_encode = vae_encoder_cls() if vae_encoder_cls else None
        if self.vae_encode is None:
            self.logger.warn(
                "Falling back to direct VAE.encode; plug a latent input for best stability."
            )
        self.empty_latent = comfy_nodes.EmptyLatentImage()
        self.lora_loader = comfy_nodes.LoraLoader()
        vae_loader_cls = getattr(comfy_nodes, "VAELoader", None)
        self.vae_loader = vae_loader_cls() if callable(vae_loader_cls) else None
        self.clip_loader = getattr(comfy_nodes, "CLIPLoader", None)
        self.dual_clip_loader = getattr(comfy_nodes, "DualCLIPLoader", None)
        self._clip_skip_warned = False
        self._model_sampling_warned = False
        self._sampler_object_warned = False
        self._preview_handlers: Optional[Tuple[Optional[Callable[..., Any]], Optional[Callable[..., Any]]]] = None
        self._preview_disabled_logged = False
        self._preview_error_logged = False
        self._preview_pil_missing_logged = False
        preview_max = None
        if latent_preview is not None:
            preview_max = getattr(latent_preview, "MAX_PREVIEW_RESOLUTION", None)
        if preview_max is None and comfy_args is not None:
            preview_max = getattr(comfy_args, "preview_size", None)
        if isinstance(preview_max, (int, float)):
            preview_max_int = int(preview_max)
            self._preview_max_resolution: Optional[int] = preview_max_int if preview_max_int > 0 else None
        else:
            self._preview_max_resolution = None

    def _load_checkpoint(self, checkpoint_name: str) -> Tuple[Any, Any, Any]:
        resolved_path = _resolve_asset_path("checkpoints", checkpoint_name)
        if resolved_path is None:
            message = f"Checkpoint '{checkpoint_name}' could not be resolved"
            self.logger.error(message)
            raise FileNotFoundError(message)
        self.logger.trace(f"Loading checkpoint: {resolved_path}")
        load_fn = getattr(comfy_sd, "load_checkpoint_guess_config", None)
        if load_fn is None:
            load_fn = getattr(comfy_sd, "load_checkpoint", None)
        if load_fn is None:
            raise RuntimeError("ComfyUI checkpoint loader is unavailable")
        try:
            bundle = load_fn(resolved_path, output_vae=True, output_clip=True)
        except TypeError:
            bundle = load_fn(resolved_path)
        model = None
        clip = None
        vae = None
        if isinstance(bundle, (list, tuple)):
            if len(bundle) >= 3:
                model, clip, vae = bundle[:3]
            elif len(bundle) == 2:
                model, clip = bundle
            elif len(bundle) == 1:
                model = bundle[0]
        else:
            model = bundle
        if model is None:
            raise RuntimeError(f"Checkpoint loader did not return a model for '{checkpoint_name}'")
        if clip is None:
            self.logger.warn("Checkpoint did not supply a CLIP; downstream overrides may fill this gap")
        if vae is None:
            self.logger.warn("Checkpoint did not supply a VAE; downstream overrides may fill this gap")
        self.logger.info(f"Checkpoint ready: {checkpoint_name}")
        return model, clip, vae

    def _load_vae_override(self, current_vae: Any, vae_name: str) -> Any:
        if not vae_name or vae_name == "<checkpoint>":
            return current_vae
        resolved_path = _resolve_asset_path("vae", vae_name)
        if resolved_path is None:
            self.logger.warn(f"VAE override '{vae_name}' could not be resolved; using checkpoint VAE")
            return current_vae
        candidate_loaders: List[Tuple[Any, Tuple[Any, ...], Dict[str, Any], bool]] = []
        direct_loader = getattr(comfy_sd, "load_vae", None)
        if direct_loader is not None:
            candidate_loaders.append((direct_loader, (resolved_path,), {}, True))
        if self.vae_loader is not None:
            candidate_loaders.append((self.vae_loader.load_vae, (vae_name,), {}, False))  # type: ignore[attr-defined]
        for loader, args, kwargs, allow_path_kw in candidate_loaders:
            try:
                loaded = loader(*args, **kwargs)  # type: ignore[misc]
            except TypeError as exc:
                if allow_path_kw:
                    try:
                        loaded = loader(path=args[0])  # type: ignore[misc]
                    except Exception as inner_exc:  # pragma: no cover - loader specific
                        self.logger.warn(f"VAE loader signature mismatch ({loader}): {inner_exc}")
                        continue
                else:
                    self.logger.warn(f"VAE loader signature mismatch ({loader}): {exc}")
                    continue
            except Exception as exc:  # pragma: no cover - loader specific
                self.logger.warn(f"VAE loader failed ({loader}): {exc}")
                continue
            vae_obj = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
            if vae_obj is None:
                continue
            self.logger.info(f"Using VAE override: {vae_name}")
            return vae_obj
        self.logger.warn("All VAE loaders failed; falling back to checkpoint VAE")
        return current_vae

    def _load_clip_override(
        self,
        current_clip: Any,
        clip_text_name: str,
        clip_vision_name: str,
    ) -> Any:
        text_override = clip_text_name not in (None, "", "<checkpoint>")
        vision_override = clip_vision_name not in (None, "", "<checkpoint>")
        if not text_override and not vision_override:
            return current_clip

        text_path = _resolve_asset_path("clip", clip_text_name) if text_override else None
        vision_path = _resolve_asset_path("clip_vision", clip_vision_name) if vision_override else None

        loader = getattr(comfy_sd, "load_clip", None)
        last_error: Optional[str] = None
        if loader is not None:
            try:
                loaded = loader(text_path, vision_path)  # type: ignore[misc]
            except TypeError:
                try:
                    kwargs: Dict[str, Any] = {}
                    if text_path is not None:
                        kwargs["clip_path"] = text_path
                    if vision_path is not None:
                        kwargs["clip_vision_path"] = vision_path
                    loaded = loader(**kwargs)  # type: ignore[misc]
                except Exception as exc:  # pragma: no cover - loader specific
                    last_error = str(exc)
                    loaded = None
            except Exception as exc:  # pragma: no cover - loader specific
                last_error = str(exc)
                loaded = None
            if loaded is not None:
                clip_obj = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
                if clip_obj is not None:
                    override_label = clip_text_name if text_override else "<checkpoint>"
                    if vision_override:
                        override_label = f"{override_label} / {clip_vision_name}"
                    self.logger.info(f"Using CLIP override: {override_label}")
                    return clip_obj

        node_loader = None
        if text_override:
            if self.dual_clip_loader is not None:
                node_loader = self.dual_clip_loader.load_clip  # type: ignore[assignment]
            elif self.clip_loader is not None and not vision_override:
                node_loader = self.clip_loader.load_clip  # type: ignore[assignment]
        if node_loader is not None:
            try:
                loaded = node_loader(clip_text_name, clip_vision_name if vision_override else None)  # type: ignore[misc]
                clip_obj = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
                if clip_obj is not None:
                    override_label = clip_text_name if text_override else "<checkpoint>"
                    if vision_override:
                        override_label = f"{override_label} / {clip_vision_name}"
                    self.logger.info(f"Using CLIP override: {override_label}")
                    return clip_obj
            except Exception as exc:  # pragma: no cover - loader specific
                last_error = str(exc)

        if last_error:
            self.logger.warn(f"Unable to load CLIP override: {last_error}")
        else:
            self.logger.warn("Clip override requested but no loader succeeded; using checkpoint clip")
        return current_clip

    def _apply_clip_skip(self, clip: Any, clip_skip: int) -> Any:
        if clip_skip <= 1:
            return clip
        setter = getattr(comfy_utils, "set_clip_skip", None)
        if setter is None and hasattr(clip, "set_clip_skip"):
            setter = clip.set_clip_skip  # type: ignore[assignment]
        if setter is None and hasattr(clip, "clip") and hasattr(clip.clip, "set_clip_skip"):
            setter = clip.clip.set_clip_skip  # type: ignore[assignment]
        if setter is None and hasattr(clip, "cond_stage_model") and hasattr(
            clip.cond_stage_model, "set_clip_skip"
        ):
            setter = clip.cond_stage_model.set_clip_skip  # type: ignore[assignment]
        if setter is None:
            if not self._clip_skip_warned:
                self._clip_skip_warned = True
                self.logger.info(
                    "Clip skip helper not available in this environment; using checkpoint configuration"
                )
            return clip
        try:
            try:
                result = setter(clip, clip_skip)  # type: ignore[misc]
            except TypeError:
                result = setter(clip_skip)  # type: ignore[misc]
            if result is not None:
                clip = result
            self.logger.info(f"Clip skip set to {clip_skip}")
        except Exception as exc:  # pragma: no cover - setter specific
            self.logger.warn(f"Clip skip update failed ({clip_skip}): {exc}")
        return clip

    def _encode_prompt(self, clip: Any, prompt: str) -> Any:
        self.logger.trace(f"Encoding prompt: {prompt[:60]}...")
        result = self.clip_encoder.encode(clip, prompt)
        return result[0] if isinstance(result, tuple) else result

    def _clone_conditioning(self, conditioning: Any) -> Any:
        return clone_conditioning_payload(conditioning)

    def _normalise_image_tensor(self, image: Any) -> torch.Tensor:
        if isinstance(image, dict):
            for key in ("image", "images", "pixels"):
                if key in image:
                    return self._normalise_image_tensor(image[key])
            raise TypeError("Image dictionary is missing 'image'/'images'/'pixels' keys")

        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
        elif torch.is_tensor(image):
            tensor = image
        else:
            raise TypeError(f"Unsupported image payload type: {type(image)!r}")

        tensor = tensor.to(dtype=torch.float32)

        # Ensure a 4D tensor [batch, *, *, *]
        while tensor.ndim < 4:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Image tensor must be rank 4 after normalisation, got shape {tuple(tensor.shape)}")

        dims = list(tensor.shape)
        spatial_axes = [1, 2, 3]
        channel_axis_candidates: List[int] = []
        for axis in spatial_axes:
            dim = dims[axis]
            if 0 < dim <= 4:
                channel_axis_candidates.append(axis)

        if not channel_axis_candidates:
            # fall back to the smallest non-zero spatial dimension
            positive_dims = [(axis, dims[axis]) for axis in spatial_axes if dims[axis] > 0]
            if not positive_dims:
                raise ValueError(f"Image tensor has zero-sized spatial dimensions: {tuple(dims)}")
            channel_axis_candidates = [min(positive_dims, key=lambda item: item[1])[0]]

        channel_axis = channel_axis_candidates[-1]
        if channel_axis == 0:
            channel_axis = 3

        permute_order = [0] + [axis for axis in spatial_axes if axis != channel_axis] + [channel_axis]
        bhwc = tensor.permute(*permute_order)

        spatial_shape = bhwc.shape[1:3]
        if 0 in spatial_shape:
            raise ValueError(f"Normalised image tensor has zero-sized spatial dimension: {tuple(bhwc.shape)}")

        bhwc = bhwc.contiguous()
        if bhwc.dtype != torch.float32:
            bhwc = bhwc.to(torch.float32)

        max_val = float(bhwc.max().item())
        min_val = float(bhwc.min().item())
        if max_val > 1.01 or min_val < -0.01:
            bhwc = bhwc / 255.0
            max_val = float(bhwc.max().item())
            min_val = float(bhwc.min().item())
        bhwc = bhwc.clamp(0.0, 1.0)
        if max_val > 1.0 or min_val < 0.0:
            self.logger.warn(
                f"Image payload outside expected range [{min_val:.3f}, {max_val:.3f}]; values clamped to [0, 1]"
            )
        return bhwc

    def _prepare_image_payload(self, image: Any) -> Dict[str, torch.Tensor]:
        if isinstance(image, (list, tuple)) and image:
            image = image[0]

        mask = None
        if isinstance(image, dict):
            if "samples" in image:
                raise TypeError("Latent-like dictionary received where IMAGE payload was expected")
            candidate = image.get("image") or image.get("images") or image.get("pixels")
            if candidate is None:
                raise KeyError("Image payload dictionary missing 'image'/'images'/'pixels' keys")
            mask = image.get("mask")
            normalised = self._normalise_image_tensor(candidate)
        else:
            normalised = self._normalise_image_tensor(image)

        payload: Dict[str, torch.Tensor] = {"image": normalised, "images": normalised}
        if mask is not None:
            payload["mask"] = mask
        return payload

    def _prepare_latent(
        self,
        vae: Any,
        width: int,
        height: int,
        latent_in: Optional[Dict[str, torch.Tensor]],
        image_in: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        width_aligned = max(8, ((int(width) + 7) // 8) * 8)
        height_aligned = max(8, ((int(height) + 7) // 8) * 8)
        if width_aligned != width or height_aligned != height:
            self.logger.warn(
                f"Latent dimensions realigned to VAE requirements: {width}x{height} -> {width_aligned}x{height_aligned}"
            )
        width = width_aligned
        height = height_aligned
        if latent_in is not None:
            self.logger.info("Using supplied latent as base")
            return clone_latent(latent_in)
        if image_in is not None:
            if isinstance(image_in, dict) and "samples" in image_in:
                self.logger.info("Image input already in latent format; using directly")
                return clone_latent(image_in)
            self.logger.info("Encoding supplied image into latent")
            try:
                image_payload = self._prepare_image_payload(image_in)
            except Exception as exc:
                raise RuntimeError(f"Unable to normalise image input for latent encoding: {exc}") from exc
            self.logger.trace(
                f"Image payload normalised shape: {tuple(image_payload['image'].shape)}"
            )
            image_tensor = image_payload["image"].to(dtype=torch.float32)
            if self.vae_encode is not None:
                try:
                    latent = self.vae_encode.encode(vae, image_tensor)
                    return clone_latent(latent)
                except Exception as exc:
                    self.logger.warn(f"VAELatentEncode failed after normalisation; attempting direct encode path ({exc})")
            if hasattr(vae, "encode"):
                self.logger.trace(f"Direct encode input BHWC shape: {tuple(image_tensor.shape)}")
                pixels = image_tensor.permute(0, 3, 1, 2).contiguous()
                self.logger.trace(f"Direct encode input BCHW shape: {tuple(pixels.shape)}")
                try:
                    encoded = vae.encode(pixels)
                except TypeError:
                    encoded = vae.encode(pixels)
                if isinstance(encoded, (dict, list, tuple)):
                    return clone_latent(encoded)
                return clone_latent({"samples": encoded})
            raise RuntimeError("Active VAE cannot encode images into latents; please supply a latent input.")
        self.logger.info("Creating empty latent with requested dimensions")
        latent = self.empty_latent.generate(width, height, 1)
        return clone_latent(latent)

    def _apply_loras(
        self,
        model: Any,
        clip: Any,
        loras: Iterable[Tuple[str, float]],
    ) -> Tuple[Any, Any]:
        patched_model, patched_clip = model, clip
        for lora_name, strength in loras:
            self.logger.trace(f"Applying LoRA {lora_name} @ {strength}")
            patched_model, patched_clip = self.lora_loader.load_lora(
                patched_model, patched_clip, lora_name, strength, strength
            )
        return patched_model, patched_clip

    def _get_preview_handlers(self) -> Tuple[Optional[Callable[..., Any]], Optional[Callable[..., Any]]]:
        if self._preview_handlers is not None:
            return self._preview_handlers

        decoder_candidates: List[Optional[Callable[..., Any]]] = []
        publisher_candidates: List[Optional[Callable[..., Any]]] = []

        if latent_preview is not None:
            for name in ("decode_latent_preview", "decode_latent", "decode"):
                decoder_candidates.append(getattr(latent_preview, name, None))
            for name in ("publish_preview", "publish"):
                publisher_candidates.append(getattr(latent_preview, name, None))

        decoder_candidates.append(getattr(comfy_utils, "decode_latent_preview", None))
        publisher_candidates.extend(
            [
                getattr(comfy_utils, "publish_preview", None),
                getattr(comfy_utils, "publish_preview_image", None),
            ]
        )

        def pick_callable(candidates: List[Optional[Callable[..., Any]]]) -> Optional[Callable[..., Any]]:
            for candidate in candidates:
                if callable(candidate):
                    return candidate
            return None

        decoder_callable = pick_callable(decoder_candidates)
        publisher_callable = pick_callable(publisher_candidates)

        if decoder_callable is None:
            def decode_with_vae(vae_obj: Any, latent_tensor: torch.Tensor) -> torch.Tensor:
                latent_dict = {"samples": latent_tensor}
                return self.vae_decode.decode(vae_obj, latent_dict)[0]

            decoder_callable = decode_with_vae

        self._preview_handlers = (decoder_callable, publisher_callable)
        return self._preview_handlers

    def _render_preview_tuple(
        self,
        image_like: Any,
    ) -> Optional[Tuple[str, Any, Optional[int]]]:
        if Image is None:
            if not self._preview_pil_missing_logged:
                self.logger.info("Pillow is unavailable; previews will be limited to progress updates only")
                self._preview_pil_missing_logged = True
            return None
        try:
            normalised = self._normalise_image_tensor(image_like)
        except Exception as exc:
            if TRACE_ENABLED:
                self.logger.trace(f"Preview normalisation failed: {exc}")
            return None
        try:
            sample = normalised[0].detach().to(device="cpu", dtype=torch.float32)
        except Exception:
            try:
                sample = normalised.detach().to(device="cpu", dtype=torch.float32)
                if sample.ndim >= 4:
                    sample = sample[0]
            except Exception as exc:
                if TRACE_ENABLED:
                    self.logger.trace(f"Preview tensor conversion failed: {exc}")
                return None
        sample = sample.clamp(0.0, 1.0)
        data = sample.cpu().numpy()
        if data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        if data.ndim != 3:
            if TRACE_ENABLED:
                self.logger.trace(f"Unexpected preview tensor rank: {data.ndim}")
            return None
        channels = data.shape[-1]
        if channels == 1:
            data = np.repeat(data, 3, axis=-1)
        elif channels == 2:
            padding = np.zeros((*data.shape[:-1], 1), dtype=data.dtype)
            data = np.concatenate([data, padding], axis=-1)
        data_uint8 = np.clip(data * 255.0, 0, 255).astype(np.uint8)
        try:
            image = Image.fromarray(data_uint8)
        except Exception as exc:
            if TRACE_ENABLED:
                self.logger.trace(f"PIL conversion failed: {exc}")
            return None
        return ("JPEG", image, self._preview_max_resolution)

    def _run_sampler(
        self,
        model: Any,
        vae: Any,
        positive: Any,
        negative: Any,
        latent: Dict[str, torch.Tensor],
        seed: int,
        steps: int,
        cfg: float,
        denoise: float,
        sampler_name: str,
        scheduler_name: str,
    ) -> Dict[str, torch.Tensor]:
        samples = latent["samples"]
        generator = torch.Generator(device=samples.device).manual_seed(int(seed))
        try:
            noise = torch.randn_like(samples, generator=generator)
        except TypeError:
            noise = torch.randn(
                samples.shape,
                dtype=samples.dtype,
                device=samples.device,
                generator=generator,
            )
        except RuntimeError as exc:
            self.logger.warn(f"Noise sampling fallback engaged: {exc}")
            devices = [samples.device] if samples.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                noise = torch.randn_like(samples)
        preview_token = f"engine-preview-{time.time():.0f}-{seed}"
        node_id_for_pbar: Optional[str] = None
        if callable(get_executing_context):  # pragma: no branch - optional context
            try:
                context = get_executing_context()
            except Exception:  # pragma: no cover - defensive
                context = None
            if context is not None:
                node_id_for_pbar = getattr(context, "node_id", None)
        pbar = comfy_utils.ProgressBar(steps, node_id=node_id_for_pbar)

        def callback(step: int, x0: torch.Tensor, *_args: Any) -> None:
            self.logger.trace(f"Preview step {step}/{steps}")
            decoder, publisher = self._get_preview_handlers()
            decoded = None
            if decoder is not None:
                try:
                    try:
                        decoded = decoder(vae, x0)
                    except TypeError:
                        decoded = decoder(x0)  # type: ignore[misc]
                except Exception as exc:  # pragma: no cover - best effort
                    if not self._preview_error_logged:
                        self.logger.warn(f"Preview decode failed; disabling thumbnails ({exc})")
                        self._preview_error_logged = True
                    decoded = None
                    self._preview_handlers = None
            elif not self._preview_disabled_logged:
                self.logger.info("Preview decoder unavailable; live previews disabled")
                self._preview_disabled_logged = True

            preview_payload: Optional[Tuple[str, Any, Optional[int]]] = None
            if decoded is not None:
                preview_payload = self._render_preview_tuple(decoded)
                if publisher is not None:
                    try:
                        publisher(decoded, preview_token)
                    except Exception as exc:  # pragma: no cover - best effort only
                        if not self._preview_error_logged:
                            self.logger.warn(f"Preview publishing disabled after error: {exc}")
                            self._preview_error_logged = True
                        self._preview_handlers = (decoder, None)
                elif preview_payload is None and not self._preview_disabled_logged:
                    self.logger.info("Preview renderer unavailable; showing progress without thumbnails")
                    self._preview_disabled_logged = True
            pbar.update_absolute(step + 1, steps, preview_payload)

        self.logger.info(
            f"Sampling with {sampler_name} + {scheduler_name}, steps={steps}, cfg={cfg:.2f}, denoise={denoise:.2f}"
        )
        model_sampling = _extract_model_sampling(model)
        model_sampling_meta: Dict[str, Any] = {}
        if model_sampling is not None:
            for attr in (
                "sigma_min",
                "sigma_max",
                "sigma_data",
                "sigma_schedule",
                "schedule",
                "model_type",
            ):
                value = _callable_or_value(getattr(model_sampling, attr, None))
                if value is not None:
                    model_sampling_meta[attr] = value
        model_type_value = _callable_or_value(getattr(model, "model_type", None))
        if model_type_value is not None:
            model_sampling_meta.setdefault("model_type", model_type_value)
        if model_sampling is None and not self._model_sampling_warned:
            self._model_sampling_warned = True
            self.logger.warn(
                "Unable to determine model sampling metadata; attempting sampler call without explicit type"
            )
        latent_tensor = latent.get("samples", latent)
        sampler_kwargs = {
            "model": model,
            "noise": noise,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler_name": scheduler_name,
            "scheduler": scheduler_name,
            "positive": positive,
            "negative": negative,
            "latent": latent_tensor,
            "seed": seed,
            "denoise": denoise,
            "disable_noise": False,
            "start_step": 0,
            "last_step": -1,
            "force_full_denoise": False,
            "callback": callback,
            "device": samples.device,
        }
        if model_sampling is not None:
            sampler_kwargs["model_sampling"] = model_sampling
        if model_sampling_meta:
            sampler_kwargs.update({key: value for key, value in model_sampling_meta.items() if key not in sampler_kwargs})

        sampler_object = _resolve_sampler_object(
            self.logger,
            sampler_name,
            scheduler_name,
            model,
            device=samples.device,
            model_sampling=model_sampling,
            model_sampling_meta=model_sampling_meta,
        )
        if sampler_object is not None:
            sampler_kwargs["sampler"] = sampler_object
            self.logger.trace(
                f"Resolved sampler object for '{sampler_name}'/{scheduler_name}: {sampler_object}"
            )
        elif not self._sampler_object_warned:
            self._sampler_object_warned = True
            self.logger.warn(
                "Sampler factory unavailable; relying on compatibility shim with sampler name only"
            )

        sigma_generator = getattr(comfy_samplers, "calculate_sigmas", None)
        if sigma_generator is not None:
            try:
                try:
                    sigma_signature = inspect.signature(sigma_generator)
                except (TypeError, ValueError):
                    sigma_signature = None

                available_values: Dict[str, Any] = {
                    "sampler": sampler_name,
                    "sampler_name": sampler_name,
                    "sampler_type": sampler_name,
                    "scheduler": scheduler_name,
                    "scheduler_name": scheduler_name,
                    "steps": steps,
                    "num_steps": steps,
                    "step_count": steps,
                    "start_step": 0,
                    "first_step": 0,
                    "last_step": -1,
                    "end_step": -1,
                    "force_full_denoise": False,
                    "full_denoise": False,
                    "denoise": denoise,
                    "denoise_strength": denoise,
                    "device": samples.device,
                    "model": model,
                    "noise": noise,
                    "latent": latent,
                }
                if model_sampling is not None:
                    available_values["model_sampling"] = model_sampling
                for key, value in model_sampling_meta.items():
                    available_values.setdefault(key, value)
                if model_type_value is not None:
                    available_values.setdefault("model_type", model_type_value)

                sigma_args: List[Any] = []
                sigma_kwargs: Dict[str, Any] = {}
                missing_required: List[str] = []

                if sigma_signature is not None:
                    for name, param in sigma_signature.parameters.items():
                        if param.kind == param.VAR_POSITIONAL:
                            continue
                        if param.kind == param.VAR_KEYWORD:
                            continue
                        value = available_values.get(name)
                        if value is None:
                            if param.default is inspect._empty and param.kind in (
                                param.POSITIONAL_ONLY,
                                param.POSITIONAL_OR_KEYWORD,
                                param.KEYWORD_ONLY,
                            ):
                                missing_required.append(name)
                            continue
                        if param.kind == param.POSITIONAL_ONLY:
                            sigma_args.append(value)
                        elif param.kind == param.POSITIONAL_OR_KEYWORD:
                            sigma_kwargs[name] = value
                        elif param.kind == param.KEYWORD_ONLY:
                            sigma_kwargs[name] = value
                else:
                    sigma_kwargs = {
                        "sampler_name": sampler_name,
                        "scheduler_name": scheduler_name,
                        "steps": steps,
                        "start_step": 0,
                        "last_step": -1,
                        "force_full_denoise": False,
                        "denoise": denoise,
                    }
                    if model_sampling is not None:
                        sigma_kwargs["model_sampling"] = model_sampling
                    for key, value in model_sampling_meta.items():
                        sigma_kwargs.setdefault(key, value)
                    if model_type_value is not None:
                        sigma_kwargs.setdefault("model_type", model_type_value)

                if missing_required:
                    raise RuntimeError(
                        "calculate_sigmas signature requires unsupported parameters: "
                        + ", ".join(missing_required)
                    )

                sigmas = sigma_generator(*sigma_args, **sigma_kwargs)
                if isinstance(sigmas, (list, tuple)):
                    sigmas = torch.tensor(sigmas, dtype=samples.dtype, device=samples.device)
                elif hasattr(sigmas, "to"):
                    sigmas = sigmas.to(device=samples.device, dtype=samples.dtype)
                sampler_kwargs["sigmas"] = sigmas
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.warn(f"Unable to precompute sigma schedule: {exc}")
        result = _invoke_sampler_with_compatibility(self.logger, sampler_kwargs)
        if isinstance(result, torch.Tensor):
            result = {"samples": result}
        return result

    def _decode(self, vae: Any, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = self.vae_decode.decode(vae, latent)[0]
        self.logger.trace("Decoded latent into image tensor")
        return images

    def _build_report(
        self,
        plans: List[GenerationPlan],
        results: List[Dict[str, Any]],
        grid_shape: Tuple[int, int],
    ) -> str:
        payload = {
            "node": "h4_PlotXY",
            "version": PLOT_NODE_VERSION,
            "grid": {"rows": grid_shape[0], "columns": grid_shape[1]},
            "runs": [
                {
                    "plan": plan.as_dict(),
                    "result": summary,
                }
                for plan, summary in zip(plans, results)
            ],
        }
        report = json.dumps(payload, indent=2)
        self.logger.info("Execution report generated")
        return report

    def run_pipeline(
        self,
        seed: int,
        steps: int,
        cfg: float,
        denoise: float,
        clip_skip: int,
        sampler_name: str,
        scheduler_name: str,
        width: int,
        height: int,
        checkpoint: str,
        vae_name: str,
        clip_text_name: str,
        clip_vision_name: str,
        positive_prompt: str,
        negative_prompt: str,
        x_axis_mode: str,
        x_axis_values: str,
        y_axis_mode: str,
        y_axis_values: str,
        bypass_engine: bool,
        model_in: Optional[Any] = None,
        clip_in: Optional[Any] = None,
        vae_in: Optional[Any] = None,
        conditioning_in: Optional[Any] = None,
        conditioning_positive_in: Optional[Any] = None,
        conditioning_negative_in: Optional[Any] = None,
        latent_in: Optional[Dict[str, torch.Tensor]] = None,
        image_in: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Any,
        Any,
        Any,
        Dict[str, torch.Tensor],
        Any,
        Any,
    ]:
        self.logger.info("Engine pipeline starting")
        bundle_cache: Dict[str, Tuple[Any, Any, Any]] = {}
        last_loaded_checkpoint: Optional[str] = None

        def cache_key_for(target_checkpoint: Optional[str]) -> str:
            target = target_checkpoint or checkpoint
            inputs_available = any(
                item is not None
                for item in (
                    model_in,
                    clip_in,
                    vae_in,
                    conditioning_in,
                    conditioning_positive_in,
                    conditioning_negative_in,
                )
            )
            if target == checkpoint and inputs_available:
                return "__inputs__"
            return f"ckpt::{target}"

        def get_model_bundle(name: Optional[str]) -> Tuple[Any, Any, Any]:
            nonlocal last_loaded_checkpoint
            target = name or checkpoint
            key = cache_key_for(name)
            if key in bundle_cache:
                return bundle_cache[key]

            base_model: Optional[Any] = None
            base_clip: Optional[Any] = None
            base_vae: Optional[Any] = None

            if key == "__inputs__":
                base_model = model_in
                base_clip = clip_in
                base_vae = vae_in
                missing_components = [component is None for component in (base_model, base_clip, base_vae)]
                if any(missing_components):
                    loaded_model, loaded_clip, loaded_vae = self._load_checkpoint(target)
                    base_model = base_model or loaded_model
                    base_clip = base_clip or loaded_clip
                    base_vae = base_vae or loaded_vae
            else:
                if last_loaded_checkpoint and last_loaded_checkpoint != target:
                    self.logger.info("Clearing previous checkpoint from memory")
                    model_management.soft_empty_cache()
                loaded_model, loaded_clip, loaded_vae = self._load_checkpoint(target)
                base_model, base_clip, base_vae = loaded_model, loaded_clip, loaded_vae
                if target == checkpoint:
                    if model_in is not None:
                        base_model = model_in
                    if vae_in is not None:
                        base_vae = vae_in
                    if clip_in is not None:
                        base_clip = clip_in
                last_loaded_checkpoint = target

            if base_model is None:
                raise RuntimeError("Model bundle missing model instance")

            if vae_in is None:
                base_vae = self._load_vae_override(base_vae, vae_name)
            if base_vae is None:
                raise RuntimeError("Model bundle missing VAE instance")

            clip_candidate = base_clip
            if clip_in is None:
                clip_candidate = self._load_clip_override(base_clip, clip_text_name, clip_vision_name)
            if clip_candidate is None:
                self.logger.warn("No CLIP available after overrides; using checkpoint CLIP as fallback")
                clip_candidate = base_clip
            if clip_candidate is None:
                raise RuntimeError("Model bundle missing CLIP instance")
            if clip_skip > 1:
                clip_candidate = self._apply_clip_skip(clip_candidate, clip_skip)

            bundle = (base_model, clip_candidate, base_vae)
            bundle_cache[key] = bundle
            return bundle

        if bypass_engine:
            self.logger.info("Bypass mode enabled; relaying inputs without sampling")
            primary_model, primary_clip, primary_vae = get_model_bundle(None)

            active_latent: Optional[Dict[str, torch.Tensor]] = None
            images: Optional[torch.Tensor] = None

            if latent_in is not None:
                active_latent = clone_latent(latent_in)
                images = self._decode(primary_vae, active_latent)
            elif image_in is not None:
                try:
                    image_payload = self._prepare_image_payload(image_in)
                except Exception as exc:
                    raise RuntimeError(f"Unable to normalise image input during bypass: {exc}") from exc
                image_tensor = image_payload["image"].to(dtype=torch.float32)
                images = image_tensor
                if self.vae_encode is not None:
                    try:
                        encoded = self.vae_encode.encode(primary_vae, image_tensor)
                        active_latent = clone_latent(encoded)
                    except Exception as exc:
                        self.logger.warn(
                            f"VAELatentEncode failed while preparing bypass latent; attempting direct encode ({exc})"
                        )
                if active_latent is None:
                    if hasattr(primary_vae, "encode"):
                        pixels = image_tensor.permute(0, 3, 1, 2).contiguous()
                        try:
                            direct_encoded = primary_vae.encode(pixels)
                        except TypeError:
                            direct_encoded = primary_vae.encode(pixels)
                        if isinstance(direct_encoded, (dict, list, tuple)):
                            active_latent = clone_latent(direct_encoded)
                        else:
                            active_latent = clone_latent({"samples": direct_encoded})
                    else:
                        raise RuntimeError(
                            "Bypass mode requires a latent input or a VAE capable of encoding images"
                        )
            else:
                self.logger.info("No latent or image input supplied; creating empty latent for bypass output")
                active_latent = self._prepare_latent(primary_vae, width, height, None, None)
                images = self._decode(primary_vae, active_latent)

            if active_latent is None or images is None:
                raise RuntimeError("Bypass mode could not prepare outputs; ensure a latent or image input is connected")

            images_out = images.detach().clone()
            stacked = images_out
            grid = images_out.clone()

            legacy_conditioning = self._clone_conditioning(conditioning_in) if conditioning_in is not None else None
            positive_source = (
                conditioning_positive_in if conditioning_positive_in is not None else legacy_conditioning
            )
            negative_source = (
                conditioning_negative_in if conditioning_negative_in is not None else legacy_conditioning
            )

            if positive_source is not None:
                positive_out = self._clone_conditioning(positive_source)
            else:
                fallback_positive = positive_prompt or ""
                positive_out = (
                    self._encode_prompt(primary_clip, fallback_positive) if primary_clip is not None else []
                )

            if negative_source is not None:
                negative_out = self._clone_conditioning(negative_source)
            else:
                fallback_negative = negative_prompt if negative_prompt.strip() else ""
                negative_out = (
                    self._encode_prompt(primary_clip, fallback_negative) if primary_clip is not None else []
                )

            latent_export = clone_latent(active_latent)
            self.logger.info("Bypass mode complete")
            return (
                stacked,
                grid,
                primary_model,
                primary_clip,
                primary_vae,
                latent_export,
                positive_out,
                negative_out,
            )

        x_descriptors = parse_axis_entries(x_axis_mode, x_axis_values)
        y_descriptors = parse_axis_entries(y_axis_mode, y_axis_values)
        plans = build_generation_matrix(
            base_checkpoint=checkpoint,
            x_descriptors=x_descriptors,
            y_descriptors=y_descriptors,
        )
        if not plans:
            raise RuntimeError("No execution plans could be constructed from the provided axis values")
        self.logger.info(f"Generated {len(plans)} execution plan(s)")
        grid_rows = max(1, len(y_descriptors))
        grid_cols = max(1, len(x_descriptors))

        primary_model, primary_clip, primary_vae = get_model_bundle(None)

        base_latent = self._prepare_latent(
            primary_vae,
            width,
            height,
            latent_in,
            image_in,
        )

        grid_images: List[torch.Tensor] = []
        image_stack: List[torch.Tensor] = []
        plan_summaries: List[Dict[str, Any]] = []

        final_model = primary_model
        final_clip = primary_clip
        final_vae = primary_vae
        final_latent: Optional[Dict[str, torch.Tensor]] = None
        final_positive: Optional[Any] = None
        final_negative: Optional[Any] = None

        legacy_conditioning = self._clone_conditioning(conditioning_in) if conditioning_in is not None else None
        positive_override_source = (
            conditioning_positive_in if conditioning_positive_in is not None else legacy_conditioning
        )
        negative_override_source = (
            conditioning_negative_in if conditioning_negative_in is not None else legacy_conditioning
        )

        for index, plan in enumerate(plans):
            self.logger.info(f"Executing plan {index + 1}/{len(plans)} :: {plan.label}")
            model_bundle = get_model_bundle(plan.checkpoint_name)
            active_model, active_clip, active_vae = model_bundle
            cfg_current = plan.overrides.get("cfg", cfg)
            steps_current = plan.overrides.get("steps", steps)
            denoise_current = plan.overrides.get("denoise", denoise)
            sampler_current = plan.overrides.get("sampler", sampler_name)
            scheduler_current = plan.overrides.get("scheduler", scheduler_name)
            seed_current = plan.overrides.get("seed", seed)
            patched_model, patched_clip = self._apply_loras(active_model, active_clip, plan.loras)
            if positive_override_source is not None:
                positive = self._clone_conditioning(positive_override_source)
            else:
                effective_prompt = positive_prompt
                if plan.prompt_suffix:
                    effective_prompt = f"{positive_prompt}\n{plan.prompt_suffix}".strip()
                positive = self._encode_prompt(patched_clip, effective_prompt)

            if negative_override_source is not None:
                negative = self._clone_conditioning(negative_override_source)
            else:
                baseline_negative = negative_prompt if negative_prompt.strip() else ""
                if cfg_current > 1.0 and baseline_negative:
                    negative = self._encode_prompt(patched_clip, baseline_negative)
                else:
                    negative = self._encode_prompt(patched_clip, "")
            if plan.checkpoint_name == checkpoint:
                latent_copy = clone_latent(base_latent)
            else:
                latent_seed = self._prepare_latent(active_vae, width, height, latent_in, image_in)
                latent_copy = clone_latent(latent_seed)
            result_latent = self._run_sampler(
                model=patched_model,
                vae=active_vae,
                positive=positive,
                negative=negative,
                latent=latent_copy,
                seed=seed_current,
                steps=steps_current,
                cfg=cfg_current,
                denoise=denoise_current,
                sampler_name=sampler_current,
                scheduler_name=scheduler_current,
            )
            try:
                decoded_images = self._decode(active_vae, result_latent)
            except Exception as exc:
                raise RuntimeError(f"Failed to decode sampler output for plan {plan.label}: {exc}") from exc
            decoded_clone = decoded_images.detach().clone()
            decoded_cpu = decoded_clone.to(device="cpu")
            grid_images.append(decoded_cpu)
            image_stack.append(decoded_cpu)
            latent_samples = result_latent.get("samples") if isinstance(result_latent, dict) else None
            plan_summary: Dict[str, Any] = {
                "label": plan.label,
                "checkpoint": plan.checkpoint_name or checkpoint,
                "seed": seed_current,
                "steps": steps_current,
                "cfg": cfg_current,
                "denoise": denoise_current,
                "sampler": sampler_current,
                "scheduler": scheduler_current,
                "decoded_shape": tuple(decoded_cpu.shape),
            }
            if torch.is_tensor(latent_samples):
                plan_summary["latent_shape"] = tuple(latent_samples.shape)
            if torch.is_tensor(decoded_cpu):
                plan_summary["pixel_range"] = (
                    float(decoded_cpu.min().item()),
                    float(decoded_cpu.max().item()),
                )
            plan_summaries.append(plan_summary)
            final_latent = result_latent
            final_positive = positive
            final_negative = negative
            if torch.cuda.is_available():  # pragma: no cover - device-specific
                torch.cuda.empty_cache()
        model_management.soft_empty_cache()

        grid = compose_image_grid(grid_images, grid_rows, grid_cols)
        grid = annotate_grid_image(grid, plans, grid_rows, grid_cols, self.logger)
        stacked = torch.cat(image_stack, dim=0) if image_stack else grid
        try:
            execution_report = self._build_report(plans, plan_summaries, (grid_rows, grid_cols))
            self.logger.trace(execution_report)
        except Exception as exc:
            self.logger.debug(f"Generation report unavailable: {exc}")
        self.logger.info("Engine pipeline complete")
        latent_export = clone_latent(final_latent or base_latent)
        if final_positive is None:
            fallback_prompt = positive_prompt or ""
            final_positive = self._encode_prompt(final_clip, fallback_prompt) if final_clip is not None else []
        if final_negative is None:
            fallback_negative_prompt = negative_prompt if negative_prompt.strip() else ""
            if final_clip is not None:
                final_negative = self._encode_prompt(final_clip, fallback_negative_prompt)
            else:
                final_negative = []
        positive_out = self._clone_conditioning(final_positive)
        negative_out = self._clone_conditioning(final_negative)
        return (
            stacked,
            grid,
            final_model,
            final_clip,
            final_vae,
            latent_export,
            positive_out,
            negative_out,
        )


class h4_DebugATron3000:
    """Adaptive router that mirrors signals per data type while logging everything."""

    SLOT_DEFINITIONS: Tuple[Tuple[str, str, str], ...] = DEBUG_SLOT_DEFINITIONS
    SLOT_TOOLTIPS: Dict[str, str] = DEBUG_SLOT_TOOLTIPS
    VAE_MODEL_FALLBACKS: Dict[str, str] = {"vae_in": "model_in"}
    BRANCH_DISPLAY_NAMES: Dict[str, str] = {}

    _VAE_ATTR_CANDIDATES: Tuple[str, ...] = (
        "vae",
        "first_stage_model",
        "autoencoder",
        "model_vae",
    )

    _MODEL_ATTR_CANDIDATES: Tuple[str, ...] = ("model", "inner_model", "diffusion_model")
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802
        optional_inputs: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for slot_name, display_name, slot_type in cls.SLOT_DEFINITIONS:
            tooltip = cls.SLOT_TOOLTIPS.get(slot_name)
            extra: Dict[str, Any] = {"tooltip": tooltip} if tooltip else {}
            extra.setdefault("default", None)
            extra.setdefault("label", display_name)
            optional_inputs[slot_name] = (slot_type, extra)
        optional_inputs["go_ultra"] = (
            "BOOLEAN",
            {
                "default": False,
                "label": " GO PLUS ULTRA?! ",
                "tooltip": "Flip the switch marked GO PLUS ULTRA?! to open deep-dive logging, previews, and artefact capture.",
            },
        )
        for control in ULTRA_CONTROL_DEFINITIONS:
            options = {
                "default": control["default"],
                "label": control["label"],
                "tooltip": control["tooltip"],
            }
            if control["type"] == "STRING":
                options.setdefault("placeholder", "torch.sqrt(latent_in['samples'].var())")
            optional_inputs[control["name"]] = (control["type"], options)
        return {
            "required": {
                "mode": (
                    ["monitor", "passthrough"],
                    {"default": "passthrough"},
                ),
            },
            "optional": optional_inputs,
        }

    @classmethod
    def _slot_display_lookup(cls) -> Dict[str, Tuple[str, str]]:
        return {name: (display, slot_type) for name, display, slot_type in cls.SLOT_DEFINITIONS}

    def _slot_branch_suffix(self, slot_name: str) -> Optional[str]:
        for suffix in self.BRANCH_DISPLAY_NAMES:
            if slot_name.endswith(suffix):
                base = slot_name[: -len(suffix)]
                candidate = f"{base}{suffix}"
                if candidate == slot_name:
                    return suffix
        return None

    def _slot_base_name(self, slot_name: str) -> str:
        suffix = self._slot_branch_suffix(slot_name)
        if suffix:
            return slot_name[: -len(suffix)]
        return slot_name

    def _branch_label(self, suffix: Optional[str]) -> Optional[str]:
        if suffix is None:
            return None
        return self.BRANCH_DISPLAY_NAMES.get(suffix)

    def _slot_variants(self, base_name: str) -> List[str]:
        variants: List[str] = []
        for slot_name, _display, _slot_type in self.SLOT_DEFINITIONS:
            if slot_name == base_name:
                variants.append(slot_name)
                continue
            suffix = self._slot_branch_suffix(slot_name)
            if suffix and self._slot_base_name(slot_name) == base_name:
                variants.append(slot_name)
        return variants

    def _matching_branch_slot(self, base_name: str, target_slot: str) -> Optional[str]:
        variants = self._slot_variants(base_name)
        if not variants:
            return None
        target_suffix = self._slot_branch_suffix(target_slot)
        if target_suffix:
            for variant in variants:
                if self._slot_branch_suffix(variant) == target_suffix:
                    return variant
        return variants[0]

    RETURN_TYPES: Tuple[str, ...] = tuple(slot[2] for slot in SLOT_DEFINITIONS)
    RETURN_NAMES: Tuple[str, ...] = tuple(slot[1] for slot in SLOT_DEFINITIONS)
    FUNCTION = "route"
    CATEGORY = "h4 Toolkit/Debug"

    def __init__(self) -> None:
        self.logger = ToolkitLogger("h4_DebugATron3000")

    def _summarise_tensor(self, tensor: torch.Tensor) -> str:
        if tensor is None:
            return "<none>"
        return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"

    def _tensor_stats(self, tensor: torch.Tensor) -> List[str]:
        stats: List[str] = []
        if tensor is None:
            return stats
        try:
            with torch.no_grad():
                flat = tensor.float() if tensor.is_floating_point() else tensor
                stats.append(
                    "min={:.4f} max={:.4f}".format(
                        float(flat.min().item()), float(flat.max().item())
                    )
                )
                if tensor.is_floating_point():
                    stats.append("mean={:.4f} std={:.4f}".format(
                        float(tensor.mean().item()), float(tensor.std(unbiased=False).item())
                    ))
        except Exception as exc:  # pragma: no cover - diagnostics best effort
            stats.append(f"stats unavailable ({exc})")
        return stats

    def _summarise_object(self, obj: Any, *, depth: int = 0) -> str:
        if obj is None:
            return "<none>"
        if torch.is_tensor(obj):
            return self._summarise_tensor(obj)
        if isinstance(obj, (list, tuple)):
            length = len(obj)
            if length == 0:
                return f"{type(obj).__name__}[0]"
            if depth >= 1:
                return f"{type(obj).__name__}[{length}]"
            head = obj[0]
            head_summary = self._summarise_object(head, depth=depth + 1)
            return f"{type(obj).__name__}[{length}] first={head_summary}"
        if isinstance(obj, dict):
            keys = list(obj.keys())
            preview = ", ".join(str(key) for key in keys[:4])
            if len(keys) > 4:
                preview += ", …"
            return f"dict[{len(keys)}] keys={{{{ {preview} }}}}" if keys else "dict[0]"
        label = None
        for attr in ("name", "model_name", "filename"):
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value:
                label = value
                break
        if label:
            return f"{type(obj).__name__}<{label}>"
        return f"{type(obj).__name__}"

    def _describe_payload(self, payload: Any, slot_type: str) -> List[str]:
        details: List[str] = []
        if payload is None:
            return details
        if torch.is_tensor(payload):
            details.append(self._summarise_tensor(payload))
            details.extend(self._tensor_stats(payload))
            return details
        if slot_type == "LATENT" and isinstance(payload, dict):
            for key, value in payload.items():
                summary = self._summarise_object(value)
                line = f"{key}: {summary}"
                details.append(line)
                if torch.is_tensor(value):
                    for stat in self._tensor_stats(value):
                        details.append(f"    {stat}")
            return details
        if slot_type == "CONDITIONING" and isinstance(payload, (list, tuple)):
            details.append(f"entries={len(payload)}")
            for idx, item in enumerate(payload[:3]):
                if isinstance(item, (list, tuple)) and item:
                    head = item[0]
                    head_summary = self._summarise_object(head)
                    details.append(f"[{idx}] primary={head_summary}")
                    if torch.is_tensor(head):
                        for stat in self._tensor_stats(head):
                            details.append(f"    {stat}")
                else:
                    details.append(f"[{idx}] {self._summarise_object(item)}")
            if len(payload) > 3:
                details.append("… truncated …")
            return details
        if isinstance(payload, dict):
            details.append(self._summarise_object(payload))
            return details
        if isinstance(payload, (list, tuple)):
            details.append(self._summarise_object(payload))
            return details
        return details

    def _ensure_ultra_directory(self) -> Optional[str]:
        base_dir: Optional[str] = None
        get_temp_dir = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp_dir):
            try:
                base_dir = get_temp_dir()
            except Exception:  # pragma: no cover - environment specific
                base_dir = None
        if not base_dir:
            get_output_dir = getattr(folder_paths, "get_output_directory", None)
            if callable(get_output_dir):
                try:
                    base_dir = get_output_dir()
                except Exception:  # pragma: no cover
                    base_dir = None
        if not base_dir:
            return None
        target = os.path.join(base_dir, "h4_debugatron_ultra")
        try:
            os.makedirs(target, exist_ok=True)
        except Exception as exc:  # pragma: no cover - filesystem issues
            self.logger.warn(f"Unable to create Go_Plus_ULTRA artefact directory: {exc}")
            return None
        return target

    def _tensor_to_image(self, tensor: torch.Tensor) -> Optional[Any]:
        if Image is None or tensor is None:
            return None
        try:
            sample = tensor.detach()
            if sample.ndim == 4:
                sample = sample[0]
            if sample.ndim == 3 and sample.shape[0] in {1, 3, 4}:
                sample = sample.permute(1, 2, 0)
            sample = sample.to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
            data = sample.numpy()
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            if data.shape[-1] == 1:
                data = np.repeat(data, 3, axis=-1)
            elif data.shape[-1] == 2:
                padding = np.zeros((*data.shape[:-1], 1), dtype=data.dtype)
                data = np.concatenate([data, padding], axis=-1)
            data_uint8 = np.clip(data * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(data_uint8)
        except Exception as exc:  # pragma: no cover - diagnostic best effort
            self.logger.warn(f"Failed to convert tensor to image preview: {exc}")
            return None

    def _decode_latent_tensor(self, vae: Any, latent: Any) -> Optional[torch.Tensor]:
        if latent is None or vae is None:
            return None
        tensor = None
        if isinstance(latent, dict):
            tensor = latent.get("samples")
        elif torch.is_tensor(latent):
            tensor = latent
        if tensor is None:
            return None
        decode_candidates: List[Callable[..., Any]] = []
        if latent_preview is not None:
            for name in ("decode_latent_preview", "decode_latent", "decode"):
                candidate = getattr(latent_preview, name, None)
                if callable(candidate):
                    decode_candidates.append(candidate)
        comfy_decode = getattr(comfy_utils, "decode_latent_preview", None)
        if callable(comfy_decode):
            decode_candidates.append(comfy_decode)
        for candidate in decode_candidates:
            try:
                decoded = candidate(vae, latent if isinstance(latent, dict) else {"samples": tensor})
                if isinstance(decoded, (tuple, list)):
                    decoded = decoded[0]
                if isinstance(decoded, dict):
                    decoded = decoded.get("image") or decoded.get("samples")
                if torch.is_tensor(decoded):
                    return decoded
            except Exception:
                continue
        direct_decode = getattr(vae, "decode", None)
        if callable(direct_decode):
            try:
                decoded = direct_decode(tensor)
                if isinstance(decoded, (tuple, list)):
                    decoded = decoded[0]
                if isinstance(decoded, dict):
                    decoded = decoded.get("image") or decoded.get("samples")
                if torch.is_tensor(decoded):
                    return decoded
            except Exception:
                return None
        return None

    def _collect_latent_snapshots(
        self,
        latent: Any,
        capture_first: bool,
        capture_mid: bool,
        capture_last: bool,
    ) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        if latent is None:
            return snapshots
        history: List[torch.Tensor] = []
        if isinstance(latent, dict):
            for key in ("samples_history", "history", "frames"):
                candidate = latent.get(key)
                if isinstance(candidate, (list, tuple)) and candidate:
                    history = [item for item in candidate if torch.is_tensor(item)]
                    if history:
                        break
            samples = latent.get("samples")
            if torch.is_tensor(samples):
                history = history or [samples]
        elif torch.is_tensor(latent):
            history = [latent]
        if not history:
            return snapshots
        picked_indices: List[Tuple[int, str]] = []
        if capture_first and history:
            picked_indices.append((0, "first"))
        if capture_mid and len(history) > 2:
            picked_indices.append((len(history) // 2, "mid"))
        elif capture_mid and history:
            picked_indices.append((0, "mid"))
        if capture_last and history:
            picked_indices.append((len(history) - 1, "last"))
        seen = set()
        for index, label in picked_indices:
            index = max(0, min(index, len(history) - 1))
            key = (index, label)
            if key in seen:
                continue
            seen.add(key)
            tensor = history[index]
            snapshots.append({"label": label, "tensor": tensor})
        return snapshots

    def _tensor_anomalies(self, tensor: torch.Tensor, label: str) -> List[str]:
        anomalies: List[str] = []
        if tensor is None:
            return anomalies
        try:
            finite_mask = torch.isfinite(tensor) if tensor.is_floating_point() else None
            if tensor.is_floating_point():
                if torch.isnan(tensor).any():
                    anomalies.append(f"{label}: contains NaN values")
                if torch.isinf(tensor).any():
                    anomalies.append(f"{label}: contains Inf values")
                if finite_mask is not None and finite_mask.any():
                    std_value = float(tensor[finite_mask].float().std(unbiased=False).item())
                    if std_value > 50.0:
                        anomalies.append(f"{label}: unusually high std {std_value:.2f}")
            if tensor.ndim >= 2:
                max_abs = float(tensor.detach().abs().max().item())
                if max_abs > 1000:
                    anomalies.append(f"{label}: extreme magnitude {max_abs:.2f}")
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            anomalies.append(f"{label}: anomaly check failed ({exc})")
        return anomalies

    def _evaluate_watch_expression(self, expression: str, context: Dict[str, Any]) -> Any:
        safe_globals = {"torch": torch, "np": np}
        try:
            return eval(expression, {"__builtins__": {}}, {**safe_globals, **context})
        except Exception as exc:
            self.logger.warn(f"Watch expression '{expression}' failed: {exc}")
            return None

    def _preview_conditioning_tokens(self, conditioning: Any) -> List[str]:
        previews: List[str] = []
        if not isinstance(conditioning, (list, tuple)) or not conditioning:
            return previews
        for idx, entry in enumerate(conditioning[:3]):
            fragments: List[str] = []
            if isinstance(entry, (list, tuple)):
                for component in entry:
                    if isinstance(component, dict):
                        for key in ("prompt", "text", "tokens"):
                            fragment = component.get(key)
                            if isinstance(fragment, str) and fragment.strip():
                                fragments.append(fragment.strip())
                    elif isinstance(component, str) and component.strip():
                        fragments.append(component.strip())
            elif isinstance(entry, dict):
                for key in ("prompt", "text", "tokens"):
                    fragment = entry.get(key)
                    if isinstance(fragment, str) and fragment.strip():
                        fragments.append(fragment.strip())
            elif isinstance(entry, str) and entry.strip():
                fragments.append(entry.strip())
            if fragments:
                merged = " | ".join(fragments)
                previews.append(f"[{idx}] {merged[:200]}")
        return previews

    def _fingerprint_model(self, model: Any) -> Optional[str]:
        if model is None or not hasattr(model, "state_dict"):
            return None
        try:
            state = model.state_dict()
        except Exception as exc:
            self.logger.warn(f"Unable to access model state_dict for fingerprinting: {exc}")
            return None
        hasher = hashlib.sha1()
        try:
            for key in sorted(state.keys()):
                value = state[key]
                hasher.update(key.encode("utf-8"))
                if isinstance(value, torch.Tensor):
                    hasher.update(str(tuple(value.shape)).encode("utf-8"))
                    hasher.update(str(value.dtype).encode("utf-8"))
                    flat = value.detach().view(-1)
                    sample = flat[: min(flat.numel(), 16)].to(device="cpu", dtype=torch.float32)
                    hasher.update(sample.numpy().tobytes())
                else:
                    hasher.update(str(type(value)).encode("utf-8"))
        except Exception as exc:
            self.logger.warn(f"Model fingerprinting failed: {exc}")
            return None
        return hasher.hexdigest()

    def _write_ultra_json_report(
        self,
        payload: Dict[str, Any],
        cache_artifacts: bool,
    ) -> Optional[str]:
        target_dir = self._ensure_ultra_directory()
        if target_dir is None or not cache_artifacts:
            return None
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(target_dir, f"debugatron_ultra_{timestamp}.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            return path
        except Exception as exc:
            self.logger.warn(f"Failed to persist Go_Plus_ULTRA JSON log: {exc}")
            return None

    def _extract_vae_from_model(self, model: Any) -> Optional[Any]:
        if model is None:
            return None
        candidate = self._find_attr_chain(model, self._VAE_ATTR_CANDIDATES)
        if candidate is not None:
            return candidate
        for intermediary_name in self._MODEL_ATTR_CANDIDATES:
            intermediary = getattr(model, intermediary_name, None)
            if intermediary is None:
                continue
            candidate = self._find_attr_chain(intermediary, self._VAE_ATTR_CANDIDATES)
            if candidate is not None:
                return candidate
        return None

    def _find_attr_chain(self, obj: Any, names: Iterable[str]) -> Optional[Any]:
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return None

    def _clone_payload_by_type(self, payload: Any, slot_type: str) -> Any:
        if payload is None:
            return None
        if slot_type == "LATENT":
            return clone_latent(payload)
        if slot_type == "CONDITIONING":
            return clone_conditioning_payload(payload)
        if slot_type in {"IMAGE", "MASK"}:
            return _clone_generic_payload(payload)
        if slot_type in {"MODEL", "CLIP", "CLIP_VISION", "VAE"}:
            return payload
        return _clone_generic_payload(payload)

    def _clone_payload_safely(self, payload: Any, slot_type: str) -> Any:
        if payload is None:
            return None
        try:
            return self._clone_payload_by_type(payload, slot_type)
        except Exception as exc:
            self.logger.warn(
                f"Failed to clone {slot_type} payload for debug router; passing original. {exc}"
            )
            return payload

    def route(
        self,
        mode: str,
        go_ultra: bool = False,
        ultra_capture_first_step: bool = ULTRA_CONTROL_DEFAULTS["ultra_capture_first_step"],
        ultra_capture_mid_step: bool = ULTRA_CONTROL_DEFAULTS["ultra_capture_mid_step"],
        ultra_capture_last_step: bool = ULTRA_CONTROL_DEFAULTS["ultra_capture_last_step"],
        ultra_preview_images: bool = ULTRA_CONTROL_DEFAULTS["ultra_preview_images"],
        ultra_json_log: bool = ULTRA_CONTROL_DEFAULTS["ultra_json_log"],
        ultra_highlight_missing_conditioning: bool = ULTRA_CONTROL_DEFAULTS["ultra_highlight_missing_conditioning"],
        ultra_token_preview: bool = ULTRA_CONTROL_DEFAULTS["ultra_token_preview"],
        ultra_latent_anomaly_checks: bool = ULTRA_CONTROL_DEFAULTS["ultra_latent_anomaly_checks"],
        ultra_model_diff_tracking: bool = ULTRA_CONTROL_DEFAULTS["ultra_model_diff_tracking"],
        ultra_watch_expression: str = ULTRA_CONTROL_DEFAULTS["ultra_watch_expression"],
        ultra_cache_artifacts: bool = ULTRA_CONTROL_DEFAULTS["ultra_cache_artifacts"],
        **dynamic_slots: Any,
    ) -> Tuple[Any, ...]:
        self.logger.info(f"Debug router engaged :: mode={mode}")
        prefix = "[h4_Debug-a-Tron-3000]"
        outputs: List[Any] = []
        connections = 0
        summary_lines: List[str] = []
        slot_reports: List[Dict[str, Any]] = []
        anomaly_messages: List[str] = []
        missing_conditioning: List[str] = []
        token_previews: Dict[str, List[str]] = {}
        preview_paths: List[str] = []
        fallback_notes: List[str] = []
        context_snapshot = dict(dynamic_slots)
        context_snapshot.update(
            {
                "go_ultra": go_ultra,
                "ultra_capture_first_step": ultra_capture_first_step,
                "ultra_capture_mid_step": ultra_capture_mid_step,
                "ultra_capture_last_step": ultra_capture_last_step,
                "ultra_preview_images": ultra_preview_images,
                "ultra_json_log": ultra_json_log,
                "ultra_highlight_missing_conditioning": ultra_highlight_missing_conditioning,
                "ultra_token_preview": ultra_token_preview,
                "ultra_latent_anomaly_checks": ultra_latent_anomaly_checks,
                "ultra_model_diff_tracking": ultra_model_diff_tracking,
                "ultra_watch_expression": ultra_watch_expression,
                "ultra_cache_artifacts": ultra_cache_artifacts,
            }
        )
        slot_lookup = self._slot_display_lookup()
        branch_stats: Dict[str, Dict[str, Any]] = {
            suffix: {"label": label, "total": 0, "connected": 0, "slots": []}
            for suffix, label in self.BRANCH_DISPLAY_NAMES.items()
        }
        for slot_name, display_name, slot_type in self.SLOT_DEFINITIONS:
            value = dynamic_slots.get(slot_name)
            if value is None and slot_name in self.VAE_MODEL_FALLBACKS:
                model_slot = self.VAE_MODEL_FALLBACKS[slot_name]
                model_payload = dynamic_slots.get(model_slot)
                fallback = self._extract_vae_from_model(model_payload)
                if fallback is not None:
                    value = fallback
                    dynamic_slots[slot_name] = fallback
                    context_snapshot[slot_name] = fallback
                    source_display, _ = slot_lookup.get(model_slot, (model_slot, ""))
                    fallback_message = (
                        f"{display_name} missing; adopted VAE from {source_display}"
                    )
                    fallback_notes.append(fallback_message)
                    self.logger.info(f"{prefix} :: {fallback_message}")
            descriptor = "<disconnected>"
            details: List[str] = []
            if value is not None:
                connections += 1
                descriptor = self._summarise_object(value)
                message = f"{prefix} :: {display_name} ({slot_type}) -> {descriptor}"
                details = self._describe_payload(value, slot_type)
                summary_lines.append(f"{display_name} ({slot_type}) -> {descriptor}")
                summary_lines.extend(f"    {line}" for line in details)
                self.logger.info(message)
                for detail in details:
                    self.logger.info(f"    {detail}")
            else:
                self.logger.trace(f"{prefix} :: {display_name} ({slot_type}) -> <disconnected>")
            cloned_value = (
                self._clone_payload_safely(value, slot_type)
                if mode in {"passthrough", "monitor"}
                else None
            )
            outputs.append(cloned_value)
            slot_reports.append(
                {
                    "slot_name": slot_name,
                    "display_name": display_name,
                    "slot_type": slot_type,
                    "connected": value is not None,
                    "descriptor": descriptor,
                    "details": details,
                }
            )
            branch_suffix = self._slot_branch_suffix(slot_name)
            if branch_suffix and branch_suffix in branch_stats:
                info = branch_stats[branch_suffix]
                info["total"] += 1
                if value is not None:
                    info["connected"] += 1
                info["slots"].append(
                    {
                        "display_name": display_name,
                        "connected": value is not None,
                        "descriptor": descriptor if value is not None else None,
                    }
                )
            if go_ultra and ultra_latent_anomaly_checks and value is not None:
                if torch.is_tensor(value):
                    anomaly_messages.extend(self._tensor_anomalies(value, display_name))
                elif isinstance(value, dict):
                    for key, tensor in value.items():
                        if torch.is_tensor(tensor):
                            anomaly_messages.extend(
                                self._tensor_anomalies(tensor, f"{display_name}.{key}")
                            )
        if connections == 0:
            self.logger.info("Debug router found no connected inputs")
        else:
            self.logger.info(f"Debug router captured {connections} input(s)")

        latent_slots = self._slot_variants("latent_in")
        vae_slots = self._slot_variants("vae_in")
        latent_payloads: Dict[str, Any] = {name: dynamic_slots.get(name) for name in latent_slots}
        vae_payloads: Dict[str, Any] = {name: dynamic_slots.get(name) for name in vae_slots}

        if go_ultra and ultra_highlight_missing_conditioning:
            for base_name, label in CONDITIONING_SLOT_LABELS.items():
                for slot_name in self._slot_variants(base_name):
                    candidate = dynamic_slots.get(slot_name)
                    display_label = label
                    branch_label = self._branch_label(self._slot_branch_suffix(slot_name))
                    if branch_label:
                        display_label = f"{label} ({branch_label})"
                    if candidate is None or (isinstance(candidate, (list, tuple)) and len(candidate) == 0):
                        warning = f"Go_Plus_ULTRA :: {display_label} input appears to be missing"
                        missing_conditioning.append(warning)
                        self.logger.warn(warning)

        if go_ultra and ultra_token_preview:
            for base_name, label in (
                ("conditioning_in", "Conditioning"),
                ("conditioning_positive_in", "Positive"),
                ("conditioning_negative_in", "Negative"),
            ):
                for slot_name in self._slot_variants(base_name):
                    candidate = dynamic_slots.get(slot_name)
                    branch_label = self._branch_label(self._slot_branch_suffix(slot_name))
                    display_label = label if branch_label is None else f"{label} ({branch_label})"
                    snippets = self._preview_conditioning_tokens(candidate)
                    if snippets:
                        token_previews[display_label] = snippets
                        self.logger.info(f"Go_Plus_ULTRA :: {display_label} token preview:")
                        for snippet in snippets:
                            self.logger.info(f"    {snippet}")

        latent_snapshots_map: Dict[str, List[Dict[str, Any]]] = {}
        if go_ultra:
            for slot_name, payload in latent_payloads.items():
                if payload is None:
                    continue
                snapshots = self._collect_latent_snapshots(
                    payload,
                    ultra_capture_first_step,
                    ultra_capture_mid_step,
                    ultra_capture_last_step,
                )
                if snapshots:
                    latent_snapshots_map[slot_name] = snapshots
        for slot_name, snapshots in latent_snapshots_map.items():
            slot_display, _slot_type = slot_lookup.get(slot_name, (slot_name, "LATENT"))
            branch_suffix = self._slot_branch_suffix(slot_name)
            branch_label = self._branch_label(branch_suffix)
            vae_slot = self._matching_branch_slot("vae_in", slot_name)
            vae_payload = vae_payloads.get(vae_slot) if vae_slot else None
            for snapshot in snapshots:
                tensor = snapshot.get("tensor")
                if tensor is None:
                    continue
                label = snapshot.get("label", "snapshot")
                summary_label = f"{slot_display} [{label}]"
                summary = self._summarise_tensor(tensor)
                summary_lines.append(f"Snapshot[{summary_label}] -> {summary}")
                self.logger.info(f"Go_Plus_ULTRA :: Snapshot[{summary_label}] -> {summary}")
                if go_ultra and ultra_latent_anomaly_checks:
                    label_descriptor = summary_label if branch_label is None else f"{branch_label} {label}"
                    anomaly_messages.extend(self._tensor_anomalies(tensor, f"Latent snapshot {label_descriptor}"))
                if go_ultra and ultra_preview_images and vae_payload is not None:
                    decoded = self._decode_latent_tensor(vae_payload, {"samples": tensor})
                    image_tensor = decoded if torch.is_tensor(decoded) else None
                    image = self._tensor_to_image(image_tensor) if image_tensor is not None else None
                    if image is not None:
                        if ultra_cache_artifacts:
                            directory = self._ensure_ultra_directory()
                            if directory is not None:
                                filename = (
                                    f"ultra_preview_{slot_name}_{label}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png"
                                )
                                path = os.path.join(directory, filename)
                                try:
                                    image.save(path)
                                    preview_paths.append(path)
                                    summary_lines.append(f"Preview[{summary_label}] saved -> {path}")
                                    self.logger.info(
                                        f"Go_Plus_ULTRA :: Saved preview image ({summary_label}) -> {path}"
                                    )
                                except Exception as exc:
                                    self.logger.warn(
                                        f"Go_Plus_ULTRA :: Failed to write preview image ({summary_label}): {exc}"
                                    )
                        else:
                            summary_lines.append(
                                f"Preview[{summary_label}] generated (caching disabled)"
                            )
                            self.logger.info(
                                f"Go_Plus_ULTRA :: Preview image ({summary_label}) generated (caching disabled)"
                            )

        if branch_stats:
            branch_summary_lines: List[str] = []
            for suffix, info in branch_stats.items():
                if info["total"] == 0:
                    continue
                line = f"{info['label']}: {info['connected']}/{info['total']} connected"
                branch_summary_lines.append(line)
                for slot_info in info["slots"]:
                    if not slot_info["connected"]:
                        branch_summary_lines.append(f"    missing -> {slot_info['display_name']}")
            if branch_summary_lines:
                summary_lines.append("Branch summaries:")
                summary_lines.extend(branch_summary_lines)

        fingerprint_records: Dict[str, Optional[str]] = {}
        if go_ultra and ultra_model_diff_tracking:
            for slot_name, display_name in (
                ("model_in", "Model"),
                ("vae_in", "VAE"),
                ("clip_in", "CLIP"),
            ):
                payload = dynamic_slots.get(slot_name)
                fingerprint = self._fingerprint_model(payload)
                if fingerprint:
                    fingerprint_records[display_name] = fingerprint
                    summary_lines.append(f"Fingerprint[{display_name}] -> {fingerprint}")
                    self.logger.info(f"Go_Plus_ULTRA :: {display_name} fingerprint {fingerprint}")

        watch_value: Any = None
        if go_ultra and ultra_watch_expression and ultra_watch_expression.strip():
            watch_value = self._evaluate_watch_expression(ultra_watch_expression, context_snapshot)
            if watch_value is not None:
                summary_lines.append(f"Watch[{ultra_watch_expression}] -> {watch_value}")
                self.logger.info(f"Go_Plus_ULTRA :: Watch[{ultra_watch_expression}] -> {watch_value}")

        if go_ultra and ultra_json_log:
            json_payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": mode,
                "connections": connections,
                "settings": {
                    "capture_first": ultra_capture_first_step,
                    "capture_mid": ultra_capture_mid_step,
                    "capture_last": ultra_capture_last_step,
                    "preview_images": ultra_preview_images,
                    "json_log": ultra_json_log,
                    "highlight_missing_conditioning": ultra_highlight_missing_conditioning,
                    "token_preview": ultra_token_preview,
                    "latent_anomaly_checks": ultra_latent_anomaly_checks,
                    "model_diff_tracking": ultra_model_diff_tracking,
                    "watch_expression": ultra_watch_expression,
                    "cache_artifacts": ultra_cache_artifacts,
                },
                "slots": slot_reports,
                "anomalies": anomaly_messages,
                "missing_conditioning": missing_conditioning,
                "token_previews": token_previews,
                "preview_paths": preview_paths,
                "fingerprints": fingerprint_records,
                "watch_result": str(watch_value) if watch_value is not None else None,
            }
            json_path = self._write_ultra_json_report(json_payload, ultra_cache_artifacts)
            if json_path:
                summary_lines.append(f"JSON log saved -> {json_path}")
                self.logger.info(f"Go_Plus_ULTRA :: JSON log saved -> {json_path}")
            elif ultra_cache_artifacts:
                self.logger.warn("Go_Plus_ULTRA :: Requested JSON persistence failed")
            else:
                summary_lines.append("JSON log generated (not persisted)")
                self.logger.info("Go_Plus_ULTRA :: JSON log generated (caching disabled)")

        if go_ultra and ultra_latent_anomaly_checks:
            for anomaly in anomaly_messages:
                self.logger.warn(f"Go_Plus_ULTRA :: {anomaly}")
            summary_lines.extend(f"Anomaly: {msg}" for msg in anomaly_messages)

        if missing_conditioning:
            summary_lines.extend(f"WARNING: {msg}" for msg in missing_conditioning)

        if fallback_notes:
            summary_lines.extend(f"NOTE: {note}" for note in fallback_notes)

        status_lines = [
            f"Mode: {mode}",
            f"Connections: {connections}",
            f"Go_Plus_ULTRA: {'ON' if go_ultra else 'OFF'}",
        ]
        if token_previews:
            summary_lines.append("Token previews:")
            for label, snippets in token_previews.items():
                for snippet in snippets:
                    summary_lines.append(f"  {label}: {snippet}")
        if summary_lines:
            status_lines.append("")
            status_lines.extend(summary_lines)
        else:
            status_lines.append("")
            status_lines.append("No inputs connected.")

        status_text = "\n".join(status_lines)
        self.logger.info(status_text)
        self.logger.info("Debug routing complete")
        return tuple(outputs)


class h4_DebugATronRouter(h4_DebugATron3000):
    """Secondary router variant that always forwards signals downstream."""

    SLOT_DEFINITIONS = ROUTER_SLOT_DEFINITIONS
    SLOT_TOOLTIPS = ROUTER_SLOT_TOOLTIPS
    VAE_MODEL_FALLBACKS = ROUTER_VAE_MODEL_FALLBACKS
    BRANCH_DISPLAY_NAMES = ROUTER_BRANCH_DISPLAY_NAMES
    RETURN_TYPES = ROUTER_SLOT_RETURN_TYPES
    RETURN_NAMES = ROUTER_SLOT_RETURN_NAMES

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802
        config = super().INPUT_TYPES()
        config["required"]["mode"] = (["monitor", "passthrough"], {"default": "passthrough"})
        return config


NODE_CLASS_MAPPINGS = {
    "h4PlotXY": h4_PlotXY,
    "h4DebugATron3000": h4_DebugATron3000,
    "h4DebugATronRouter": h4_DebugATronRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "h4PlotXY": f"h4 : The Engine (Simple Sampler+Plot) v{PLOT_NODE_VERSION}",
    "h4DebugATron3000": f"h4 : Debug-a-Tron-3000 (v{DEBUG_NODE_VERSION})",
    "h4DebugATronRouter": f"h4 : Debug-a-Tron-3000 Router (v{DEBUG_NODE_VERSION})",
}

NODE_TOOLTIP_MAPPINGS = {
    "h4PlotXY": PLOT_WIDGET_TOOLTIPS,
    "h4DebugATron3000": DEBUG_WIDGET_TOOLTIPS,
    "h4DebugATronRouter": DEBUG_WIDGET_TOOLTIPS,
}

__all__ = [
    "h4_PlotXY",
    "h4_DebugATron3000",
    "h4_DebugATronRouter",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NODE_TOOLTIP_MAPPINGS",
]

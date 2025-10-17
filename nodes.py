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
import re
import textwrap
import time
import weakref
from dataclasses import dataclass, field
import copy
import html
import secrets
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union, cast
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
AXIS_DRIVER_VERSION = "0.6.0"
UTILITY_NODE_VERSION = "1.1.0"
VARIANATOR_VERSION = "1.0.0"

AXIS_DRIVER_MAX_ITEMS = 8
AXIS_DRIVER_SLOT_ORDER: Tuple[str, ...] = ("X", "Y", "Z")
AXIS_DRIVER_SUPPORTED_PRESETS: Tuple[str, ...] = (
    "none",
    "prompt",
    "checkpoint",
    "lora",
    "sampler",
    "scheduler",
    "steps",
    "cfg",
    "denoise",
    "seed",
)

AXIS_DRIVER_PRESET_DEFAULTS: Dict[str, str] = {
    "X": "checkpoint",
    "Y": "prompt",
    "Z": "none",
}

AXIS_DRIVER_DEFAULT_STYLE: Dict[str, Any] = {
    "font_size": 22,
    "font_family": "DejaVuSans",
    "font_colour": "#FFFFFF",
    "background": "black60",
    "alignment": "center",
    "label_position": "top_left",
    "label_layout": "overlay",
    "custom_label_x": "X",
    "custom_label_y": "Y",
    "custom_label_z": "Z",
    "show_axis_headers": True,
}

AXIS_DRIVER_DEFAULT_STATE: Dict[str, Any] = {
    "axes": [
        {"slot": "X", "preset": "checkpoint", "items": []},
        {"slot": "Y", "preset": "prompt", "items": []},
        {"slot": "Z", "preset": "none", "items": []},
    ],
    "style": AXIS_DRIVER_DEFAULT_STYLE,
}


UI_DISABLED_TOKEN = "NONE"


TRACE_ENABLED = os.getenv("H4_TOOLKIT_TRACE", "0") not in {"0", "false", "False", ""}

_PROXY_REGISTRY = weakref.WeakKeyDictionary()


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
        if "sigmas" in dropped:
            logger.warn(
                "Sampler compatibility shim discarded the sigma schedule parameter; sampler will recalc on its default device"
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
        message = exc.args[0] if exc.args else ""
        if isinstance(message, str) and "unexpected keyword argument" in message:
            logger.warn(f"Sampler compatibility shim retrying with raw kwargs due to signature mismatch: {message}")
            return sampler_fn(**sampler_kwargs)
        raise


class _DeviceAlignedDiffusionProxy:
    __slots__ = ("_inner", "_device", "_logger", "_sigma_fix_logged", "_call_logged", "__weakref__")

    def __init__(self, inner: Any, device: torch.device, logger: ToolkitLogger) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_logger", logger)
        object.__setattr__(self, "_sigma_fix_logged", False)
        object.__setattr__(self, "_call_logged", False)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def _coerce_timestep(self, timestep: Any, reference: Optional[torch.Tensor]) -> Any:
        device = object.__getattribute__(self, "_device")
        logger = object.__getattribute__(self, "_logger")
        logged = object.__getattribute__(self, "_sigma_fix_logged")
        if isinstance(timestep, torch.Tensor):
            if timestep.device != device:
                coerced = timestep.to(device=device)
                if not logged:
                    logger.warn(f"Diffusion timestep migrated {timestep.device} -> {device}")
                    object.__setattr__(self, "_sigma_fix_logged", True)
                return coerced
            return timestep
        if isinstance(timestep, (float, int)):
            dtype = reference.dtype if isinstance(reference, torch.Tensor) else torch.float32
            return torch.as_tensor(timestep, dtype=dtype, device=device)
        if hasattr(timestep, "to"):
            try:
                coerced = timestep.to(device=device)
                if not logged:
                    logger.warn("Diffusion timestep coerced to device via .to()")
                    object.__setattr__(self, "_sigma_fix_logged", True)
                return coerced
            except Exception:
                pass
        return timestep

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inner = object.__getattribute__(self, "_inner")
        args_list = list(args)
        reference_tensor: Optional[torch.Tensor] = None
        if args_list and isinstance(args_list[0], torch.Tensor):
            reference_tensor = args_list[0]
        elif isinstance(kwargs.get("xc"), torch.Tensor):
            reference_tensor = kwargs["xc"]
        timestep_value: Any = None
        timestep_index: Optional[int] = None
        if len(args_list) >= 2:
            timestep_index = 1
            timestep_value = args_list[1]
        elif "t" in kwargs:
            timestep_value = kwargs["t"]
        elif "timestep" in kwargs:
            timestep_value = kwargs["timestep"]
        logger = object.__getattribute__(self, "_logger")
        call_logged = object.__getattribute__(self, "_call_logged")
        if not call_logged:
            desc: str
            if isinstance(timestep_value, torch.Tensor):
                desc = f"tensor[{timestep_value.device}]"
            else:
                desc = type(timestep_value).__name__
            logger.info(f"[DeviceCheck] diffusion_model inbound timestep={desc}")
            object.__setattr__(self, "_call_logged", True)
        if timestep_value is not None:
            coerced = self._coerce_timestep(timestep_value, reference_tensor)
            if timestep_index is not None:
                args_list[timestep_index] = coerced
            elif "t" in kwargs:
                kwargs["t"] = coerced
            else:
                kwargs["timestep"] = coerced
        return inner(*tuple(args_list), **kwargs)

    def update_device(self, device: torch.device) -> None:
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_sigma_fix_logged", False)
        object.__setattr__(self, "_call_logged", False)


# ====================================================================================================
# Device Alignment Proxies
# ====================================================================================================
class _TimeEmbedProxy(torch.nn.Module):
    """A proxy specifically for the time_embed module to ensure its input is on the correct device."""
    def __init__(self, inner: torch.nn.Module, device: torch.device, logger: ToolkitLogger):
        super().__init__()
        self.inner = inner
        self.device = device
        self.logger = logger
        self._forward_logged = False

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        self.inner.to(self.device)
        if not self._forward_logged:
            self.logger.info("[DeviceCheck] time_embed proxy validating payload devices")
            self._forward_logged = True
        if temb.device != self.device:
            self.logger.warn(f"[DeviceCheck] Coercing time_embed input from {temb.device} to {self.device}")
            temb = temb.to(self.device)
        return self.inner(temb)

    def update_device(self, device: torch.device) -> None:
        self.device = device
        self.inner.to(device)
        self._forward_logged = False


class _ModuleDeviceProxy(torch.nn.Module):
    """Generic proxy that keeps a child module and its tensor inputs on a specific device."""

    def __init__(
        self,
        inner: torch.nn.Module,
        device: torch.device,
        logger: ToolkitLogger,
        module_label: str,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.device = device
        self.logger = logger
        self.module_label = module_label
        self._logged = False

    def _coerce_payload(self, value: Any) -> Any:
        return _DeviceAlignedModelProxy._payload_to_device(
            value,
            self.device,
            logger=self.logger,
            allow_dtype_adjustment=self.device.type == "cuda",
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.inner.to(self.device)
        if not self._logged:
            self.logger.info(f"[DeviceCheck] {self.module_label} proxy enforcing device alignment on {self.device}")
            self._logged = True
        coerced_args = tuple(self._coerce_payload(arg) for arg in args)
        coerced_kwargs = {key: self._coerce_payload(val) for key, val in kwargs.items()}
        return self.inner(*coerced_args, **coerced_kwargs)

    def update_device(self, device: torch.device) -> None:
        self.device = device
        self.inner.to(device)
        self._logged = False


class _DeviceAlignedModelProxy(torch.nn.Module):
    _calc_patch_applied: bool = False
    _calc_cond_batch_original: Optional[Callable[..., Any]] = None

    def __init__(self, inner: Any, device: torch.device, logger: ToolkitLogger) -> None:
        super().__init__()
        # Use a direct assignment to a private attribute to avoid torch.nn.Module's magic
        self._inner = inner
        self._device = device
        self._logger = logger
        self._sigma_fix_logged = False
        self._apply_logged = False
        self._child_proxies: Dict[str, "_DeviceAlignedModelProxy"] = {}
        self._delegated_attributes: Set[str] = set()
        self._module_proxies: List[torch.nn.Module] = []
        self._module_hook_targets: List[torch.nn.Module] = []

        logger.info(
            f"[DeviceCheck] installing model proxy for {inner.__class__.__name__} on {device}"
        )
        self._ensure_calc_patch(logger)

        # Landmark: Primary Module Device Sync
        self._move_target_to_device(self._inner, device, inner.__class__.__name__)

        # Landmark: Time Embed Proxy Injection
        # The time_embed module is often on a child object (e.g., diffusion_model), not the top-level model.
        # We need to find it and patch it there.
        target_for_patch = None
        diffusion_model = getattr(self._inner, "diffusion_model", None)
        if isinstance(diffusion_model, torch.nn.Module):
            self._move_target_to_device(diffusion_model, device, "diffusion_model")
        if diffusion_model is not None and hasattr(diffusion_model, "time_embed"):
            logger.info("[DeviceCheck] Found diffusion_model, targeting it for time_embed patch.")
            target_for_patch = diffusion_model
        elif hasattr(self._inner, "time_embed"):
            logger.info("[DeviceCheck] Found time_embed on main model, targeting it for patch.")
            target_for_patch = self._inner

        if target_for_patch:
            time_embed_module = getattr(target_for_patch, "time_embed", None)
            if time_embed_module is not None and isinstance(time_embed_module, torch.nn.Module):
                if not isinstance(time_embed_module, _TimeEmbedProxy):
                    try:
                        # Directly replace the attribute on the target module
                        proxy = _TimeEmbedProxy(time_embed_module, device, logger)
                        setattr(target_for_patch, "time_embed", proxy)
                        logger.info(f"[DeviceCheck] Time embed proxy installed on {target_for_patch.__class__.__name__}.")
                        self._module_proxies.append(proxy)

                    except Exception as exc:
                        logger.warn(f"Failed to install time_embed proxy: {exc}")
                else:
                    time_embed_module.update_device(device)
                    logger.info("[DeviceCheck] Time embed proxy was already installed.")
                    if time_embed_module not in self._module_proxies:
                        self._module_proxies.append(time_embed_module)
            else:
                logger.info("[DeviceCheck] Target for patch has no time_embed module or it's not a torch.nn.Module.")
        else:
            logger.info("[DeviceCheck] Could not find a suitable target for time_embed proxy installation.")

        # Landmark: Label Embedding Proxy Injection
        label_target = None
        if diffusion_model is not None and hasattr(diffusion_model, "label_emb"):
            label_target = diffusion_model
        elif hasattr(self._inner, "label_emb"):
            label_target = self._inner
        if label_target is not None:
            self._install_generic_module_proxy(label_target, "label_emb", "label_emb")
        else:
            logger.info("[DeviceCheck] No label_emb attribute detected for proxy installation.")

        # Landmark: Diffusion Block Proxy Installation
        if diffusion_model is not None:
            self._install_block_proxies(diffusion_model)
        else:
            self._install_block_proxies(self._inner)

        # Explicitly wrap the diffusion_model if it exists and isn't already a proxy
        diffusion_model = getattr(self._inner, "diffusion_model", None)
        if diffusion_model is not None and not isinstance(diffusion_model, _DeviceAlignedModelProxy):
            try:
                # This is a 'child' module, so we assign it directly.
                # PyTorch's __setattr__ will register it.
                self.diffusion_model = _DeviceAlignedModelProxy(diffusion_model, device, logger)
            except Exception as exc:
                logger.warn(f"Failed to monkey-patch diffusion_model attribute: {exc}")

        self._wrap_known_children()
        try:
            _PROXY_REGISTRY[inner] = self
        except TypeError:
            pass

    def __getattr__(self, name: str) -> Any:
        # Avoid recursion errors with nn.Module's internal attributes
        if name.startswith("_"):
            raise AttributeError(f"Private attribute '{name}' not found")
        
        # Check for our custom attributes first
        if name in {"_inner", "_device", "_logger", "_sigma_fix_logged", "_apply_logged", "_child_proxies", "_delegated_attributes", "_module_proxies", "_module_hook_targets"}:
            return self.__dict__[name]

        # If it's a proxied child, return it
        if name in self._child_proxies:
            return self._child_proxies[name]
            
        # Special handling for common model attributes to ensure they are wrapped
        if name in {"inner_model", "model", "wrapped_model", "wrapped", "model_inner", "model_core"}:
            return self._resolve_child_attribute(name)

        # Finally, delegate to the inner object
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Prevent torch.nn.Module from hijacking our private attributes
        if name in {"_inner", "_device", "_logger", "_sigma_fix_logged", "_apply_logged", "_child_proxies", "_delegated_attributes", "_module_proxies", "_module_hook_targets"}:
            self.__dict__[name] = value
        # If the value is a Module, let the parent class handle it to register it properly
        elif isinstance(value, torch.nn.Module):
            super().__setattr__(name, value)
        # Otherwise, set it on the inner object
        else:
            setattr(self._inner, name, value)

    def update_device(self, device: torch.device) -> None:
        self._device = device
        for proxy in self._child_proxies.values():
            proxy.update_device(device)
        # Also update device of registered sub-modules that are proxies
        for module in self.children():
            if isinstance(module, _DeviceAlignedModelProxy):
                module.update_device(device)
        for module_proxy in list(self._module_proxies):
            update_fn = getattr(module_proxy, "update_device", None)
            if callable(update_fn):
                update_fn(device)
            elif hasattr(module_proxy, "_h4_device_alignment"):
                setattr(module_proxy, "_h4_device_alignment_logged", False)
                setattr(module_proxy, "_h4_device_alignment_device", device)
        for module in list(self._module_hook_targets):
            if hasattr(module, "_h4_device_alignment"):
                setattr(module, "_h4_device_alignment_logged", False)
                setattr(module, "_h4_device_alignment_device", device)
            self._move_target_to_device(module, device, getattr(module, "_h4_device_alignment_label", module.__class__.__name__))
        self._move_target_to_device(self._inner, device, self._inner.__class__.__name__)
        diffusion_model = getattr(self._inner, "diffusion_model", None)
        self._move_target_to_device(diffusion_model, device, "diffusion_model")
        self._sigma_fix_logged = False
        self._apply_logged = False

    def _install_generic_module_proxy(self, host: Any, attribute: str, label: str) -> None:
        logger = self._logger
        if host is None:
            return
        module = getattr(host, attribute, None)
        if module is None or not isinstance(module, torch.nn.Module):
            logger.info(f"[DeviceCheck] {label} proxy skipped; attribute missing or not a module.")
            return
        if isinstance(module, _ModuleDeviceProxy):
            module.update_device(self._device)
            logger.info(f"[DeviceCheck] {label} proxy already active on {host.__class__.__name__}.")
            if module not in self._module_proxies:
                self._module_proxies.append(module)
            return
        try:
            proxy = _ModuleDeviceProxy(module, self._device, logger, label)
            setattr(host, attribute, proxy)
            self._module_proxies.append(proxy)
            logger.info(f"[DeviceCheck] {label} proxy installed on {host.__class__.__name__}.")
        except Exception as exc:
            logger.warn(f"[DeviceCheck] Failed installing {label} proxy on {host.__class__.__name__}: {exc}")
            
    def _move_target_to_device(self, target: Any, device: torch.device, label: str) -> None:
        if target is None:
            return
        if not isinstance(target, torch.nn.Module):
            return
        move_to = getattr(target, "to", None)
        if not callable(move_to):
            return
        try:
            move_to(device)
        except Exception as exc:
            self._logger.warn(f"[DeviceCheck] Unable to move {label} to {device}: {exc}")
            return
        self._realign_module_parameters(target, device, label)

    def _realign_module_parameters(self, target: torch.nn.Module, device: torch.device, label: str, depth: int = 0, visited: Optional[Set[int]] = None) -> None:
        if visited is None:
            visited = set()
        target_id = id(target)
        if target_id in visited:
            return
        visited.add(target_id)
        moved_any = False
        for name, parameter in list(target._parameters.items()):  # type: ignore[attr-defined]
            if parameter is None:
                continue
            if parameter.device != device:
                parameter.data = parameter.data.to(device=device)
                moved_any = True
        for name, buffer in list(target._buffers.items()):  # type: ignore[attr-defined]
            if buffer is None:
                continue
            if hasattr(buffer, "device") and buffer.device != device:
                target._buffers[name] = buffer.to(device=device)  # type: ignore[attr-defined]
                moved_any = True
        if moved_any and depth == 0:
            self._logger.info(f"[DeviceCheck] {label} parameters realigned to {device}")
        for child in target.children():
            if isinstance(child, torch.nn.Module):
                self._realign_module_parameters(child, device, label, depth + 1, visited)

    def _install_block_proxies(self, host: Any) -> None:
        if host is None:
            return
        block_specs = (
            ("input_blocks", "input_block"),
            ("middle_block", "middle_block"),
            ("output_blocks", "output_block"),
        )
        for attr, label_prefix in block_specs:
            blocks = getattr(host, attr, None)
            if blocks is None:
                continue
            if isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)):
                iterator = enumerate(blocks)
            else:
                iterator = ((None, blocks),)
            for index, block in iterator:
                label = f"{label_prefix}[{index}]" if index is not None else label_prefix
                self._ensure_module_hook(block, label)

    def _ensure_module_hook(self, module: Any, label: str) -> None:
        if module is None or not isinstance(module, torch.nn.Module):
            return
        if module in self._module_hook_targets:
            setattr(module, "_h4_device_alignment_logged", False)
            setattr(module, "_h4_device_alignment_device", self._device)
            return

        logger = self._logger
        owner_ref = weakref.ref(self)
        static_label = label
        self._move_target_to_device(module, self._device, static_label)

        def _pre_hook(mod: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            owner = owner_ref()
            if owner is None:
                return args, kwargs
            device = getattr(mod, "_h4_device_alignment_device", owner._device)
            mod.to(device)
            if not getattr(mod, "_h4_device_alignment_logged", False):
                logger.info(f"[DeviceCheck] {static_label} hook enforcing device alignment on {device}")
                setattr(mod, "_h4_device_alignment_logged", True)
            coerced_args = tuple(_DeviceAlignedModelProxy._payload_to_device(arg, device) for arg in args)
            coerced_kwargs = {key: _DeviceAlignedModelProxy._payload_to_device(val, device) for key, val in kwargs.items()}
            return coerced_args, coerced_kwargs

        handle = module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        module._h4_device_alignment = handle
        module._h4_device_alignment_logged = False
        module._h4_device_alignment_label = static_label
        module._h4_device_alignment_device = self._device
        self._module_hook_targets.append(module)

    @staticmethod
    def _eligible_for_wrapping(candidate: Any) -> bool:
        if candidate is None:
            return False
        if isinstance(candidate, _DeviceAlignedModelProxy):
            return True
        if isinstance(candidate, torch.nn.Module):
            return True
        if hasattr(candidate, "apply_model") and callable(getattr(candidate, "apply_model")):
            return True
        if hasattr(candidate, "diffusion_model"):
            return True
        if hasattr(candidate, "model") and hasattr(candidate, "inner_model"):
            return True
        return False

    def _obtain_child_proxy(self, candidate: Any, device: torch.device, logger: ToolkitLogger) -> Optional["_DeviceAlignedModelProxy"]:
        if candidate is None:
            return None
        if candidate is self._inner:
            return None
        if isinstance(candidate, _DeviceAlignedModelProxy):
            candidate.update_device(device)
            return candidate
        if not _DeviceAlignedModelProxy._eligible_for_wrapping(candidate):
            return None
        proxy: Optional[_DeviceAlignedModelProxy] = None
        try:
            proxy = _PROXY_REGISTRY.get(candidate)
        except TypeError:
            proxy = None
        if proxy is not None:
            proxy.update_device(device)
            return proxy
        proxy = _DeviceAlignedModelProxy(candidate, device, logger)
        try:
            _PROXY_REGISTRY[candidate] = proxy
        except TypeError:
            pass
        return proxy

    def _register_child_proxy(self, attr: str, proxy: "_DeviceAlignedModelProxy") -> None:
        self._child_proxies[attr] = proxy
        self._delegated_attributes.add(attr)
        try:
            # Use super's setattr to register it as a submodule
            super().__setattr__(attr, proxy)
        except Exception:
            pass

    def _wrap_known_children(self) -> None:
        device = self._device
        logger = self._logger
        candidate_attrs = (
            "inner_model",
            "model",
            "wrapped_model",
            "wrapped",
            "model_inner",
            "model_core",
        )
        for attr in candidate_attrs:
            if not hasattr(self._inner, attr):
                continue
            target = getattr(self._inner, attr)
            proxy = self._obtain_child_proxy(target, device, logger)
            if proxy is None:
                continue
            self._register_child_proxy(attr, proxy)
            break

    def _resolve_child_attribute(self, name: str) -> Any:
        if name in self._child_proxies:
            return self._child_proxies[name]
        if not hasattr(self._inner, name):
            raise AttributeError(name)
        device = self._device
        logger = self._logger
        target = getattr(self._inner, name)
        proxy = self._obtain_child_proxy(target, device, logger)
        if proxy is not None:
            self._register_child_proxy(name, proxy)
            return proxy
        return target

    def _coerce_sigma(self, sigma: Any, reference: Optional[torch.Tensor]) -> Any:
        device = self._device
        logger = self._logger
        fixed_flag = self._sigma_fix_logged
        if isinstance(sigma, torch.Tensor):
            if sigma.device != device:
                coerced = sigma.to(device=device)
                if not fixed_flag:
                    logger.warn(
                        f"Sampler sigma tensor arrived on {sigma.device}; moved to {device}"
                    )
                    self._sigma_fix_logged = True
                return coerced
            return sigma
        if isinstance(sigma, (float, int)):
            dtype = reference.dtype if isinstance(reference, torch.Tensor) else torch.float32
            return torch.as_tensor(sigma, dtype=dtype, device=device)
        if hasattr(sigma, "to"):
            try:
                coerced_any = sigma.to(device=device)
                if hasattr(coerced_any, "device") and getattr(coerced_any, "device") != device:
                    coerced_any = torch.as_tensor(coerced_any, device=device)
                if not fixed_flag:
                    logger.warn(
                        f"Sampler sigma object coerced onto {device} via .to() conversion"
                    )
                    self._sigma_fix_logged = True
                return coerced_any
            except Exception:
                pass
        return sigma

    def _align_sigma_args(
        self,
        args: List[Any],
        kwargs: Dict[str, Any],
        input_ref: Optional[torch.Tensor],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        logger = self._logger
        device = self._device
        sigma_value: Any = None
        sigma_index: Optional[int] = None
        if len(args) >= 2:
            sigma_index = 1
            sigma_value = args[1]
        elif "sigma" in kwargs:
            sigma_value = kwargs["sigma"]
        elif "timestep" in kwargs:
            sigma_value = kwargs["timestep"]
        apply_logged = self._apply_logged
        if not apply_logged:
            inbound_desc: str
            if isinstance(sigma_value, torch.Tensor):
                inbound_desc = f"tensor[{sigma_value.device}]"
            else:
                inbound_desc = type(sigma_value).__name__
            logger.info(f"[DeviceCheck] apply_model inbound sigma={inbound_desc}")
            self._apply_logged = True
        if sigma_value is not None:
            before_device: Optional[str] = None
            if isinstance(sigma_value, torch.Tensor):
                before_device = str(sigma_value.device)
            sigma_value = self._coerce_sigma(sigma_value, input_ref)
            if isinstance(sigma_value, torch.Tensor):
                after_device = str(sigma_value.device)
                if before_device is not None and before_device != after_device:
                    logger.info(
                        f"[DeviceCheck] sigma realigned {before_device} -> {after_device}"
                    )
            if sigma_index is not None:
                args[sigma_index] = sigma_value
            elif "sigma" in kwargs:
                kwargs["sigma"] = sigma_value
            else:
                kwargs["timestep"] = sigma_value
        else:
            logger.info(
                f"[DeviceCheck] sigma parameter missing; args={len(args)} kwargs={list(kwargs.keys())}"
            )
        if isinstance(sigma_value, torch.Tensor) and sigma_value.device != device:
            logger.warn(
                f"[DeviceCheck] sigma remains on {sigma_value.device} expected {device}"
            )
        return args, kwargs

    @staticmethod
    def _payload_to_device(
        value: Any,
        device: torch.device,
        *,
        logger: Optional[ToolkitLogger] = None,
        allow_dtype_adjustment: bool = False,
    ) -> Any:
        if value is None:
            return None
        if torch.is_tensor(value):
            return _DeviceAlignedModelProxy._tensor_to_device(
                value,
                device,
                logger=logger,
                allow_dtype_adjustment=allow_dtype_adjustment,
            )
        if isinstance(value, dict):
            return {
                key: _DeviceAlignedModelProxy._payload_to_device(
                    item,
                    device,
                    logger=logger,
                    allow_dtype_adjustment=allow_dtype_adjustment,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                _DeviceAlignedModelProxy._payload_to_device(
                    item,
                    device,
                    logger=logger,
                    allow_dtype_adjustment=allow_dtype_adjustment,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                _DeviceAlignedModelProxy._payload_to_device(
                    item,
                    device,
                    logger=logger,
                    allow_dtype_adjustment=allow_dtype_adjustment,
                )
                for item in value
            )
        if isinstance(value, set):
            return {
                _DeviceAlignedModelProxy._payload_to_device(
                    item,
                    device,
                    logger=logger,
                    allow_dtype_adjustment=allow_dtype_adjustment,
                )
                for item in value
            }
        move_to = getattr(value, "to", None)
        if callable(move_to):
            try:
                return move_to(device=device)
            except TypeError:
                try:
                    return move_to(device)
                except Exception:
                    pass
            except torch.cuda.OutOfMemoryError as exc:
                if device.type == "cuda" and allow_dtype_adjustment:
                    if logger is not None:
                        logger.warn(f"[DeviceCheck] Payload move to {device} hit OOM ({exc}); retrying with dtype adjustment")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        return move_to(device=device, dtype=torch.float16)
                    except (TypeError, torch.cuda.OutOfMemoryError):
                        pass
                raise
            except Exception:
                pass
        return value

    @staticmethod
    def _tensor_to_device(
        tensor: torch.Tensor,
        device: torch.device,
        *,
        logger: Optional[ToolkitLogger] = None,
        allow_dtype_adjustment: bool = False,
    ) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        try:
            return tensor.to(device=device, non_blocking=True)
        except torch.cuda.OutOfMemoryError as exc:
            if device.type != "cuda":
                raise
            if logger is not None:
                logger.warn(f"[DeviceCheck] Tensor move to {device} failed ({exc}); attempting recovery")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if allow_dtype_adjustment and tensor.is_floating_point() and tensor.dtype != torch.float16:
                try:
                    return tensor.to(device=device, dtype=torch.float16, non_blocking=True)
                except torch.cuda.OutOfMemoryError as retry_exc:
                    if logger is not None:
                        logger.error(
                            f"[DeviceCheck] Tensor fallback to float16 on {device} still failed ({retry_exc}); reraising"
                        )
                    raise
            raise

    @staticmethod
    def _infer_module_device(module: torch.nn.Module) -> Optional[torch.device]:
        try:
            parameter = next(module.parameters())
            return parameter.device
        except StopIteration:
            pass
        except Exception:
            return None
        try:
            buffer = next(module.buffers())
            return buffer.device
        except StopIteration:
            return None
        except Exception:
            return None

    @staticmethod
    def _infer_module_dtype(module: torch.nn.Module) -> Optional[torch.dtype]:
        try:
            parameter = next(module.parameters())
            return parameter.dtype
        except StopIteration:
            pass
        except Exception:
            return None
        try:
            buffer = next(module.buffers())
            return buffer.dtype
        except StopIteration:
            return None
        except Exception:
            return None

    def _align_argument_devices(
        self,
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        device = self._device
        for index, value in enumerate(args):
            args[index] = self._payload_to_device(value, device)
        for key, value in list(kwargs.items()):
            kwargs[key] = self._payload_to_device(value, device)
        return args, kwargs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._move_target_to_device(self._inner, self._device, self._inner.__class__.__name__)
        diffusion_model = getattr(self._inner, "diffusion_model", None)
        self._move_target_to_device(diffusion_model, self._device, "diffusion_model")
        args_list = list(args)
        reference_tensor: Optional[torch.Tensor] = None
        if args_list and isinstance(args_list[0], torch.Tensor):
            reference_tensor = args_list[0]
        elif isinstance(kwargs.get("x"), torch.Tensor):
            reference_tensor = kwargs["x"]
        args_aligned, kwargs_aligned = self._align_sigma_args(args_list, dict(kwargs), reference_tensor)
        args_aligned, kwargs_aligned = self._align_argument_devices(args_aligned, kwargs_aligned)
        return self._inner(*tuple(args_aligned), **kwargs_aligned)

    def apply_model(self, *args: Any, **kwargs: Any) -> Any:
        apply_fn = getattr(self._inner, "apply_model")
        self._move_target_to_device(self._inner, self._device, self._inner.__class__.__name__)
        diffusion_model = getattr(self._inner, "diffusion_model", None)
        self._move_target_to_device(diffusion_model, self._device, "diffusion_model")
        args_list = list(args)
        reference_tensor: Optional[torch.Tensor] = None
        if args_list and isinstance(args_list[0], torch.Tensor):
            reference_tensor = args_list[0]
        elif isinstance(kwargs.get("input_x"), torch.Tensor):
            reference_tensor = kwargs["input_x"]
        args_aligned, kwargs_aligned = self._align_sigma_args(args_list, dict(kwargs), reference_tensor)
        args_aligned, kwargs_aligned = self._align_argument_devices(args_aligned, kwargs_aligned)
        return apply_fn(*tuple(args_aligned), **kwargs_aligned)

    def release(self) -> None:
        for module in list(self._module_hook_targets):
            handle = getattr(module, "_h4_device_alignment", None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass
            for attr in (
                "_h4_device_alignment",
                "_h4_device_alignment_logged",
                "_h4_device_alignment_device",
                "_h4_device_alignment_label",
            ):
                if hasattr(module, attr):
                    try:
                        delattr(module, attr)
                    except AttributeError:
                        setattr(module, attr, None)
        for child in list(self._child_proxies.values()):
            if isinstance(child, _DeviceAlignedModelProxy):
                try:
                    child.release()
                except Exception:
                    pass
        self._module_hook_targets.clear()
        self._module_proxies.clear()
        self._child_proxies.clear()
        self._delegated_attributes.clear()
        try:
            existing = _PROXY_REGISTRY.get(self._inner)
            if existing is self:
                del _PROXY_REGISTRY[self._inner]
        except Exception:
            pass

    @classmethod
    def _ensure_calc_patch(cls, logger: ToolkitLogger) -> None:
        if cls._calc_patch_applied:
            return
        try:
            import comfy.samplers as comfy_samplers  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive only
            logger.warn(f"Unable to import comfy.samplers for timestep coercion: {exc}")
            return
        original = getattr(comfy_samplers, "calc_cond_batch", None)
        if original is None or not callable(original):
            logger.warn("calc_cond_batch missing; cannot install timestep coercion")
            return

        def patched_calc_cond_batch(model: Any, conds: Any, x_in: Any, timestep: Any, model_options: Any) -> Any:
            coerced = cls._coerce_external_timestep(timestep, x_in, getattr(model, "device", None))
            if coerced is not timestep:
                logger.info(
                    f"[DeviceCheck] calc_cond_batch timestep realigned to {getattr(coerced, 'device', 'unknown')}"
                )
            return original(model, conds, x_in, coerced, model_options)

        comfy_samplers.calc_cond_batch = patched_calc_cond_batch
        cls._calc_cond_batch_original = original
        cls._calc_patch_applied = True
        logger.info("[DeviceCheck] calc_cond_batch timestep coercion installed")

    @staticmethod
    def _coerce_external_timestep(value: Any, reference: Any, fallback_device: Optional[Union[str, torch.device]]) -> Any:
        target_device: Optional[torch.device] = None
        target_dtype: torch.dtype = torch.float32
        if isinstance(reference, torch.Tensor):
            target_device = reference.device
            target_dtype = reference.dtype
        elif isinstance(fallback_device, torch.device):
            target_device = fallback_device
        elif isinstance(fallback_device, str):
            try:
                target_device = torch.device(fallback_device)
            except Exception:
                target_device = None
        if target_device is None:
            if isinstance(value, torch.Tensor):
                target_device = value.device
                target_dtype = value.dtype
            else:
                return value
        if isinstance(value, torch.Tensor):
            if value.device != target_device or value.dtype != target_dtype:
                return value.to(device=target_device, dtype=target_dtype)
            return value
        if isinstance(value, (float, int)):
            return torch.as_tensor(value, dtype=target_dtype, device=target_device)
        if hasattr(value, "to"):
            try:
                coerced_any = value.to(device=target_device)
                if isinstance(coerced_any, torch.Tensor):
                    if coerced_any.device != target_device or coerced_any.dtype != target_dtype:
                        coerced_any = coerced_any.to(device=target_device, dtype=target_dtype)
                return coerced_any
            except Exception:
                return value
        return value


class _DeviceAlignedDiffusionProxy:
    # This class is now obsolete and will be removed.
    pass


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



AXIS_DRIVER_WIDGET_TOOLTIPS = {
    "config": "JSON configuration for the Axis Driver UI. Connect the outputs to nodes expecting axis payloads.",
}


DEBUG_WIDGET_TOOLTIPS = {
    "mode": "Monitor keeps signals internal. Passthrough relays inputs to outputs while logging.",
    "go_ultra": "Ask the router to GO PLUS ULTRA?! and unlock a diagnostics panel packed with snapshots, previews, JSON logs, and anomaly detectors.",
}

CONSOLE_COLOR_SCHEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "background": "#0b0c10",
        "panel": "#1f2833",
        "text": "#c5c6c7",
        "accent": "#66fcf1",
        "muted": "#45a29e",
    },
    "light": {
        "background": "#f7f7f7",
        "panel": "#ffffff",
        "text": "#1f2933",
        "accent": "#2563eb",
        "muted": "#4b5563",
    },
    "matrix": {
        "background": "#050805",
        "panel": "#0f2010",
        "text": "#9aff9a",
        "accent": "#39ff14",
        "muted": "#2f7044",
    },
    "cyberpunk": {
        "background": "#120022",
        "panel": "#1e0140",
        "text": "#f6f4ff",
        "accent": "#f72585",
        "muted": "#7209b7",
    },
}

CONSOLE_VERBOSITY_LEVELS: Dict[str, int] = {
    "minimal": 0,
    "normal": 2,
    "verbose": 8,
    "trace": -1,
}

EXECUTION_LOGGER_WIDGET_TOOLTIPS: Dict[str, str] = {
    "log_level": "Select the console severity used for emitted messages.",
    "prefix": "Human-readable tag prepended to every log entry.",
    "show_types": "Include Python type names for each connected payload.",
    "show_shapes": "Append tensor shapes and conditioning lengths when available.",
}

SEED_BROADCASTER_WIDGET_TOOLTIPS: Dict[str, str] = {
    "seed": "Base seed anchor. Used directly in fixed mode and as the starting point when you manually override other modes.",
    "mode": "Choose how the generator produces seeds: hold the anchor, auto-increment, or roll a fresh random number.",
    "increment_step": "Amount to add whenever increment mode advances. Ignored outside increment mode.",
    "auto_advance": "When enabled the node advances itself every execution. Disable to hold the current seed until you press Randomize or edit the value manually.",
    "random_digits": "How many digits to use when generating random seeds (1-12).",
}

VARIANATOR_PROFILE_RANGES: Dict[str, Tuple[float, float]] = {
    "minimal": (0.30, 0.40),
    "moderate": (0.40, 0.50),
    "major": (0.50, 0.55),
}

VARIANATOR_WIDGET_TOOLTIPS: Dict[str, str] = {
    "variation_count": "How many alternates to produce from the supplied latent. Each run clones the latent so upstream branches remain untouched.",
    "variation_profile": "Controls the denoise band. Minimal keeps things tight, major allows bolder remixing.",
    "seed_mode": "Fixed reuses the anchor seed, increment marches upward, random draws a deterministic sequence from the anchor.",
    "base_seed": "Anchor seed. Acts as the starting point for increment mode or the RNG seed for random mode.",
    "sampler_name": "Which sampler to use for every variation.",
    "scheduler_name": "Noise schedule for sampler execution.",
    "steps": "Sampler steps per variation. Keep modest; we are refining, not regenerating from scratch.",
    "cfg": "Classifier-free guidance shared across the batch.",
    "go_ultra": "Flip to capture JSON logs and optional preview artefacts for every variation.",
    "ultra_json_log": "Persist a JSON dossier of the run when GO PLUS ULTRA?! is on.",
    "ultra_cache_artifacts": "When GO PLUS ULTRA?! is on, save preview PNGs alongside the JSON log.",
    "prompt_jitter_enabled": "Append weighted prompt fragments per variation. Requires a positive prompt and CLIP handle.",
    "prompt_jitter_strength": "Maximum deviation when no explicit token range is provided (1.0 ± strength).",
    "prompt_jitter_tokens": "Optional token list for jittering. One per line. Format: token|min|max (weights).",
    "style_mix_enabled": "Blend a secondary conditioning or prompt into the main prompt using the mix ratio.",
    "style_mix_ratio": "0 keeps the base prompt. 1.0 swaps fully to the style-mix source.",
    "base_positive_prompt": "Fallback positive prompt when no conditioning input is connected (also used for prompt jitter).",
    "base_negative_prompt": "Fallback negative prompt when no conditioning input is connected.",
    "style_mix_prompt": "Optional prompt string to blend when style mixing is enabled and no conditioning input is provided.",
    "style_mix_negative_prompt": "Optional negative prompt counterpart for style mixing.",
}

VARIANATOR_MAX_VARIATIONS = 16
VARIANATOR_SEED_LIMIT = 2**63 - 1

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

    def debug(self, message: str) -> None:
        self._emit("DEBUG", Style.NORMAL, message)

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


def _axis_driver_default_state() -> Dict[str, Any]:
    return copy.deepcopy(AXIS_DRIVER_DEFAULT_STATE)


def _axis_driver_normalise_style(raw: Any) -> Dict[str, Any]:
    style = copy.deepcopy(AXIS_DRIVER_DEFAULT_STYLE)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in style:
                style[key] = value
    return style


def _axis_driver_normalise_item(preset: str, raw: Any) -> Dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    label = str(payload.get("label") or payload.get("source_label") or "").strip()
    value = payload.get("value")
    if value is None and "raw_value" in payload:
        value = payload.get("raw_value")
    overrides = payload.get("overrides")
    overrides = overrides if isinstance(overrides, dict) else {}
    strength_raw = payload.get("strength")
    strength: Optional[float]
    if preset == "lora":
        try:
            strength = float(strength_raw)
        except (TypeError, ValueError):
            strength = 0.75
    else:
        strength = None
    return {
        "label": label,
        "value": value,
        "strength": strength,
        "overrides": copy.deepcopy(overrides),
    }


def _axis_driver_normalise_axis(slot: str, raw: Any) -> Dict[str, Any]:
    preset_default = AXIS_DRIVER_PRESET_DEFAULTS.get(slot, "none")
    payload = raw if isinstance(raw, dict) else {}
    preset_raw = str(payload.get("preset") or preset_default).lower()
    preset = preset_raw if preset_raw in AXIS_DRIVER_SUPPORTED_PRESETS else preset_default
    items_raw = payload.get("items") if isinstance(payload, dict) else None
    items: List[Dict[str, Any]] = []
    if isinstance(items_raw, list):
        for entry in items_raw[:AXIS_DRIVER_MAX_ITEMS]:
            items.append(_axis_driver_normalise_item(preset, entry))
    return {"slot": slot, "preset": preset, "items": items}


def _axis_driver_normalise_state(raw: Any) -> Dict[str, Any]:
    state = _axis_driver_default_state()
    if not isinstance(raw, dict):
        return state

    state["style"] = _axis_driver_normalise_style(raw.get("style"))
    slot_lookup = {axis["slot"]: axis for axis in state["axes"]}
    axes_raw = raw.get("axes")
    if isinstance(axes_raw, list):
        for entry in axes_raw:
            slot = str(entry.get("slot") if isinstance(entry, dict) else "").upper()
            if slot in slot_lookup:
                slot_lookup[slot].update(_axis_driver_normalise_axis(slot, entry))
    # Ensure order and fallback defaults are preserved
    ordered_axes: List[Dict[str, Any]] = []
    for slot in AXIS_DRIVER_SLOT_ORDER:
        base_axis = slot_lookup.get(slot, _axis_driver_normalise_axis(slot, {}))
        if base_axis.get("preset") not in AXIS_DRIVER_SUPPORTED_PRESETS:
            base_axis["preset"] = AXIS_DRIVER_PRESET_DEFAULTS.get(slot, "none")
        if not isinstance(base_axis.get("items"), list):
            base_axis["items"] = []
        base_axis["items"] = base_axis["items"][:AXIS_DRIVER_MAX_ITEMS]
        ordered_axes.append(base_axis)
    state["axes"] = ordered_axes
    return state


def _axis_driver_slot_payload(state: Dict[str, Any], slot: str) -> Dict[str, Any]:
    axes = state.get("axes") if isinstance(state, dict) else None
    axis: Optional[Dict[str, Any]] = None
    if isinstance(axes, list):
        for entry in axes:
            if isinstance(entry, dict) and entry.get("slot") == slot:
                axis = entry
                break
    if axis is None:
        axis = _axis_driver_normalise_axis(slot, {})
    payload_items: List[Dict[str, Any]] = []
    for item in axis.get("items", []):
        if not isinstance(item, dict):
            continue
        payload_items.append(
            {
                "label": str(item.get("label") or ""),
                "value": item.get("value"),
                "strength": item.get("strength"),
                "overrides": copy.deepcopy(item.get("overrides") if isinstance(item.get("overrides"), dict) else {}),
            }
        )
    return {
        "slot": slot,
        "preset": axis.get("preset", "none"),
        "items": payload_items,
        "style": copy.deepcopy(state.get("style", {})),
    }


def _axis_driver_legacy_summary(state: Dict[str, Any]) -> str:
    lines: List[str] = []
    axes = state.get("axes") if isinstance(state, dict) else []
    for axis in axes:
        if not isinstance(axis, dict):
            continue
        slot = axis.get("slot", "?")
        preset = axis.get("preset", "none")
        header = f"Axis {slot} ({preset})"
        items = axis.get("items") if isinstance(axis.get("items"), list) else []
        if not items or preset == "none":
            lines.append(f"{header}: <disabled>")
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("value") or ""
            strength = item.get("strength")
            if preset == "lora" and strength is not None:
                lines.append(f"{header}: {label} @ {strength}")
            else:
                lines.append(f"{header}: {label}")
    style = state.get("style") if isinstance(state, dict) else None
    if isinstance(style, dict):
        layout = style.get("label_layout", "overlay")
        lines.append(f"Style: layout={layout}, font={style.get('font_family', 'default')} size={style.get('font_size', 22)}")
    return "\n".join(lines)


def _axis_driver_parse_config(config_text: str) -> Dict[str, Any]:
    if not isinstance(config_text, str) or not config_text.strip():
        return _axis_driver_default_state()
    try:
        raw = json.loads(config_text)
    except Exception:
        GLOBAL_LOGGER.warn("Axis Driver config parse failed; reverting to defaults")
        return _axis_driver_default_state()
    return _axis_driver_normalise_state(raw)


def _axis_driver_payload_to_descriptors(
    payload: Optional[Dict[str, Any]],
) -> Tuple[List[AxisDescriptor], Optional[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return [], None

    preset = str(payload.get("preset") or "none").lower()
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    style = payload.get("style") if isinstance(payload.get("style"), dict) else None

    if preset not in AXIS_DRIVER_SUPPORTED_PRESETS or not items:
        return [], style

    descriptors: List[AxisDescriptor] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("value") or "").strip()
        overrides_raw = item.get("overrides") if isinstance(item.get("overrides"), dict) else {}
        overrides = copy.deepcopy(overrides_raw)
        value = item.get("value")

        if preset == "prompt":
            text_value = "" if value is None else str(value)
            descriptors.append(
                AxisDescriptor(
                    source_label=label or text_value or "prompt",
                    prompt_suffix=text_value,
                    overrides=overrides,
                )
            )
        elif preset == "checkpoint":
            candidate = str(value or "")
            resolved = resolve_checkpoint_name(candidate) or candidate
            if not resolved:
                continue
            descriptors.append(
                AxisDescriptor(
                    source_label=label or resolved,
                    checkpoint=resolved,
                    overrides=overrides,
                )
            )
        elif preset == "lora":
            candidate = str(value or "")
            resolved = resolve_lora_name(candidate) or candidate
            try:
                strength = float(item.get("strength", 0.75))
            except (TypeError, ValueError):
                strength = 0.75
            descriptors.append(
                AxisDescriptor(
                    source_label=label or resolved,
                    loras=[(resolved, strength)],
                    overrides=overrides,
                )
            )
        elif preset == "sampler":
            if not value:
                continue
            sampler_name = str(value)
            lookup = SAMPLER_LOOKUP.get(sampler_name.lower())
            if lookup is None:
                GLOBAL_LOGGER.warn(f"Axis Driver sampler '{sampler_name}' not recognised; skipping entry")
                continue
            overrides.setdefault("sampler", lookup)
            descriptors.append(
                AxisDescriptor(
                    source_label=label or sampler_name,
                    overrides=overrides,
                )
            )
        elif preset == "scheduler":
            if not value:
                continue
            scheduler_name = str(value)
            lookup = SCHEDULER_LOOKUP.get(scheduler_name.lower())
            if lookup is None:
                GLOBAL_LOGGER.warn(f"Axis Driver scheduler '{scheduler_name}' not recognised; skipping entry")
                continue
            overrides.setdefault("scheduler", lookup)
            descriptors.append(
                AxisDescriptor(
                    source_label=label or scheduler_name,
                    overrides=overrides,
                )
            )
        elif preset == "steps":
            try:
                steps_value = max(1, int(value))
            except (TypeError, ValueError):
                GLOBAL_LOGGER.warn(f"Axis Driver steps value '{value}' invalid; skipping entry")
                continue
            overrides.setdefault("steps", steps_value)
            descriptors.append(
                AxisDescriptor(
                    source_label=label or str(steps_value),
                    overrides=overrides,
                )
            )
        elif preset == "cfg":
            try:
                cfg_value = float(value)
            except (TypeError, ValueError):
                GLOBAL_LOGGER.warn(f"Axis Driver CFG value '{value}' invalid; skipping entry")
                continue
            overrides.setdefault("cfg", round(cfg_value, 3))
            descriptors.append(
                AxisDescriptor(
                    source_label=label or str(cfg_value),
                    overrides=overrides,
                )
            )
        elif preset == "denoise":
            try:
                denoise_value = float(value)
            except (TypeError, ValueError):
                GLOBAL_LOGGER.warn(f"Axis Driver denoise value '{value}' invalid; skipping entry")
                continue
            overrides.setdefault("denoise", max(0.0, min(1.0, round(denoise_value, 3))))
            descriptors.append(
                AxisDescriptor(
                    source_label=label or str(denoise_value),
                    overrides=overrides,
                )
            )
        elif preset == "seed":
            try:
                seed_value = int(value)
            except (TypeError, ValueError):
                GLOBAL_LOGGER.warn(f"Axis Driver seed value '{value}' invalid; skipping entry")
                continue
            overrides.setdefault("seed", seed_value)
            descriptors.append(
                AxisDescriptor(
                    source_label=label or str(seed_value),
                    overrides=overrides,
                )
            )

    return descriptors, style


class h4_AxisDriver:
    """Companion node that serialises structured axis presets for The Engine."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802
        default_config = json.dumps(AXIS_DRIVER_DEFAULT_STATE, indent=2)
        return {
            "required": {
                "config": (
                    "STRING",
                    {
                        "default": default_config,
                        "multiline": True,
                        "tooltip": AXIS_DRIVER_WIDGET_TOOLTIPS.get("config"),
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("axis_x", "axis_y", "axis_z", "legacy_summary")
    FUNCTION = "emit"
    CATEGORY = "h4 Toolkit/Generation"

    def __init__(self) -> None:
        self.logger = ToolkitLogger("h4_AxisDriver")

    def emit(self, config: str) -> Tuple[str, str, str, str]:
        state = _axis_driver_parse_config(config)
        normalised = json.dumps(state, indent=2)
        if normalised != config:
            self.logger.trace("Axis Driver config normalised for downstream consumers")
        slot_payloads = {
            slot: json.dumps(_axis_driver_slot_payload(state, slot), indent=2)
            for slot in AXIS_DRIVER_SLOT_ORDER
        }
        summary = _axis_driver_legacy_summary(state)
        return (
            slot_payloads.get("X", ""),
            slot_payloads.get("Y", ""),
            slot_payloads.get("Z", ""),
            summary,
        )


class h4_SeedBroadcaster:
    """Utility node that generates reproducible seeds with lightweight sequencing."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 123456789,
                        "min": 0,
                        "max": 2**63 - 1,
                        "tooltip": SEED_BROADCASTER_WIDGET_TOOLTIPS.get("seed"),
                    },
                ),
                "mode": (
                    ["fixed", "increment", "random"],
                    {
                        "default": "fixed",
                        "tooltip": SEED_BROADCASTER_WIDGET_TOOLTIPS.get("mode"),
                    },
                ),
                "increment_step": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 2**31 - 1,
                        "tooltip": SEED_BROADCASTER_WIDGET_TOOLTIPS.get("increment_step"),
                    },
                ),
                "auto_advance": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": SEED_BROADCASTER_WIDGET_TOOLTIPS.get("auto_advance"),
                    },
                ),
                "random_digits": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 12,
                        "tooltip": SEED_BROADCASTER_WIDGET_TOOLTIPS.get("random_digits"),
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate"
    CATEGORY = "h4 Toolkit/Utility"

    def __init__(self) -> None:
        self.logger = ToolkitLogger("h4_SeedBroadcaster")
        self._current_seed: Optional[int] = None
        self._anchor_seed: Optional[int] = None
        self._last_mode: Optional[str] = None
        self._last_random_digits: int = 10
        self._last_emitted_seed: Optional[int] = None

    @staticmethod
    def _coerce_seed(value: Optional[Any]) -> int:
        try:
            integer = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            integer = 0
        limit = 2**63 - 1
        return max(0, min(integer, limit))

    @staticmethod
    def _normalise_mode(mode: Optional[str]) -> str:
        if not mode:
            return "fixed"
        candidate = str(mode).strip().lower()
        return candidate if candidate in {"fixed", "increment", "random"} else "fixed"

    @staticmethod
    def _normalise_step(value: Optional[Any]) -> int:
        try:
            step = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            step = 1
        return max(1, min(step, 2**31 - 1))

    @staticmethod
    def _normalise_digits(value: Optional[Any]) -> int:
        try:
            digits = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            digits = 10
        return max(1, min(digits, 12))

    def _generate_random_seed(self, digits: int) -> int:
        digits = self._normalise_digits(digits)
        upper_bound = min(10**digits - 1, 2**63 - 1)
        lower_bound = 0 if digits == 1 else 10 ** (digits - 1)
        span = upper_bound - lower_bound + 1
        return lower_bound + secrets.randbelow(span)

    def generate(
        self,
        seed: int,
        mode: str,
        increment_step: int,
        auto_advance: bool,
        random_digits: int,
    ) -> Tuple[int]:
        base_seed = self._coerce_seed(seed)
        mode_normalised = self._normalise_mode(mode)
        step = self._normalise_step(increment_step)
        digits = self._normalise_digits(random_digits)
        auto_flag = bool(auto_advance)

        changed_mode = mode_normalised != self._last_mode
        base_changed = self._anchor_seed is None or base_seed != self._anchor_seed
        digits_changed = mode_normalised == "random" and digits != self._last_random_digits

        need_reset = (
            self._current_seed is None
            or changed_mode
            or digits_changed
            or (base_changed and mode_normalised != "random")
        )
        manual_override = (
            mode_normalised == "random"
            and base_changed
            and not changed_mode
            and not digits_changed
        )

        if need_reset:
            if mode_normalised == "random" and not base_changed:
                self._current_seed = self._generate_random_seed(digits)
            else:
                self._current_seed = base_seed
                if mode_normalised == "random" and base_changed:
                    self.logger.info(
                        f"[Seed Generator] Manual override detected; adopting seed {self._current_seed}"
                    )
        elif manual_override:
            self._current_seed = base_seed

        result = int(self._current_seed or 0)

        if auto_flag:
            if mode_normalised == "increment":
                next_seed = self._coerce_seed(result + step)
                self._current_seed = next_seed
            elif mode_normalised == "random":
                self._current_seed = self._generate_random_seed(digits)
            else:
                self._current_seed = base_seed
        else:
            self._current_seed = result

        self._anchor_seed = base_seed
        self._last_mode = mode_normalised
        self._last_random_digits = digits

        if self._last_emitted_seed != result:
            self.logger.info(
                f"[Seed Generator] mode={mode_normalised} auto={'on' if auto_flag else 'off'} emitted seed {result}"
            )
        else:
            self.logger.trace(
                f"[Seed Generator] mode={mode_normalised} reusing seed {result}"
            )
        self._last_emitted_seed = result
        return (result,)


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


def _normalise_model_family_tag(tag: Optional[str]) -> Optional[str]:
    if tag is None:
        return None
    lowered = str(tag).strip().lower().replace(" ", "")
    if not lowered:
        return None
    normalisers = {
        "stable-diffusionxl": "sdxl",
        "stablediffusionxl": "sdxl",
        "sdxl": "sdxl",
        "xl": "sdxl",
        "stable-diffusion1.5": "sd15",
        "stablediffusion1.5": "sd15",
        "sd1.5": "sd15",
        "sd15": "sd15",
        "1.5": "sd15",
        "v1-5": "sd15",
    }
    for needle, family in normalisers.items():
        if needle in lowered:
            return family
    segments = [segment for segment in re.split(r"[^a-z0-9]+", lowered) if segment]
    if any(segment in {"sdxl", "xl"} for segment in segments):
        return "sdxl"
    if any(segment in {"sd15", "15"} for segment in segments):
        return "sd15"
    return None


def _infer_family_from_name(path_like: Optional[str]) -> Optional[str]:
    if not path_like:
        return None
    candidate = str(path_like).replace("\\", "/").lower()
    return _normalise_model_family_tag(candidate)


def _family_from_file(path: Optional[str], logger: Optional[Any] = None, asset_label: str = "asset") -> Optional[str]:
    if not path:
        return None
    metadata_family: Optional[str] = None
    try:
        if str(path).lower().endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as handle:
                meta = handle.metadata() or {}
            for key in ("ss_base_model_version", "base_model", "format"):
                family = _normalise_model_family_tag(meta.get(key))
                if family:
                    metadata_family = family
                    break
    except Exception as exc:  # pragma: no cover - depends on external files
        if logger is not None:
            logger.warn(f"[LoRA Inspector] Could not read metadata for {asset_label} '{path}': {exc}")
    return metadata_family or _infer_family_from_name(path)


def _infer_family_from_model_object(model_obj: Any) -> Optional[str]:
    candidates = [model_obj, getattr(model_obj, "model", None), getattr(model_obj, "wrapped_model", None)]
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        for attr_name in ("model_type", "base_model", "architecture", "name", "type"):
            value = getattr(candidate, attr_name, None)
            family = _normalise_model_family_tag(value)
            if family:
                return family
        config = getattr(candidate, "model_config", None)
        config_items: Dict[str, Any] = {}
        if isinstance(config, dict):
            config_items = config
        elif hasattr(config, "__dict__"):
            config_items = dict(vars(config))
        elif hasattr(config, "_asdict") and callable(getattr(config, "_asdict")):
            try:
                config_items = dict(config._asdict())  # type: ignore[misc]
            except Exception:  # pragma: no cover - fallback best effort
                config_items = {}
        if config_items:
            for key in ("model_type", "base_model", "architecture", "name", "type"):
                family = _normalise_model_family_tag(config_items.get(key))
                if family:
                    return family
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


def _parse_hex_colour(value: Optional[str], default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if not isinstance(value, str):
        return default
    token = value.strip().lstrip("#")
    if len(token) == 3:
        token = "".join(ch * 2 for ch in token)
    if len(token) != 6:
        return default
    try:
        channels = tuple(int(token[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return default
    return channels  # type: ignore[return-value]


def _resolve_background_colour(value: Optional[str], alpha_default: int) -> Optional[Tuple[int, int, int, int]]:
    alpha = max(0, min(255, alpha_default))
    presets = {
        "black60": (0, 0, 0, int(round(255 * 0.6))),
        "black80": (0, 0, 0, int(round(255 * 0.8))),
        "white40": (255, 255, 255, int(round(255 * 0.4))),
    }
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"none", "transparent"}:
            return None
        preset = presets.get(token)
        if preset is not None:
            return preset
        if token.startswith("#") or all(ch in "0123456789abcdef" for ch in token if ch != "#"):
            red, green, blue = _parse_hex_colour(token, (0, 0, 0))
            return (red, green, blue, alpha)
    return (0, 0, 0, alpha)


def _load_preferred_font(family: Optional[str], size: int) -> Any:
    if ImageFont is None:  # pragma: no cover - Pillow optional
        raise RuntimeError("Pillow ImageFont unavailable")
    target_size = size if size > 0 else 18
    candidates: List[str] = []
    if isinstance(family, str) and family.strip():
        candidate = family.strip()
        candidates.append(candidate)
        lowered = candidate.lower()
        if not lowered.endswith((".ttf", ".otf", ".ttc")):
            candidates.append(f"{candidate}.ttf")
            candidates.append(f"{candidate}.otf")
        if os.path.isfile(candidate):
            candidates.insert(0, candidate)
    candidates.extend(["DejaVuSans.ttf", "arial.ttf"])
    for entry in candidates:
        try:
            return ImageFont.truetype(entry, target_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _axis_descriptor_label(descriptor: Optional[AxisDescriptor], fallback: str) -> str:
    if descriptor is None:
        return fallback
    for candidate in (descriptor.source_label, descriptor.describe()):
        text = str(candidate or "").strip()
        if text and text not in {"identity", "∅"}:
            return text
    return fallback


def _normalise_label_text(raw: str, fallback: str) -> str:
    text = str(raw or "").strip()
    if not text or text.lower() == "identity" or text == "∅":
        text = fallback
    text = text.replace(" | ", "\n")
    return text if text else fallback


def _wrap_label_lines(text: str, max_chars: int, limit: int = 4) -> List[str]:
    width = max(1, max_chars)
    lines: List[str] = []
    for block in text.split("\n"):
        candidate = block.strip()
        if not candidate:
            continue
        wrapped = textwrap.wrap(candidate, width=width) or [candidate]
        lines.extend(wrapped)
        if len(lines) >= limit:
            break
    return lines[:limit] if lines else [text.strip() or "base"]


def _measure_text(
    draw: Any,
    text: str,
    font: Any,
    *,
    spacing: int,
    align: str,
) -> Tuple[int, int]:
    if hasattr(draw, "multiline_textbbox"):
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=align)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    width, height = draw.multiline_textsize(text, font=font, spacing=spacing)
    return width, height


def annotate_grid_image(
    grid_tensor: torch.Tensor,
    plans: List[GenerationPlan],
    rows: int,
    cols: int,
    logger: ToolkitLogger,
    style: Optional[Dict[str, Any]] = None,
    x_axis: Optional[List[AxisDescriptor]] = None,
    y_axis: Optional[List[AxisDescriptor]] = None,
) -> torch.Tensor:
    if Image is None or ImageDraw is None or ImageFont is None:
        logger.trace("Skipping grid annotation; Pillow not available")
        return grid_tensor
    if not plans:
        return grid_tensor

    style_data: Dict[str, Any] = dict(style or {})
    layout = str(style_data.get("label_layout", "overlay") or "overlay").lower()
    if layout not in {"overlay", "border", "none"}:
        layout = "overlay"
    if layout == "none":
        return grid_tensor

    font_size = int(style_data.get("font_size") or 18)
    alignment = str(style_data.get("alignment", "left") or "left").lower()
    if alignment not in {"left", "center", "right"}:
        alignment = "left"
    label_position = str(style_data.get("label_position", "top_left") or "top_left").lower()
    if label_position not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
        label_position = "top_left"
    background_rgba = _resolve_background_colour(style_data.get("background"), 200 if layout == "border" else 160)
    text_colour = _parse_hex_colour(style_data.get("font_colour"), (255, 255, 255))
    spacing = max(2, font_size // 6)

    try:
        font = _load_preferred_font(style_data.get("font_family"), font_size)
    except Exception as exc:  # pragma: no cover - font lookup failure
        logger.debug(f"Font preference unavailable ({exc}); falling back to default")
        font = ImageFont.load_default()

    try:
        device = grid_tensor.device
        grid_cpu = grid_tensor.detach().to("cpu")
        base = grid_cpu[0].float().clamp(0.0, 1.0)
        image_array = (base.numpy() * 255.0).astype(np.uint8)
        image = Image.fromarray(image_array)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warn(f"Unable to prepare grid for annotation: {exc}")
        return grid_tensor

    if layout == "border":
        annotated_image = _render_border_layout(
            image,
            plans,
            rows,
            cols,
            font,
            text_colour,
            background_rgba,
            alignment,
            spacing,
            font_size,
            style_data,
            x_axis or [],
            y_axis or [],
            logger,
        )
    else:
        annotated_image = _render_overlay_layout(
            image,
            plans,
            rows,
            cols,
            font,
            text_colour,
            background_rgba,
            alignment,
            label_position,
            spacing,
            font_size,
        )

    annotated_rgb = annotated_image.convert("RGB") if annotated_image.mode != "RGB" else annotated_image
    annotated_array = np.asarray(annotated_rgb, dtype=np.float32) / 255.0
    annotated_tensor = torch.from_numpy(annotated_array).to(grid_tensor.dtype).unsqueeze(0)
    return annotated_tensor.to(device)


def _render_overlay_layout(
    image: Any,
    plans: List[GenerationPlan],
    rows: int,
    cols: int,
    font: Any,
    text_colour: Tuple[int, int, int],
    background_rgba: Optional[Tuple[int, int, int, int]],
    alignment: str,
    label_position: str,
    spacing: int,
    font_size: int,
) -> Any:
    draw = ImageDraw.Draw(image, "RGBA")
    cols_safe = max(1, cols)
    rows_safe = max(1, rows)
    cell_width = image.width // cols_safe
    cell_height = image.height // rows_safe
    margin = max(6, font_size // 3)
    approx_char_width = max(font_size * 0.55, 6.0)
    available_text_width = max(16, cell_width - (margin * 2))
    max_chars = max(4, int(available_text_width / approx_char_width))

    for index, plan in enumerate(plans):
        row_index = index // cols_safe
        col_index = index % cols_safe
        cell_origin_x = col_index * cell_width
        cell_origin_y = row_index * cell_height
        fallback_label = f"Run {index + 1}"
        label_text = _normalise_label_text(plan.label, fallback_label)
        wrapped_lines = _wrap_label_lines(label_text, max_chars)
        text_block = "\n".join(wrapped_lines)
        text_width, text_height = _measure_text(draw, text_block, font, spacing=spacing, align=alignment)
        rect_padding = 6
        rect_width = text_width + rect_padding * 2
        rect_height = text_height + rect_padding * 2
        rect_width = min(rect_width, cell_width - margin)
        rect_height = min(rect_height, cell_height - margin)

        if label_position in {"top_left", "top_right"}:
            rect_y = cell_origin_y + margin
        else:
            rect_y = cell_origin_y + cell_height - margin - rect_height

        if label_position in {"top_left", "bottom_left"}:
            rect_x = cell_origin_x + margin
        else:
            rect_x = cell_origin_x + cell_width - margin - rect_width

        rect_x = max(cell_origin_x + 2, min(rect_x, cell_origin_x + cell_width - rect_width - 2))
        rect_y = max(cell_origin_y + 2, min(rect_y, cell_origin_y + cell_height - rect_height - 2))

        if background_rgba is not None and background_rgba[3] > 0:
            draw.rectangle(
                (
                    rect_x,
                    rect_y,
                    rect_x + rect_width,
                    rect_y + rect_height,
                ),
                fill=background_rgba,
            )

        if alignment == "center":
            text_x = rect_x + (rect_width - text_width) / 2
        elif alignment == "right":
            text_x = rect_x + rect_width - rect_padding - text_width
        else:
            text_x = rect_x + rect_padding
        text_y = rect_y + (rect_height - text_height) / 2
        draw.multiline_text(
            (text_x, text_y),
            text_block,
            font=font,
            fill=text_colour,
            spacing=spacing,
            align=alignment,
        )
    return image


def _render_border_layout(
    image: Any,
    plans: List[GenerationPlan],
    rows: int,
    cols: int,
    font: Any,
    text_colour: Tuple[int, int, int],
    background_rgba: Optional[Tuple[int, int, int, int]],
    alignment: str,
    spacing: int,
    font_size: int,
    style: Dict[str, Any],
    x_axis: List[AxisDescriptor],
    y_axis: List[AxisDescriptor],
    logger: ToolkitLogger,
) -> Any:
    draw_probe = ImageDraw.Draw(image, "RGBA")
    cols_safe = max(1, cols)
    rows_safe = max(1, rows)
    cell_width = image.width // cols_safe
    cell_height = image.height // rows_safe
    header_padding = max(6, font_size // 3)

    col_labels: List[str] = []
    if x_axis:
        for index in range(min(len(x_axis), cols_safe)):
            col_labels.append(_axis_descriptor_label(x_axis[index], f"Column {index + 1}"))
    if len(col_labels) < cols_safe:
        for index in range(len(col_labels), cols_safe):
            fallback = plans[index].label if index < len(plans) else ""
            col_labels.append(_axis_descriptor_label(None, _normalise_label_text(fallback, f"Column {index + 1}")))

    row_labels: List[str] = []
    if y_axis:
        for index in range(min(len(y_axis), rows_safe)):
            row_labels.append(_axis_descriptor_label(y_axis[index], f"Row {index + 1}"))
    if len(row_labels) < rows_safe:
        for index in range(len(row_labels), rows_safe):
            plan_index = index * cols_safe
            fallback = plans[plan_index].label if plan_index < len(plans) else ""
            row_labels.append(_axis_descriptor_label(None, _normalise_label_text(fallback, f"Row {index + 1}")))

    approx_char_width = max(font_size * 0.55, 6.0)
    available_col_width = max(24, cell_width - header_padding * 2)
    max_chars_col = max(4, int(available_col_width / approx_char_width))

    col_blocks: List[Tuple[str, int, int]] = []
    max_col_height = 0
    for index, label in enumerate(col_labels):
        text = _normalise_label_text(label, f"Column {index + 1}")
        wrapped = _wrap_label_lines(text, max_chars_col)
        text_block = "\n".join(wrapped)
        width, height = _measure_text(draw_probe, text_block, font, spacing=spacing, align=alignment)
        col_blocks.append((text_block, width, height))
        max_col_height = max(max_col_height, height)

    row_blocks: List[Tuple[str, int, int]] = []
    max_row_width = 0
    for index, label in enumerate(row_labels):
        text_block = _normalise_label_text(label, f"Row {index + 1}")
        width, height = _measure_text(draw_probe, text_block, font, spacing=spacing, align="left")
        row_blocks.append((text_block, width, height))
        max_row_width = max(max_row_width, width)

    axis_headers_enabled = bool(style.get("show_axis_headers", True))
    axis_x_text_block = ""
    axis_y_text_block = ""
    axis_x_metrics = (0, 0)
    axis_y_metrics = (0, 0)
    if axis_headers_enabled:
        candidate_x = str(style.get("custom_label_x") or "").strip()
        if candidate_x:
            axis_x_text_block = _normalise_label_text(candidate_x, candidate_x)
            axis_x_metrics = _measure_text(draw_probe, axis_x_text_block, font, spacing=spacing, align="center")
        candidate_y = str(style.get("custom_label_y") or "").strip()
        if candidate_y:
            axis_y_text_block = _normalise_label_text(candidate_y, candidate_y)
            axis_y_metrics = _measure_text(draw_probe, axis_y_text_block, font, spacing=spacing, align="left")

    column_panel_height = (max_col_height + header_padding * 2) if col_blocks else 0
    row_panel_width = (max_row_width + header_padding * 2) if row_blocks else 0
    axis_x_panel_height = (axis_x_metrics[1] + header_padding * 2) if axis_x_text_block else 0
    axis_y_panel_width = (axis_y_metrics[0] + header_padding * 2) if axis_y_text_block else 0

    top_panel_height = int(axis_x_panel_height + column_panel_height)
    left_panel_width = int(axis_y_panel_width + row_panel_width)

    new_width = image.width + left_panel_width
    new_height = image.height + top_panel_height
    canvas = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    canvas.paste(image, (left_panel_width, top_panel_height))
    draw_canvas = ImageDraw.Draw(canvas, "RGBA")

    if background_rgba is not None and background_rgba[3] > 0:
        draw_canvas.rectangle((0, 0, new_width, top_panel_height), fill=background_rgba)
        draw_canvas.rectangle((0, top_panel_height, left_panel_width, new_height), fill=background_rgba)

    text_fill = text_colour

    if axis_x_text_block:
        axis_x_x = left_panel_width + (image.width - axis_x_metrics[0]) / 2
        axis_x_y = max(header_padding // 2, header_padding)
        draw_canvas.multiline_text(
            (axis_x_x, axis_x_y),
            axis_x_text_block,
            font=font,
            fill=text_fill,
            spacing=spacing,
            align="center",
        )

    if axis_y_text_block:
        axis_y_x = max(header_padding // 2, header_padding)
        axis_y_y = top_panel_height + (image.height - axis_y_metrics[1]) / 2
        draw_canvas.multiline_text(
            (axis_y_x, axis_y_y),
            axis_y_text_block,
            font=font,
            fill=text_fill,
            spacing=spacing,
            align="left",
        )

    column_offset_y = axis_x_panel_height if axis_x_panel_height else 0
    for index, (text_block, width, height) in enumerate(col_blocks):
        origin_x = left_panel_width + index * cell_width
        if alignment == "center":
            text_x = origin_x + (cell_width - width) / 2
        elif alignment == "right":
            text_x = origin_x + cell_width - header_padding - width
        else:
            text_x = origin_x + header_padding
        text_y = column_offset_y + header_padding
        draw_canvas.multiline_text(
            (text_x, text_y),
            text_block,
            font=font,
            fill=text_fill,
            spacing=spacing,
            align=alignment,
        )

    row_offset_x = axis_y_panel_width if axis_y_panel_width else 0
    for index, (text_block, width, height) in enumerate(row_blocks):
        origin_y = top_panel_height + index * cell_height
        text_x = header_padding + row_offset_x
        text_y = origin_y + (cell_height - height) / 2
        text_y = max(top_panel_height + header_padding, text_y)
        draw_canvas.multiline_text(
            (text_x, text_y),
            text_block,
            font=font,
            fill=text_fill,
            spacing=spacing,
            align="left",
        )

    return canvas


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

        checkpoint_choices = (UI_DISABLED_TOKEN,) + tuple(checkpoint_list)
        default_checkpoint = checkpoint_choices[1] if len(checkpoint_choices) > 1 else UI_DISABLED_TOKEN
        vae_choices = (UI_DISABLED_TOKEN, "<checkpoint>") + tuple(vae_list)
        clip_choices = (UI_DISABLED_TOKEN, "<checkpoint>") + tuple(clip_list)
        clip_vision_choices = (UI_DISABLED_TOKEN, "<checkpoint>") + tuple(clip_vision_list)

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
                "checkpoint": (
                    checkpoint_choices,
                    {"default": default_checkpoint},
                ),
                "vae_name": (vae_choices, {"default": "<checkpoint>"}),
                "clip_text_name": (clip_choices, {"default": "<checkpoint>"}),
                "clip_vision_name": (clip_vision_choices, {"default": "<checkpoint>"}),
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
                "axis_x": ("STRING", {"default": "", "forceInput": True}),
                "axis_y": ("STRING", {"default": "", "forceInput": True}),
                "axis_z": ("STRING", {"default": "", "forceInput": True}),
                "legacy_summary": ("STRING", {"default": "", "forceInput": True}),
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
        self._model_family_cache: Dict[int, str] = {}

    def _load_checkpoint(self, checkpoint_name: str) -> Tuple[Any, Any, Any]:
        resolved_path = _resolve_asset_path("checkpoints", checkpoint_name)
        if resolved_path is None:
            message = f"Checkpoint '{checkpoint_name}' could not be resolved"
            self.logger.error(message)
            raise FileNotFoundError(message)
        self.logger.trace(f"Loading checkpoint: {resolved_path}")
        checkpoint_family = _family_from_file(resolved_path, self.logger, "Checkpoint")
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
        try:
            setattr(model, "_h4_checkpoint_path", resolved_path)
        except Exception:  # pragma: no cover - attribute may be read-only
            pass
        try:
            setattr(model, "_h4_checkpoint_name", checkpoint_name)
        except Exception:  # pragma: no cover - attribute may be read-only
            pass
        if checkpoint_family:
            try:
                setattr(model, "_h4_checkpoint_family", checkpoint_family)
            except Exception:  # pragma: no cover - attribute may be read-only
                pass
            self._model_family_cache[id(model)] = checkpoint_family
        self.logger.info(f"Checkpoint ready: {checkpoint_name}")
        return model, clip, vae

    def _load_vae_override(self, current_vae: Any, vae_name: str) -> Any:
        if not vae_name or vae_name in {"<checkpoint>", UI_DISABLED_TOKEN}:
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
        text_override = clip_text_name not in (None, "", "<checkpoint>", UI_DISABLED_TOKEN)
        vision_override = clip_vision_name not in (None, "", "<checkpoint>", UI_DISABLED_TOKEN)
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

    def _conditioning_to_device(self, conditioning: Any, device: torch.device) -> Any:
        if conditioning is None:
            return None
        to_method = getattr(conditioning, "to", None)
        if callable(to_method) and not isinstance(conditioning, torch.Tensor):
            try:
                return to_method(device=device)
            except TypeError:
                try:
                    return to_method(device)
                except Exception:
                    pass
            except Exception:
                pass
        if torch.is_tensor(conditioning):
            source_device = conditioning.device
            try:
                return conditioning.to(device=device)
            except torch.cuda.OutOfMemoryError as exc:
                if device.type != "cuda":
                    raise
                self.logger.warn(
                    f"[DeviceCheck] Conditioning tensor move {source_device}->{device} OOM ({exc}); attempting recovery"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if conditioning.is_floating_point() and conditioning.dtype != torch.float16:
                    try:
                        return conditioning.to(device=device, dtype=torch.float16)
                    except torch.cuda.OutOfMemoryError as retry_exc:
                        self.logger.error(
                            f"[DeviceCheck] Conditioning tensor fallback to float16 failed ({retry_exc}); reraising"
                        )
                raise
        if isinstance(conditioning, list):
            return [self._conditioning_to_device(item, device) for item in conditioning]
        if isinstance(conditioning, tuple):
            return tuple(self._conditioning_to_device(item, device) for item in conditioning)
        if isinstance(conditioning, dict):
            return {key: self._conditioning_to_device(value, device) for key, value in conditioning.items()}
        return conditioning

    def _latent_to_device(self, latent: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        for key, value in list(latent.items()):
            if torch.is_tensor(value):
                latent[key] = value.to(device=device)
            elif hasattr(value, "to") and callable(getattr(value, "to")):
                to_method = getattr(value, "to")
                try:
                    latent[key] = to_method(device=device)
                except TypeError:
                    try:
                        latent[key] = to_method(device)
                    except Exception:
                        latent[key] = value
                except Exception:
                    latent[key] = value
            elif isinstance(value, dict):
                latent[key] = self._latent_to_device(dict(value), device)
            elif isinstance(value, list):
                latent[key] = [self._conditioning_to_device(item, device) for item in value]
            elif isinstance(value, tuple):
                latent[key] = tuple(self._conditioning_to_device(item, device) for item in value)
        return latent

    # --- LANDMARK: Device proxy cache helpers ---
    def _obtain_model_proxy(self, model: Any, device: torch.device) -> _DeviceAlignedModelProxy:
        if isinstance(model, _DeviceAlignedModelProxy):
            model.update_device(device)
            return model
        proxy = getattr(model, "_h4_device_proxy", None)
        if isinstance(proxy, _DeviceAlignedModelProxy):
            proxy.update_device(device)
            return proxy
        proxy = _DeviceAlignedModelProxy(model, device, self.logger)
        try:
            setattr(model, "_h4_device_proxy", proxy)
        except Exception:
            self.logger.debug("Device proxy memoization skipped; model does not expose attribute slot")
        return proxy

    def _release_model_proxy(self, model: Any) -> None:
        if model is None:
            return
        proxy = getattr(model, "_h4_device_proxy", None)
        if isinstance(proxy, _DeviceAlignedModelProxy):
            try:
                proxy.release()
            except Exception:
                pass
            try:
                delattr(model, "_h4_device_proxy")
            except AttributeError:
                setattr(model, "_h4_device_proxy", None)


    def _find_tensor_device_mismatches(
        self,
        payload: Any,
        device: torch.device,
    ) -> List[Tuple[str, torch.device]]:
        mismatches: List[Tuple[str, torch.device]] = []
        if payload is None:
            return mismatches
        visited: Set[int] = set()

        def _walk(value: Any, path: str) -> None:
            obj_id = id(value)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if torch.is_tensor(value):
                if value.device != device:
                    mismatches.append((path or "<root>", value.device))
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    key_path = f"{path}.{key}" if path else str(key)
                    _walk(item, key_path)
                return
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    key_path = f"{path}[{index}]" if path else f"[{index}]"
                    _walk(item, key_path)
                return
            attr_device = getattr(value, "device", None)
            if isinstance(attr_device, torch.device) and attr_device != device:
                mismatches.append((path or value.__class__.__name__, attr_device))

        _walk(payload, "")
        return mismatches

    def _enforce_payload_device(
        self,
        label: str,
        payload: Any,
        device: torch.device,
        converter: Callable[[Any, torch.device], Any],
    ) -> Any:
        if payload is None:
            return None
        mismatches = self._find_tensor_device_mismatches(payload, device)
        if not mismatches:
            return payload
        detail = ", ".join(f"{path}:{dev}" for path, dev in mismatches[:4])
        self.logger.warn(
            f"{label} contains tensors on unexpected device(s): {detail}; aligning to {device}."
        )
        adjusted = converter(payload, device)
        remaining = self._find_tensor_device_mismatches(adjusted, device)
        if remaining:
            detail_remaining = ", ".join(f"{path}:{dev}" for path, dev in remaining[:4])
            raise RuntimeError(
                f"{label} still reports tensors on incorrect device(s) after correction: {detail_remaining}"
            )
        return adjusted

    def _collect_tensor_devices(self, payload: Any) -> List[str]:
        devices: Set[str] = set()
        visited: Set[int] = set()

        def _walk(value: Any) -> None:
            obj_id = id(value)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if torch.is_tensor(value):
                devices.add(str(value.device))
                return
            if isinstance(value, dict):
                for item in value.values():
                    _walk(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _walk(item)
                return
            attr_device = getattr(value, "device", None)
            if isinstance(attr_device, torch.device):
                devices.add(str(attr_device))

        _walk(payload)
        return sorted(devices)

    def _resolve_device(self, model: Any, fallback: torch.device) -> torch.device:
        def _normalise_device(value: Any) -> Optional[torch.device]:
            if isinstance(value, torch.device):
                return value
            if isinstance(value, str):
                try:
                    return torch.device(value)
                except Exception:
                    return None
            return None

        preferred_gpu: Optional[torch.device] = None
        if torch.cuda.is_available():
            try:
                preferred_gpu = torch.device("cuda:0")
            except Exception:
                preferred_gpu = None

        cpu_candidate: Optional[torch.device] = None

        candidate = _normalise_device(getattr(model, "device", None))
        if candidate is not None:
            if candidate.type == "cuda":
                return candidate
            cpu_candidate = candidate

        module_attr = getattr(model, "model", None)
        if isinstance(module_attr, torch.nn.Module):
            try:
                module_param = next(module_attr.parameters())
            except StopIteration:
                module_param = None
            except Exception:
                module_param = None
            if module_param is not None:
                module_device = module_param.device
                if module_device.type == "cuda":
                    return module_device
                cpu_candidate = module_device

        try:
            resolved_raw = model_management.get_torch_device()
        except Exception:
            resolved_raw = None
        resolved = _normalise_device(resolved_raw)
        if resolved is not None:
            if resolved.type == "cuda":
                return resolved
            cpu_candidate = resolved

        if preferred_gpu is not None:
            return preferred_gpu
        if cpu_candidate is not None:
            return cpu_candidate
        return fallback

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
        lora_sequence = list(loras)
        if not lora_sequence:
            return patched_model, patched_clip

        def _record_family(target: Any, family: Optional[str]) -> None:
            if target is None or family is None:
                return
            try:
                setattr(target, "_h4_checkpoint_family", family)
            except Exception:  # pragma: no cover - attribute may be read-only
                pass
            self._model_family_cache[id(target)] = family

        active_family: Optional[str] = self._model_family_cache.get(id(model))
        if active_family is None:
            candidate = getattr(model, "_h4_checkpoint_family", None)
            active_family = _normalise_model_family_tag(candidate)
        if active_family is None:
            inferred = _infer_family_from_model_object(model)
            active_family = inferred
        if active_family is None:
            self.logger.info(
                "Checkpoint family could not be determined; LoRA compatibility checks will be best-effort"
            )
        else:
            _record_family(model, active_family)

        for lora_name, strength in lora_sequence:
            if strength == 0:
                self.logger.trace(f"Skipping LoRA {lora_name} with zero strength")
                continue
            resolved_path = _resolve_asset_path("loras", lora_name) or lora_name
            lora_family = _family_from_file(resolved_path, self.logger, "LoRA")
            if lora_family:
                self.logger.trace(f"LoRA {lora_name} declared family '{lora_family}'")
            else:
                self.logger.trace(f"LoRA {lora_name} has no detectable family metadata")
            if active_family and lora_family and active_family != lora_family:
                message = (
                    f"LoRA '{lora_name}' targets '{lora_family}' but checkpoint family is '{active_family}'"
                )
                self.logger.error(message)
                raise RuntimeError(message)
            if active_family is None and lora_family:
                active_family = lora_family
                self.logger.info(
                    f"Assuming checkpoint family '{active_family}' based on LoRA '{lora_name}' metadata"
                )
            self.logger.trace(f"Applying LoRA {lora_name} @ {strength}")
            patched_model, patched_clip = self.lora_loader.load_lora(
                patched_model, patched_clip, lora_name, strength, strength
            )
            if active_family:
                _record_family(patched_model, active_family)
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
        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise RuntimeError("Latent payload is missing a 'samples' tensor for sampling")
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
        if noise.device != samples.device:
            noise = noise.to(device=samples.device)
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
    axis_x: Optional[str] = None,
    axis_y: Optional[str] = None,
    axis_z: Optional[str] = None,
    legacy_summary: Optional[str] = None,
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

        # --- LANDMARK: Selection normalisation helpers ---
        def _normalise_checkpoint(value: Optional[str]) -> Optional[str]:
            candidate = str(value or "").strip()
            if not candidate or candidate.upper() == UI_DISABLED_TOKEN:
                return None
            return candidate

        def _resolve_model_checkpoint_tag(model_obj: Any) -> Optional[str]:
            if model_obj is None:
                return None
            candidates = (
                getattr(model_obj, "_h4_checkpoint_name", None),
                getattr(model_obj, "_h4_checkpoint_path", None),
                getattr(model_obj, "checkpoint_name", None),
            )
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    value = candidate.strip()
                    return os.path.basename(value) if os.path.sep in value else value
            inner = getattr(model_obj, "model", None)
            if inner is not None and inner is not model_obj:
                return _resolve_model_checkpoint_tag(inner)
            return None

        def _locate_component(root: Any, attrs: Tuple[str, ...]) -> Optional[Any]:
            for attr in attrs:
                if hasattr(root, attr):
                    value = getattr(root, attr)
                    if value is not None:
                        return value
            return None

        def _extract_embedded_components(model_obj: Any) -> Tuple[Optional[Any], Optional[Any]]:
            if model_obj is None:
                return (None, None)
            clip_candidate = _locate_component(
                model_obj,
                ("clip", "cond_stage_model", "clip_model", "text_encoder"),
            )
            if clip_candidate is None:
                inner = getattr(model_obj, "model", None)
                if inner is not None and inner is not model_obj:
                    clip_candidate = _locate_component(
                        inner,
                        ("clip", "cond_stage_model", "clip_model", "text_encoder"),
                    )
            vae_candidate = _locate_component(
                model_obj,
                ("first_stage_model", "vae", "autoencoder", "decoder"),
            )
            if vae_candidate is None:
                inner = getattr(model_obj, "model", None)
                if inner is not None and inner is not model_obj:
                    vae_candidate = _locate_component(
                        inner,
                        ("first_stage_model", "vae", "autoencoder", "decoder"),
                    )
            return clip_candidate, vae_candidate

        selected_checkpoint = _normalise_checkpoint(checkpoint)
        base_model_tag = _resolve_model_checkpoint_tag(model_in)
        primary_checkpoint_name = selected_checkpoint or base_model_tag
        prefer_inputs_for_base = bool(model_in is not None)
        base_key = "__input__" if prefer_inputs_for_base else (primary_checkpoint_name or "__anonymous__")
        if base_key == "__anonymous__":
            raise RuntimeError("No checkpoint selected and no MODEL input supplied.")

        active_bundle_key: Optional[str] = None
        active_bundle: Optional[Tuple[Any, Any, Any]] = None

        # --- LANDMARK: Model bundle cache ---
        def release_active_bundle() -> None:
            nonlocal active_bundle_key, active_bundle
            if active_bundle_key is None:
                return
            if active_bundle_key != "__input__":
                self.logger.info("Clearing previous checkpoint from memory")
                model_management.soft_empty_cache()
                if torch.cuda.is_available():  # pragma: no cover - device-specific
                    torch.cuda.empty_cache()
            if active_bundle is not None:
                self._release_model_proxy(active_bundle[0])
            active_bundle_key = None
            active_bundle = None

        def ensure_bundle(target_key: str, target_checkpoint: Optional[str], prefer_inputs: bool) -> Tuple[Any, Any, Any]:
            nonlocal active_bundle_key, active_bundle
            if active_bundle_key == target_key and active_bundle is not None:
                return active_bundle
            if active_bundle_key != target_key:
                release_active_bundle()

            base_model: Optional[Any] = None
            base_clip: Optional[Any] = None
            base_vae: Optional[Any] = None
            clip_from_input = False
            vae_from_input = False

            if prefer_inputs:
                if model_in is None:
                    raise RuntimeError("MODEL input is required when checkpoint selection is NONE.")
                base_model = model_in
                embedded_clip, embedded_vae = _extract_embedded_components(base_model)
                clip_candidate = clip_in or embedded_clip
                vae_candidate = vae_in or embedded_vae
                fallback_checkpoint = target_checkpoint or primary_checkpoint_name
                if (clip_candidate is None or vae_candidate is None) and fallback_checkpoint is not None:
                    loaded_model, loaded_clip, loaded_vae = self._load_checkpoint(fallback_checkpoint)
                    clip_candidate = clip_candidate or loaded_clip
                    vae_candidate = vae_candidate or loaded_vae
                missing: List[str] = []
                if clip_candidate is None:
                    missing.append("CLIP")
                if vae_candidate is None:
                    missing.append("VAE")
                if missing:
                    detail = " and ".join(missing)
                    raise RuntimeError(
                        f"Unable to resolve {detail} from supplied model; wire explicit inputs or select a checkpoint."
                    )
                base_clip = clip_candidate
                base_vae = vae_candidate
                clip_from_input = clip_in is not None
                vae_from_input = vae_in is not None
            else:
                if target_checkpoint is None:
                    raise RuntimeError("No checkpoint could be resolved for execution plan.")
                loaded_model, loaded_clip, loaded_vae = self._load_checkpoint(target_checkpoint)
                base_model = loaded_model
                base_clip = loaded_clip
                base_vae = loaded_vae
                if model_in is not None and primary_checkpoint_name is not None and target_checkpoint == primary_checkpoint_name:
                    base_model = model_in
                if clip_in is not None and primary_checkpoint_name is not None and target_checkpoint == primary_checkpoint_name:
                    base_clip = clip_in
                    clip_from_input = True
                if vae_in is not None and primary_checkpoint_name is not None and target_checkpoint == primary_checkpoint_name:
                    base_vae = vae_in
                    vae_from_input = True

            if base_model is None:
                raise RuntimeError("Model bundle missing model instance")

            final_vae = base_vae
            if final_vae is None:
                raise RuntimeError("Model bundle missing VAE instance before overrides")
            if not vae_from_input:
                final_vae = self._load_vae_override(final_vae, vae_name)
            if final_vae is None:
                raise RuntimeError("Model bundle missing VAE instance")

            final_clip = base_clip
            if final_clip is None:
                raise RuntimeError("Model bundle missing CLIP instance before overrides")
            if not clip_from_input:
                final_clip = self._load_clip_override(final_clip, clip_text_name, clip_vision_name)
            if final_clip is None:
                self.logger.warn("No CLIP available after overrides; using checkpoint CLIP as fallback")
                final_clip = base_clip
            if final_clip is None:
                raise RuntimeError("Model bundle missing CLIP instance")
            if clip_skip > 1:
                final_clip = self._apply_clip_skip(final_clip, clip_skip)

            bundle = (base_model, final_clip, final_vae)
            active_bundle_key = target_key
            active_bundle = bundle
            return bundle

        primary_model, primary_clip, primary_vae = ensure_bundle(
            base_key,
            primary_checkpoint_name,
            prefer_inputs_for_base,
        )

        if bypass_engine:
            self.logger.info("Bypass mode enabled; relaying inputs without sampling")
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
        axis_driver_style: Optional[Dict[str, Any]] = None

        def _apply_axis_driver_override(slot_label: str, payload_text: Optional[str]) -> Optional[List[AxisDescriptor]]:
            if not payload_text:
                return None
            try:
                payload = json.loads(payload_text)
            except Exception as exc:
                self.logger.warn(f"Axis Driver payload for {slot_label} axis could not be parsed: {exc}")
                return None
            descriptors, style = _axis_driver_payload_to_descriptors(payload)
            nonlocal axis_driver_style
            if style and axis_driver_style is None:
                axis_driver_style = style
            if descriptors:
                self.logger.info(
                    f"Axis Driver override applied to {slot_label} axis with {len(descriptors)} item(s)"
                )
                return descriptors
            return None

        override_x = _apply_axis_driver_override("X", axis_x)
        if override_x is not None:
            x_descriptors = override_x
        override_y = _apply_axis_driver_override("Y", axis_y)
        if override_y is not None:
            y_descriptors = override_y
        override_z = _apply_axis_driver_override("Z", axis_z)
        if override_z is not None:
            self.logger.info(
                "Axis Driver Z axis data received; third-dimension plotting will be supported in a future release"
            )
        if legacy_summary and legacy_summary.strip():
            self.logger.trace("Axis Driver summary:\n" + legacy_summary.strip())
        if axis_driver_style:
            style_snapshot = {
                "label_layout": axis_driver_style.get("label_layout"),
                "font_family": axis_driver_style.get("font_family"),
                "font_size": axis_driver_style.get("font_size"),
            }
            self.logger.trace(
                "Axis Driver style preferences captured: "
                + ", ".join(f"{key}={value}" for key, value in style_snapshot.items() if value is not None)
            )

        plans = build_generation_matrix(
            base_checkpoint=primary_checkpoint_name,
            x_descriptors=x_descriptors,
            y_descriptors=y_descriptors,
        )
        if not plans:
            raise RuntimeError("No execution plans could be constructed from the provided axis values")
        self.logger.info(f"Generated {len(plans)} execution plan(s)")
        grid_rows = max(1, len(y_descriptors))
        grid_cols = max(1, len(x_descriptors))

        # --- LANDMARK: Execution plan grouping ---
        plan_targets: List[Dict[str, Any]] = []
        plan_groups: Dict[str, List[int]] = {}
        group_order: List[str] = []

        for index, plan in enumerate(plans):
            resolved_checkpoint = _normalise_checkpoint(plan.checkpoint_name)
            if resolved_checkpoint is None:
                resolved_checkpoint = primary_checkpoint_name
            prefer_inputs = False
            if model_in is not None:
                if resolved_checkpoint is None:
                    prefer_inputs = True
                elif base_model_tag and resolved_checkpoint == base_model_tag:
                    prefer_inputs = True
                elif resolved_checkpoint == primary_checkpoint_name:
                    prefer_inputs = True
            key = "__input__" if prefer_inputs else (resolved_checkpoint or "__anonymous__")
            plan_targets.append(
                {
                    "checkpoint": resolved_checkpoint,
                    "prefer_inputs": prefer_inputs,
                    "key": key,
                }
            )
            if key not in plan_groups:
                plan_groups[key] = []
                group_order.append(key)
            plan_groups[key].append(index)

        if "__anonymous__" in plan_groups:
            raise RuntimeError(
                "No checkpoint available for one or more plans; supply a MODEL input or select a checkpoint."
            )

        base_latent = self._prepare_latent(
            primary_vae,
            width,
            height,
            latent_in,
            image_in,
        )

        grid_slots: List[Optional[torch.Tensor]] = [None] * len(plans)
        stack_slots: List[Optional[torch.Tensor]] = [None] * len(plans)
        summary_slots: List[Optional[Dict[str, Any]]] = [None] * len(plans)

        final_index = len(plans) - 1
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

        for group_key in group_order:
            indices = plan_groups[group_key]
            meta_first = plan_targets[indices[0]]
            active_model, active_clip, active_vae = ensure_bundle(
                meta_first["key"],
                meta_first["checkpoint"],
                meta_first["prefer_inputs"],
            )
            for plan_index in indices:
                plan = plans[plan_index]
                meta = plan_targets[plan_index]
                if meta["key"] != group_key:
                    active_model, active_clip, active_vae = ensure_bundle(
                        meta["key"],
                        meta["checkpoint"],
                        meta["prefer_inputs"],
                    )
                self.logger.info(f"Executing plan {plan_index + 1}/{len(plans)} :: {plan.label}")
                if plan_index == final_index:
                    final_model = active_model
                    final_clip = active_clip
                    final_vae = active_vae
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

                use_base_latent = meta["prefer_inputs"] or meta["checkpoint"] == primary_checkpoint_name
                if use_base_latent:
                    latent_copy = clone_latent(base_latent)
                else:
                    latent_seed = self._prepare_latent(active_vae, width, height, latent_in, image_in)
                    latent_copy = clone_latent(latent_seed)

                sample_tensor = latent_copy.get("samples")
                if isinstance(sample_tensor, torch.Tensor):
                    sample_device = sample_tensor.device
                else:
                    sample_device = torch.device("cpu")
                target_device = self._resolve_device(patched_model, sample_device)
                try:
                    patched_model = self._obtain_model_proxy(patched_model, target_device)
                except Exception as exc:
                    self.logger.warn(f"Device alignment proxy unavailable; continuing without shim ({exc})")

                latent_copy = cast(
                    Dict[str, torch.Tensor],
                    self._enforce_payload_device(
                        "Latent payload",
                        latent_copy,
                        target_device,
                        self._latent_to_device,
                    ),
                )
                positive = self._enforce_payload_device(
                    "Positive conditioning",
                    positive,
                    target_device,
                    self._conditioning_to_device,
                )
                negative = self._enforce_payload_device(
                    "Negative conditioning",
                    negative,
                    target_device,
                    self._conditioning_to_device,
                )

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
                grid_slots[plan_index] = decoded_cpu
                stack_slots[plan_index] = decoded_cpu
                latent_samples = result_latent.get("samples") if isinstance(result_latent, dict) else None
                plan_summary: Dict[str, Any] = {
                    "label": plan.label,
                    "checkpoint": meta["checkpoint"] or primary_checkpoint_name or "<input>",
                    "seed": seed_current,
                    "steps": steps_current,
                    "cfg": cfg_current,
                    "denoise": denoise_current,
                    "sampler": sampler_current,
                    "scheduler": scheduler_current,
                    "decoded_shape": tuple(decoded_cpu.shape),
                }
                if isinstance(latent_samples, torch.Tensor):
                    latent_samples_tensor = cast(torch.Tensor, latent_samples)
                    plan_summary["latent_shape"] = tuple(latent_samples_tensor.shape)
                plan_summary["pixel_range"] = (
                    float(decoded_cpu.min().item()),
                    float(decoded_cpu.max().item()),
                )
                summary_slots[plan_index] = plan_summary

                if plan_index == final_index:
                    final_latent = result_latent
                    final_positive = positive
                    final_negative = negative

                if torch.cuda.is_available():  # pragma: no cover - device-specific
                    torch.cuda.empty_cache()

        release_active_bundle()

        plan_summaries: List[Dict[str, Any]] = []
        grid_images: List[torch.Tensor] = []
        image_stack: List[torch.Tensor] = []
        for idx in range(len(plans)):
            grid_tensor = grid_slots[idx]
            stack_tensor = stack_slots[idx]
            summary_payload = summary_slots[idx]
            if grid_tensor is None or stack_tensor is None or summary_payload is None:
                raise RuntimeError(f"Execution plan {plans[idx].label} did not produce an image result")
            grid_images.append(grid_tensor)
            image_stack.append(stack_tensor)
            plan_summaries.append(summary_payload)

        grid = compose_image_grid(grid_images, grid_rows, grid_cols)
        grid = annotate_grid_image(
            grid,
            plans,
            grid_rows,
            grid_cols,
            self.logger,
            style=axis_driver_style,
            x_axis=x_descriptors,
            y_axis=y_descriptors,
        )
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


class h4_Varianator:
    """Standalone variation generator that riffs on a supplied latent."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802
        sampler_default = SAMPLER_CHOICES[0] if SAMPLER_CHOICES else "euler"
        scheduler_default = SCHEDULER_CHOICES[0] if SCHEDULER_CHOICES else "normal"
        profile_keys = list(VARIANATOR_PROFILE_RANGES.keys())
        if not profile_keys:
            profile_keys = ["moderate"]
        profile_choices = tuple(profile_keys)
        profile_default = profile_keys[1] if len(profile_keys) > 1 else profile_keys[0]
        return {
            "required": {
                "variation_count": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": VARIANATOR_MAX_VARIATIONS,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("variation_count"),
                    },
                ),
                "variation_profile": (
                    profile_choices,
                    {
                        "default": profile_default,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("variation_profile"),
                    },
                ),
                "seed_mode": (
                    ["fixed", "increment", "random"],
                    {
                        "default": "increment",
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("seed_mode"),
                    },
                ),
                "base_seed": (
                    "INT",
                    {
                        "default": 123456789,
                        "min": 0,
                        "max": VARIANATOR_SEED_LIMIT,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("base_seed"),
                    },
                ),
                "sampler_name": (
                    SAMPLER_CHOICES,
                    {
                        "default": sampler_default,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("sampler_name"),
                    },
                ),
                "scheduler_name": (
                    SCHEDULER_CHOICES,
                    {
                        "default": scheduler_default,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("scheduler_name"),
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 18,
                        "min": 1,
                        "max": 150,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("steps"),
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 1.0,
                        "max": 30.0,
                        "step": 0.1,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("cfg"),
                    },
                ),
                "go_ultra": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label": " GO PLUS ULTRA?! ",
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("go_ultra"),
                    },
                ),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "latent_in": ("LATENT", {"forceInput": True}),
                "positive_in": ("CONDITIONING",),
                "negative_in": ("CONDITIONING",),
                "style_positive_in": ("CONDITIONING",),
                "style_negative_in": ("CONDITIONING",),
                "prompt_jitter_enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("prompt_jitter_enabled"),
                    },
                ),
                "prompt_jitter_strength": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("prompt_jitter_strength"),
                    },
                ),
                "prompt_jitter_tokens": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("prompt_jitter_tokens"),
                    },
                ),
                "style_mix_enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("style_mix_enabled"),
                    },
                ),
                "style_mix_ratio": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("style_mix_ratio"),
                    },
                ),
                "base_positive_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("base_positive_prompt"),
                    },
                ),
                "base_negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("base_negative_prompt"),
                    },
                ),
                "style_mix_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("style_mix_prompt"),
                    },
                ),
                "style_mix_negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("style_mix_negative_prompt"),
                    },
                ),
                "ultra_json_log": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("ultra_json_log"),
                    },
                ),
                "ultra_cache_artifacts": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": VARIANATOR_WIDGET_TOOLTIPS.get("ultra_cache_artifacts"),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("variations", "latent_batch", "summary")
    FUNCTION = "generate"
    CATEGORY = "h4 Toolkit/Generation"

    def __init__(self) -> None:
        self.logger = ToolkitLogger("h4_Varianator")
        self._engine = h4_PlotXY()
        self._engine.logger = self.logger

    @staticmethod
    def _normalise_count(value: Any) -> int:
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 1
        return max(1, min(count, VARIANATOR_MAX_VARIATIONS))

    @staticmethod
    def _coerce_seed(value: Any) -> int:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            integer = 0
        return max(0, min(integer, VARIANATOR_SEED_LIMIT))

    @staticmethod
    def _normalise_seed_mode(mode: Optional[str]) -> str:
        candidate = str(mode or "").strip().lower()
        return candidate if candidate in {"fixed", "increment", "random"} else "fixed"

    @staticmethod
    def _resolve_profile(profile: Optional[str]) -> Tuple[str, Tuple[float, float]]:
        key = str(profile or "").strip().lower()
        if key not in VARIANATOR_PROFILE_RANGES:
            key = "moderate"
        low, high = VARIANATOR_PROFILE_RANGES.get(key, (0.4, 0.5))
        if low > high:
            low, high = high, low
        low = max(0.0, min(low, 1.0))
        high = max(0.0, min(high, 1.0))
        return key, (low, high)

    def _parse_jitter_tokens(self, raw: str, default_strength: float) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if not raw:
            return entries
        default_low = max(0.0, 1.0 - float(default_strength))
        default_high = max(default_low, 1.0 + float(default_strength))
        for line in raw.splitlines():
            token = line.strip()
            if not token:
                continue
            pieces = [segment.strip() for segment in token.split("|") if segment.strip()]
            if not pieces:
                continue
            name = pieces[0]
            if not name:
                continue
            try:
                lower = float(pieces[1]) if len(pieces) > 1 else default_low
                upper = float(pieces[2]) if len(pieces) > 2 else default_high
            except (TypeError, ValueError):
                lower, upper = default_low, default_high
            if upper < lower:
                lower, upper = upper, lower
            lower = max(0.0, min(lower, 3.0))
            upper = max(0.0, min(upper, 3.5))
            entries.append({"token": name, "min": lower, "max": upper})
        return entries

    def _apply_prompt_jitter(
        self,
        base_prompt: str,
        entries: List[Dict[str, Any]],
        rng: random.Random,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        prompt = base_prompt.strip() if base_prompt else ""
        adjustments: List[Dict[str, Any]] = []
        for entry in entries:
            low = float(entry["min"])
            high = float(entry["max"])
            if high < low:
                low, high = high, low
            weight = rng.uniform(low, high)
            adjustments.append({"token": entry["token"], "weight": weight})
        if adjustments:
            jitter_line = " ".join(f"({item['token']}:{item['weight']:.3f})" for item in adjustments)
            prompt = f"{prompt}\n{jitter_line}" if prompt else jitter_line
        return prompt, adjustments

    def _blend_conditioning(self, base: Any, style: Any, ratio: float) -> Any:
        ratio = max(0.0, min(float(ratio), 1.0))
        if style is None or ratio <= 0.0:
            return self._engine._clone_conditioning(base)
        if base is None:
            return None
        if torch.is_tensor(base) and torch.is_tensor(style):
            return base * (1.0 - ratio) + style * ratio
        if isinstance(base, dict) and isinstance(style, dict):
            blended: Dict[str, Any] = {}
            for key in set(base.keys()) | set(style.keys()):
                if key in base and key in style:
                    blended[key] = self._blend_conditioning(base[key], style[key], ratio)
                elif key in base:
                    blended[key] = self._engine._clone_conditioning(base[key])
                else:
                    blended[key] = self._engine._clone_conditioning(style[key])
            return blended
        if isinstance(base, (list, tuple)) and isinstance(style, (list, tuple)):
            limit = min(len(base), len(style))
            blended_items = [self._blend_conditioning(base[idx], style[idx], ratio) for idx in range(limit)]
            if isinstance(base, tuple):
                tail = [self._engine._clone_conditioning(item) for item in base[limit:]]
                return tuple(blended_items + tail)
            tail_list = [self._engine._clone_conditioning(item) for item in base[limit:]]
            return blended_items + tail_list
        return self._engine._clone_conditioning(base)

    def _ensure_ultra_directory(self) -> Optional[str]:
        base_dir: Optional[str] = None
        get_temp_dir = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp_dir):
            try:
                candidate = get_temp_dir()
                if isinstance(candidate, str):
                    base_dir = candidate
            except Exception:
                base_dir = None
        if not base_dir:
            get_output_dir = getattr(folder_paths, "get_output_directory", None)
            if callable(get_output_dir):
                try:
                    candidate = get_output_dir()
                    if isinstance(candidate, str):
                        base_dir = candidate
                except Exception:
                    base_dir = None
        if not base_dir:
            return None
        target = os.path.join(base_dir, "h4_varianator_ultra")
        try:
            os.makedirs(target, exist_ok=True)
        except Exception as exc:
            self.logger.warn(f"Unable to create Go_Plus_ULTRA artefact directory: {exc}")
            return None
        return target

    def _tensor_to_image(self, tensor: torch.Tensor) -> Optional[Any]:
        if Image is None or tensor is None:
            return None
        try:
            normalised = self._engine._normalise_image_tensor(tensor)
            sample = normalised[0].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
            array = sample.numpy()
            if array.ndim == 2:
                array = np.expand_dims(array, axis=-1)
            channels = array.shape[-1]
            if channels == 1:
                array = np.repeat(array, 3, axis=-1)
            elif channels == 2:
                pad = np.zeros((*array.shape[:-1], 1), dtype=array.dtype)
                array = np.concatenate([array, pad], axis=-1)
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(array)
        except Exception as exc:
            self.logger.warn(f"Failed to convert tensor to preview image: {exc}")
            return None

    def _write_ultra_json(self, payload: Dict[str, Any], persist: bool) -> Tuple[Optional[str], str]:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if not persist:
            return None, text
        directory = self._ensure_ultra_directory()
        if directory is None:
            return None, text
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(directory, f"varianator_{timestamp}.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            return path, text
        except Exception as exc:
            self.logger.warn(f"Unable to write Go_Plus_ULTRA log: {exc}")
            return None, text

    def generate(
        self,
        variation_count: int,
        variation_profile: str,
        seed_mode: str,
        base_seed: int,
        sampler_name: str,
        scheduler_name: str,
        steps: int,
        cfg: float,
        go_ultra: bool,
        model: Optional[Any] = None,
        clip: Optional[Any] = None,
        vae: Optional[Any] = None,
        latent_in: Optional[Dict[str, torch.Tensor]] = None,
        positive_in: Optional[Any] = None,
        negative_in: Optional[Any] = None,
        style_positive_in: Optional[Any] = None,
        style_negative_in: Optional[Any] = None,
        prompt_jitter_enabled: bool = False,
        prompt_jitter_strength: float = 0.15,
        prompt_jitter_tokens: str = "",
        style_mix_enabled: bool = False,
        style_mix_ratio: float = 0.35,
        base_positive_prompt: str = "",
        base_negative_prompt: str = "",
        style_mix_prompt: str = "",
        style_mix_negative_prompt: str = "",
        ultra_json_log: bool = True,
        ultra_cache_artifacts: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        self.logger.info("Varianator engaged")
        count = self._normalise_count(variation_count)
        profile_key, (profile_min, profile_max) = self._resolve_profile(variation_profile)
        mode = self._normalise_seed_mode(seed_mode)
        anchor_seed = self._coerce_seed(base_seed)
        steps = max(1, int(steps))
        cfg = float(cfg)
        if model is None or vae is None:
            raise RuntimeError("Varianator requires MODEL and VAE inputs.")
        if latent_in is None:
            raise RuntimeError("Varianator requires a latent input to remix.")

        sampler_label = sampler_name or (SAMPLER_CHOICES[0] if SAMPLER_CHOICES else "euler")
        scheduler_label = scheduler_name or (SCHEDULER_CHOICES[0] if SCHEDULER_CHOICES else "normal")

        jitter_entries = self._parse_jitter_tokens(prompt_jitter_tokens, float(prompt_jitter_strength))
        style_mix_ratio = max(0.0, min(float(style_mix_ratio), 1.0))

        seed_rng = random.Random(anchor_seed)
        denoise_rng = random.Random(anchor_seed ^ 0xBADF00D)

        style_positive_source: Optional[Any] = None
        style_negative_source: Optional[Any] = None
        if style_mix_enabled:
            if style_positive_in is not None:
                style_positive_source = self._engine._clone_conditioning(style_positive_in)
            elif style_mix_prompt.strip() and clip is not None:
                style_positive_source = self._engine._encode_prompt(clip, style_mix_prompt.strip())
            if style_negative_in is not None:
                style_negative_source = self._engine._clone_conditioning(style_negative_in)
            elif style_mix_negative_prompt.strip() and clip is not None:
                style_negative_source = self._engine._encode_prompt(clip, style_mix_negative_prompt.strip())
            if style_mix_enabled and style_positive_source is None and style_negative_source is None:
                self.logger.warn("Style mix enabled but no style source provided; disabling mix")
                style_mix_enabled = False

        variations: List[torch.Tensor] = []
        latent_batches: List[torch.Tensor] = []
        metadata: List[Dict[str, Any]] = []
        last_latent_dict: Optional[Dict[str, torch.Tensor]] = None

        for index in range(count):
            if mode == "fixed":
                seed_value = anchor_seed
            elif mode == "increment":
                seed_value = self._coerce_seed(anchor_seed + index)
            else:
                seed_value = seed_rng.randint(0, VARIANATOR_SEED_LIMIT)
            denoise_value = float(denoise_rng.uniform(profile_min, profile_max))

            prompt_text_used: Optional[str] = None
            jitter_applied: List[Dict[str, Any]] = []

            if positive_in is not None and not (prompt_jitter_enabled or style_mix_enabled):
                positive = self._engine._clone_conditioning(positive_in)
            else:
                if clip is None:
                    raise RuntimeError(
                        "Varianator needs a CLIP input when prompt jitter or style mixing is active or no positive conditioning is wired."
                    )
                prompt_seed = (anchor_seed + index * 7919) & 0xFFFFFFFFFFFF
                jitter_rng = random.Random(prompt_seed)
                prompt_text = base_positive_prompt or ""
                if prompt_jitter_enabled and jitter_entries:
                    prompt_text, jitter_applied = self._apply_prompt_jitter(prompt_text, jitter_entries, jitter_rng)
                prompt_text_used = prompt_text
                positive = self._engine._encode_prompt(clip, prompt_text)

            if style_mix_enabled and style_positive_source is not None:
                positive = self._blend_conditioning(positive, style_positive_source, style_mix_ratio)

            if negative_in is not None:
                negative = self._engine._clone_conditioning(negative_in)
            else:
                if clip is None:
                    negative = []
                else:
                    negative_prompt = base_negative_prompt or ""
                    negative = self._engine._encode_prompt(clip, negative_prompt)

            if style_mix_enabled and style_negative_source is not None:
                negative = self._blend_conditioning(negative, style_negative_source, style_mix_ratio)

            latent_seed = clone_latent(latent_in)
            result_latent = self._engine._run_sampler(
                model=model,
                vae=vae,
                positive=positive,
                negative=negative,
                latent=latent_seed,
                seed=seed_value,
                steps=steps,
                cfg=cfg,
                denoise=max(0.0, min(denoise_value, 1.0)),
                sampler_name=sampler_label,
                scheduler_name=scheduler_label,
            )
            last_latent_dict = result_latent if isinstance(result_latent, dict) else None
            decoded = self._engine._decode(vae, result_latent)
            decoded_cpu = decoded.detach().clone().to(device="cpu")
            variations.append(decoded_cpu)

            latent_tensor = None
            if isinstance(result_latent, dict):
                latent_tensor = result_latent.get("samples")
            elif isinstance(result_latent, torch.Tensor):
                latent_tensor = result_latent
            if isinstance(latent_tensor, torch.Tensor):
                tensor_value = cast(torch.Tensor, latent_tensor)
                latent_cpu = tensor_value.detach().clone().to(device="cpu")
                latent_batches.append(latent_cpu)

            preview_path: Optional[str] = None
            if go_ultra and ultra_cache_artifacts:
                image = self._tensor_to_image(decoded_cpu)
                directory = self._ensure_ultra_directory()
                if image is not None and directory is not None:
                    filename = f"variation_{index + 1:02d}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png"
                    target = os.path.join(directory, filename)
                    try:
                        image.save(target)
                        preview_path = target
                    except Exception as exc:
                        self.logger.warn(f"Failed to persist variation preview: {exc}")

            entry = {
                "index": index,
                "seed": int(seed_value),
                "denoise": float(round(denoise_value, 4)),
                "profile": profile_key,
                "prompt": prompt_text_used,
                "jitter": jitter_applied,
                "style_mix_ratio": style_mix_ratio if style_mix_enabled else 0.0,
                "image_path": preview_path,
                "image_shape": tuple(decoded_cpu.shape),
            }
            if latent_batches:
                entry["latent_shape"] = tuple(latent_batches[-1].shape)
            metadata.append(entry)

            if torch.cuda.is_available():  # pragma: no cover - device-specific cleanup
                torch.cuda.empty_cache()

        model_management.soft_empty_cache()

        if not variations:
            raise RuntimeError("No variations were generated")

        images_out = torch.cat(variations, dim=0).to(dtype=torch.float32)

        if latent_batches:
            latent_out: Dict[str, torch.Tensor] = {"samples": torch.cat(latent_batches, dim=0)}
        elif last_latent_dict and isinstance(last_latent_dict.get("samples"), torch.Tensor):
            latent_out = {"samples": last_latent_dict["samples"].detach().clone().to(device="cpu")}
        else:
            latent_out = clone_latent(latent_in)
            samples = latent_out.get("samples")
            if isinstance(samples, torch.Tensor):
                samples_value = cast(torch.Tensor, samples)
                samples_cpu = samples_value.detach().clone().to(device="cpu")
                latent_out["samples"] = samples_cpu

        summary_lines = [
            f"Varianator generated {len(metadata)} variation(s) using profile '{profile_key}'.",
            f"Sampler={sampler_label} Scheduler={scheduler_label} Steps={steps} CFG={cfg:.2f}",
        ]
        for entry in metadata:
            line = f"[{entry['index'] + 1:02d}] seed={entry['seed']} denoise={entry['denoise']:.3f}"
            if entry.get("image_path"):
                line += f" preview={entry['image_path']}"
            summary_lines.append(line)
            if entry.get("jitter"):
                jitter_desc = ", ".join(
                    f"{item['token']}:{item['weight']:.2f}"
                    for item in entry["jitter"]
                )
                summary_lines.append(f"    jitter -> {jitter_desc}")

        ultra_path: Optional[str] = None
        ultra_payload_text: Optional[str] = None
        if go_ultra:
            ultra_payload = {
                "node": "h4_Varianator",
                "version": VARIANATOR_VERSION,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "profile": profile_key,
                "seed_mode": mode,
                "seed_anchor": anchor_seed,
                "sampler": sampler_label,
                "scheduler": scheduler_label,
                "steps": steps,
                "cfg": cfg,
                "variations": metadata,
            }
            if ultra_json_log:
                ultra_path, ultra_payload_text = self._write_ultra_json(ultra_payload, ultra_cache_artifacts)
                if ultra_path:
                    self.logger.info(f"Go_Plus_ULTRA :: JSON log saved -> {ultra_path}")
                elif not ultra_cache_artifacts:
                    self.logger.info("Go_Plus_ULTRA :: JSON log generated (not persisted)")
                else:
                    self.logger.warn("Go_Plus_ULTRA :: Failed to persist JSON log")
            else:
                ultra_payload_text = json.dumps(ultra_payload, indent=2, ensure_ascii=False)

        if ultra_path:
            summary_lines.append(f"Ultra log saved -> {ultra_path}")
        elif ultra_json_log and go_ultra and not ultra_cache_artifacts:
            summary_lines.append("Ultra log generated (not persisted)")

        summary_text = "\n".join(summary_lines)
        self.logger.info(summary_text)
        return images_out, latent_out, summary_text


class h4_DebugATron3000:
    """Adaptive router that mirrors signals per data type while logging everything."""

    SLOT_DEFINITIONS: Tuple[Tuple[str, str, str], ...] = DEBUG_SLOT_DEFINITIONS
    SLOT_TOOLTIPS: Dict[str, str] = DEBUG_SLOT_TOOLTIPS
    VAE_MODEL_FALLBACKS: Dict[str, str] = {"vae_in": "model_in"}
    BRANCH_DISPLAY_NAMES: Dict[str, str] = {}
    LEGACY_INPUT_ALIASES: Dict[str, str] = {
        "model": "model_in",
        "clip": "clip_in",
        "clip_vision": "clip_vision_in",
        "vae": "vae_in",
        "conditioning": "conditioning_in",
        "positive": "conditioning_positive_in",
        "negative": "conditioning_negative_in",
        "latent": "latent_in",
        "image": "image_in",
        "mask": "mask_in",
    }

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
                candidate = get_temp_dir()
                if isinstance(candidate, str):
                    base_dir = candidate
            except Exception:  # pragma: no cover - environment specific
                base_dir = None
        if not base_dir:
            get_output_dir = getattr(folder_paths, "get_output_directory", None)
            if callable(get_output_dir):
                try:
                    candidate = get_output_dir()
                    if isinstance(candidate, str):
                        base_dir = candidate
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
        requested_mode = mode
        mode_normalised = (mode or "passthrough").strip().lower()
        if mode_normalised not in {"monitor", "passthrough"}:
            self.logger.warn(
                f"Debug router received unsupported mode '{requested_mode}'; falling back to passthrough"
            )
            mode_normalised = "passthrough"
        self.logger.info(f"Debug router engaged :: mode={mode_normalised}")
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
        for legacy_name, canonical_name in self.LEGACY_INPUT_ALIASES.items():
            if canonical_name not in dynamic_slots and legacy_name in dynamic_slots:
                dynamic_slots[canonical_name] = dynamic_slots[legacy_name]
                context_snapshot[canonical_name] = dynamic_slots[legacy_name]
                self.logger.warn(
                    f"Detected legacy input '{legacy_name}'. Please reconnect to '{canonical_name}' for future runs."
                )
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
                if mode_normalised in {"passthrough", "monitor"}
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
                "mode": mode_normalised,
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
            f"Mode: {mode_normalised}",
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


class h4_ExecutionLogger(h4_DebugATron3000):
    """Console mirror node that summarises connected payloads into a log string."""

    SLOT_DEFINITIONS = DEBUG_SLOT_DEFINITIONS
    SLOT_TOOLTIPS = DEBUG_SLOT_TOOLTIPS
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_summary",)
    FUNCTION = "log"
    CATEGORY = "h4 Toolkit/Debug"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802
        optional_inputs: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for slot_name, display_name, slot_type in cls.SLOT_DEFINITIONS:
            tooltip = cls.SLOT_TOOLTIPS.get(slot_name)
            options: Dict[str, Any] = {"default": None, "label": display_name}
            if tooltip:
                options["tooltip"] = tooltip
            optional_inputs[slot_name] = (slot_type, options)
        optional_inputs.update(
            {
                "show_types": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "Show payload types",
                        "tooltip": EXECUTION_LOGGER_WIDGET_TOOLTIPS.get("show_types"),
                    },
                ),
                "show_shapes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "Show tensor shapes",
                        "tooltip": EXECUTION_LOGGER_WIDGET_TOOLTIPS.get("show_shapes"),
                    },
                ),
            }
        )
        return {
            "required": {
                "log_level": (
                    ["INFO", "DEBUG", "WARNING"],
                    {
                        "default": "INFO",
                        "label": "Log level",
                        "tooltip": EXECUTION_LOGGER_WIDGET_TOOLTIPS.get("log_level"),
                    },
                ),
                "prefix": (
                    "STRING",
                    {
                        "default": "[EXEC]",
                        "tooltip": EXECUTION_LOGGER_WIDGET_TOOLTIPS.get("prefix"),
                    },
                ),
            },
            "optional": optional_inputs,
        }

    def __init__(self) -> None:
        super().__init__()
        self.logger = ToolkitLogger("h4_ExecutionLogger")

    def log(
        self,
        log_level: str,
        prefix: str,
        show_types: bool = True,
        show_shapes: bool = True,
        **dynamic_slots: Any,
    ) -> Tuple[str]:
        level = (log_level or "INFO").strip().upper()
        log_methods = {
            "DEBUG": self.logger.debug,
            "INFO": self.logger.info,
            "WARNING": self.logger.warn,
        }
        emit = log_methods.get(level, self.logger.info)
        summary_lines: List[str] = []
        for slot_name, display_name, slot_type in self.SLOT_DEFINITIONS:
            payload = dynamic_slots.get(slot_name)
            if payload is None:
                continue
            descriptor = self._summarise_object(payload)
            line = f"{prefix} {display_name}: {descriptor}".strip()
            emit(line)
            summary_lines.append(f"{display_name}: {descriptor}")
            if show_types:
                type_name = type(payload).__name__
                type_line = f"    type={type_name}"
                emit(f"{prefix} {type_line}".strip())
                summary_lines.append(type_line)
            if show_shapes:
                for detail in self._describe_payload(payload, slot_type):
                    detail_line = f"    {detail}"
                    emit(f"{prefix} {detail}".strip())
                    summary_lines.append(detail_line)
        if not summary_lines:
            summary_lines.append(f"{prefix} No inputs connected.")
            emit(summary_lines[0])
        summary_text = "\n".join(summary_lines)
        return (summary_text,)


class h4_DebugATron3000Console(h4_DebugATron3000):
    """Output-node variant that emits an HTML dossier alongside routed payloads."""

    SLOT_DEFINITIONS = DEBUG_SLOT_DEFINITIONS
    SLOT_TOOLTIPS = DEBUG_SLOT_TOOLTIPS
    RETURN_TYPES = ("STRING",) + DEBUG_SLOT_RETURN_TYPES
    RETURN_NAMES = ("html_log",) + DEBUG_SLOT_RETURN_NAMES
    FUNCTION = "render"
    CATEGORY = "h4 Toolkit/Debug"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802
        config = super().INPUT_TYPES()
        optional = config.setdefault("optional", {})
        optional.update(
            {
                "display_mode": (
                    ["console_only", "html_only", "both"],
                    {
                        "default": "both",
                        "label": "Display mode",
                        "tooltip": "Choose whether to emit console logs, HTML, or both.",
                    },
                ),
                "color_scheme": (
                    list(CONSOLE_COLOR_SCHEMES.keys()),
                    {
                        "default": "cyberpunk",
                        "label": "Colour scheme",
                        "tooltip": "Palette applied to the rendered HTML dossier.",
                    },
                ),
                "verbosity": (
                    list(CONSOLE_VERBOSITY_LEVELS.keys()),
                    {
                        "default": "normal",
                        "label": "Verbosity",
                        "tooltip": "Controls how many diagnostic lines are included per slot.",
                    },
                ),
                "show_timestamps": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "Show timestamps",
                        "tooltip": "When enabled, each slot section includes the current time stamp.",
                    },
                ),
            }
        )
        return config

    def __init__(self) -> None:
        super().__init__()
        self.logger = ToolkitLogger("h4_DebugATronConsole")

    def render(
        self,
        mode: str,
        display_mode: str = "both",
        color_scheme: str = "cyberpunk",
        verbosity: str = "normal",
        show_timestamps: bool = True,
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
    ) -> Tuple[str, ...]:
        routed_outputs = super().route(
            mode,
            go_ultra=go_ultra,
            ultra_capture_first_step=ultra_capture_first_step,
            ultra_capture_mid_step=ultra_capture_mid_step,
            ultra_capture_last_step=ultra_capture_last_step,
            ultra_preview_images=ultra_preview_images,
            ultra_json_log=ultra_json_log,
            ultra_highlight_missing_conditioning=ultra_highlight_missing_conditioning,
            ultra_token_preview=ultra_token_preview,
            ultra_latent_anomaly_checks=ultra_latent_anomaly_checks,
            ultra_model_diff_tracking=ultra_model_diff_tracking,
            ultra_watch_expression=ultra_watch_expression,
            ultra_cache_artifacts=ultra_cache_artifacts,
            **dynamic_slots,
        )

        include_html = (display_mode or "both").lower() in {"html_only", "both"}
        verbosity_limit = CONSOLE_VERBOSITY_LEVELS.get(verbosity, 2)
        palette = CONSOLE_COLOR_SCHEMES.get(color_scheme, CONSOLE_COLOR_SCHEMES["cyberpunk"])

        sections: List[str] = []
        timestamp = datetime.utcnow().strftime("%H:%M:%S") if show_timestamps else ""
        for slot_name, display_name, slot_type in self.SLOT_DEFINITIONS:
            payload = dynamic_slots.get(slot_name)
            connected = payload is not None
            descriptor = self._summarise_object(payload) if connected else "<disconnected>"
            details = self._describe_payload(payload, slot_type) if connected else []
            if verbosity_limit >= 0:
                details = details[:verbosity_limit]
            escaped_title = html.escape(display_name)
            escaped_descriptor = html.escape(descriptor)
            detail_items = "".join(
                f"<li>{html.escape(detail)}</li>" for detail in details
            )
            timestamp_html = f"<span class=\"timestamp\">{html.escape(timestamp)}</span>" if timestamp else ""
            section = (
                f"<section class=\"slot\">"
                f"<header><span class=\"title\">{escaped_title}</span>{timestamp_html}</header>"
                f"<div class=\"descriptor\">{escaped_descriptor}</div>"
            )
            if detail_items:
                section += f"<ul class=\"details\">{detail_items}</ul>"
            section += "</section>"
            sections.append(section)

        html_report = ""
        if include_html:
            css = textwrap.dedent(
                f"""
                <style>
                    .h4-console {{
                        background: {palette['background']};
                        color: {palette['text']};
                        font-family: 'Segoe UI', 'Roboto', sans-serif;
                        padding: 16px;
                    }}
                    .h4-console .slot {{
                        background: {palette['panel']};
                        border-left: 4px solid {palette['accent']};
                        margin-bottom: 12px;
                        padding: 12px 16px;
                        border-radius: 8px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
                    }}
                    .h4-console .slot header {{
                        display: flex;
                        justify-content: space-between;
                        font-weight: 600;
                        letter-spacing: 0.05em;
                        margin-bottom: 6px;
                        color: {palette['accent']};
                    }}
                    .h4-console .slot .descriptor {{
                        font-size: 0.95rem;
                        margin-bottom: 4px;
                    }}
                    .h4-console .slot .details {{
                        list-style: disc;
                        margin: 0 0 0 20px;
                        color: {palette['muted']};
                    }}
                    .h4-console .timestamp {{
                        font-size: 0.8rem;
                        color: {palette['muted']};
                    }}
                </style>
                """
            ).strip()
            body = "".join(sections)
            html_report = f"{css}<div class=\"h4-console\">{body}</div>"

        return (html_report,) + routed_outputs


NODE_CLASS_MAPPINGS = {
    "h4AxisDriver": h4_AxisDriver,
    "h4PlotXY": h4_PlotXY,
    "h4Varianator": h4_Varianator,
    "h4DebugATron3000": h4_DebugATron3000,
    "h4DebugATronRouter": h4_DebugATronRouter,
    "h4DebugATron3000Console": h4_DebugATron3000Console,
    "h4ExecutionLogger": h4_ExecutionLogger,
    "h4SeedBroadcaster": h4_SeedBroadcaster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "h4AxisDriver": f"h4 : Axis Driver (v{AXIS_DRIVER_VERSION})",
    "h4PlotXY": f"h4 : The Engine (Simple Sampler+Plot) v{PLOT_NODE_VERSION}",
    "h4Varianator": f"h4 : The Varianator (v{VARIANATOR_VERSION})",
    "h4DebugATron3000": f"h4 : Debug-a-Tron-3000 (v{DEBUG_NODE_VERSION})",
    "h4DebugATronRouter": f"h4 : Debug-a-Tron-3000 Router (v{DEBUG_NODE_VERSION})",
    "h4DebugATron3000Console": "h4 : Debug-a-Tron-3000 CONSOLE (v3.0.0) - Now in Technicolor™",
    "h4ExecutionLogger": "h4 : Execution Logger (v1.0.0) - Console Mirror",
    "h4SeedBroadcaster": "h4 : Seed Broadcaster (v1.1.0) - Seed Generator",
}

NODE_TOOLTIP_MAPPINGS = {
    "h4AxisDriver": AXIS_DRIVER_WIDGET_TOOLTIPS,
    "h4PlotXY": PLOT_WIDGET_TOOLTIPS,
    "h4Varianator": VARIANATOR_WIDGET_TOOLTIPS,
    "h4DebugATron3000": DEBUG_WIDGET_TOOLTIPS,
    "h4DebugATronRouter": DEBUG_WIDGET_TOOLTIPS,
    "h4DebugATron3000Console": DEBUG_WIDGET_TOOLTIPS,
    "h4ExecutionLogger": EXECUTION_LOGGER_WIDGET_TOOLTIPS,
    "h4SeedBroadcaster": SEED_BROADCASTER_WIDGET_TOOLTIPS,
}

__all__ = [
    "h4_AxisDriver",
    "h4_PlotXY",
    "h4_Varianator",
    "h4_DebugATron3000",
    "h4_DebugATronRouter",
    "h4_DebugATron3000Console",
    "h4_ExecutionLogger",
    "h4_SeedBroadcaster",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NODE_TOOLTIP_MAPPINGS",
]

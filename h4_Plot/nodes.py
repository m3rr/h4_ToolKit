# -*- coding: utf-8 -*-
"""h4 Toolkit custom nodes for ComfyUI.

This module implements the Plot node (end-to-end sampler + grid generator with
progress previews) and the Debug-a-tron-3000 adaptive router/inspector. Both
nodes are instrumented with extremely verbose logging so that every action is
visible in the Stability Matrix / ComfyUI console during development.

The implementation leans on existing ComfyUI building blocks (samplers, VAE
utilities, CLIP encoders, etc.) to ensure forward compatibility while keeping
resource usage as low as practical. All helper utilities in this file are kept
lightweight and are re-used by both nodes to honour the "three node" policy.
"""

from __future__ import annotations

import functools
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from colorama import Fore, Style

try:  # pragma: no cover - optional preview module
    import latent_preview
except ImportError:  # pragma: no cover - preview is optional
    latent_preview = None

import comfy.model_management as model_management
import comfy.samplers as comfy_samplers
import comfy.sd as comfy_sd
import comfy.utils as comfy_utils
import folder_paths
import nodes as comfy_nodes

TOOLKIT_VERSION = "1.0.0"
PLOT_NODE_VERSION = "1.2.0"
DEBUG_NODE_VERSION = "2.0.0"

PLOT_WIDGET_TOOLTIPS: Dict[str, str] = {
    "seed": "Random generator seed. Use the same value to make fair comparisons across runs.",
    "steps": "Number of denoising steps. Higher values can add detail at the cost of runtime.",
    "cfg": "Classifier free guidance strength. Set to 1 to ignore the external negative prompt.",
    "width": "Output width in pixels. Stay close to the base model's native resolution to avoid artefacts.",
    "height": "Output height in pixels. Match width for square generations or adjust to taste.",
    "checkpoint": "Primary checkpoint to load when no axis entry overrides it.",
    "positive_prompt": "Text that describes what you want to see. Plain English encouraged.",
    "negative_prompt": "Terms that should be avoided. Ignored automatically when CFG is 1.",
    "x_axis_values": "One modifier per line. Prefix with checkpoint:, lora:, or leave plain text to extend the prompt.",
    "y_axis_values": "Same syntax as the X axis. The cartesian product builds your comparison grid.",
}

DEBUG_WIDGET_TOOLTIPS: Dict[str, str] = {
    "mode": "Monitor captures signals without forwarding. Passthrough logs and forwards everything.",
    "orientation": "Choose Horizontal to lay sockets left-to-right or Vertical to stack them top-to-bottom.",
}

# Verbosity flag lives at module level so we can quickly disable the firehose.
VERBOSE_LOGGING = True


class ToolkitLogger:
    """Structured, colourised logger for development-time visibility."""

    def __init__(self, component: str) -> None:
        self.component = component
        self._start_ts = time.time()

    def _emit(self, level: str, colour: str, message: str) -> None:
        if not VERBOSE_LOGGING:
            return
        elapsed = time.time() - self._start_ts
        prefix = f"[{elapsed:8.3f}s][{self.component}][{level}]"
        print(f"{colour}{prefix} {message}{Style.RESET_ALL}")

    def trace(self, message: str) -> None:
        self._emit("TRACE", Fore.MAGENTA, message)

    def info(self, message: str) -> None:
        self._emit("INFO", Fore.CYAN, message)

    def warn(self, message: str) -> None:
        self._emit("WARN", Fore.YELLOW, message)

    def error(self, message: str) -> None:
        self._emit("ERROR", Fore.RED, message)


GLOBAL_LOGGER = ToolkitLogger("h4_ToolKit")


@dataclass(frozen=True)
class AxisDescriptor:
    """Represents a user-specified modifier from the X/Y axis panels."""

    source_label: str
    kind: str  # checkpoint | lora | prompt | identity
    name: Optional[str] = None
    strength: float = 1.0
    prompt_suffix: str = ""

    def describe(self) -> str:
        if self.kind == "checkpoint" and self.name:
            return f"checkpoint={self.name}"
        if self.kind == "lora" and self.name:
            return f"lora={self.name}@{self.strength:.2f}"
        if self.kind == "prompt" and self.prompt_suffix:
            return f"prompt+={self.prompt_suffix}"
        return "identity"


@dataclass
class GenerationPlan:
    """Concrete execution plan for a single plot cell."""

    label: str
    checkpoint_name: Optional[str]
    prompt_suffix: str
    loras: List[Tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "checkpoint": self.checkpoint_name,
            "prompt_suffix": self.prompt_suffix,
            "loras": [
                {"name": name, "strength": strength} for name, strength in self.loras
            ],
        }


def _split_lines(value: str) -> List[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def parse_axis_entries(raw_text: str) -> List[AxisDescriptor]:
    """Parses axis definitions into typed descriptors."""

    entries: List[AxisDescriptor] = []
    if raw_text.strip():
        GLOBAL_LOGGER.trace(
            f"Parsing axis entries from text ({len(raw_text.splitlines())} lines)"
        )
    checkpoints = set(folder_paths.get_filename_list("checkpoints"))
    loras = set(folder_paths.get_filename_list("loras"))
    for raw_line in _split_lines(raw_text):
        lowered = raw_line.lower()
        if lowered in ("none", "base", "default"):
            entries.append(AxisDescriptor(raw_line, "identity"))
            continue
        if lowered.startswith("checkpoint:"):
            name = raw_line.split(":", 1)[1].strip()
            if not name:
                continue
            entries.append(AxisDescriptor(raw_line, "checkpoint", name=name))
            continue
        if lowered.startswith("lora:"):
            remainder = raw_line.split(":", 1)[1].strip()
            if "@" in remainder:
                name_part, strength_part = remainder.split("@", 1)
            elif "|" in remainder:
                name_part, strength_part = remainder.split("|", 1)
            else:
                name_part, strength_part = remainder, "1.0"
            name = name_part.strip()
            strength = float(strength_part.strip()) if strength_part.strip() else 1.0
            entries.append(AxisDescriptor(raw_line, "lora", name=name, strength=strength))
            continue
        if raw_line in checkpoints:
            entries.append(AxisDescriptor(raw_line, "checkpoint", name=raw_line))
            continue
        if raw_line in loras:
            entries.append(AxisDescriptor(raw_line, "lora", name=raw_line, strength=1.0))
            continue
        entries.append(AxisDescriptor(raw_line, "prompt", prompt_suffix=raw_line))
    return entries


def build_generation_matrix(
    base_checkpoint: Optional[str],
    x_descriptors: List[AxisDescriptor],
    y_descriptors: List[AxisDescriptor],
) -> List[GenerationPlan]:
    """Computes the cartesian product of axis descriptors into concrete plans."""

    def expand(axis_desc: List[AxisDescriptor]) -> List[List[AxisDescriptor]]:
        return [[desc] for desc in axis_desc] or [[AxisDescriptor("∅", "identity")]]

    runs: List[GenerationPlan] = []
    for x_bundle in expand(x_descriptors):
        for y_bundle in expand(y_descriptors):
            combined = x_bundle + y_bundle
            checkpoint_name = base_checkpoint
            suffixes: List[str] = []
            loras: List[Tuple[str, float]] = []
            label_parts: List[str] = []
            for descriptor in combined:
                label_parts.append(descriptor.describe())
                if descriptor.kind == "checkpoint" and descriptor.name:
                    checkpoint_name = descriptor.name
                elif descriptor.kind == "lora" and descriptor.name:
                    loras.append((descriptor.name, descriptor.strength))
                elif descriptor.kind == "prompt" and descriptor.prompt_suffix:
                    suffixes.append(descriptor.prompt_suffix)
            label = " | ".join([part for part in label_parts if part != "identity"]) or "base"
            plan = GenerationPlan(
                label=label,
                checkpoint_name=checkpoint_name,
                prompt_suffix="\n".join(suffixes) if suffixes else "",
                loras=loras,
            )
            runs.append(plan)
            GLOBAL_LOGGER.trace(
                f"Planned run: checkpoint={plan.checkpoint_name}, prompt+={plan.prompt_suffix!r}, loras={plan.loras}"
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


def compose_image_grid(images: List[torch.Tensor]) -> torch.Tensor:
    """Stack a list of decoded image tensors into a single grid image."""

    if not images:
        raise ValueError("compose_image_grid called with no images")
    batched: List[torch.Tensor] = []
    for tensor in images:
        if tensor.ndim == 4:
            for item in tensor:
                batched.append(item)
        else:
            batched.append(tensor)
    total = len(batched)
    rows, cols = auto_square_layout(total)
    GLOBAL_LOGGER.trace(
        f"Composing grid with {total} tiles -> layout {rows}x{cols}"
    )
    height = batched[0].shape[1]
    width = batched[0].shape[2]
    channels = batched[0].shape[0]
    grid = torch.zeros((channels, rows * height, cols * width), dtype=batched[0].dtype, device=batched[0].device)
    for idx, tile in enumerate(batched):
        r = idx // cols
        c = idx % cols
        grid[:, r * height : (r + 1) * height, c * width : (c + 1) * width] = tile
    return grid.unsqueeze(0)


def clone_latent(latent: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in latent.items()}


class PlotNode:
    """All-in-one checkpoint loader, sampler, scheduler, and grid plotter."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802 - ComfyUI naming convention
        checkpoint_list = folder_paths.get_filename_list("checkpoints")
        default_checkpoint = checkpoint_list[0] if checkpoint_list else ""
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 30.0, "step": 0.1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "checkpoint": (checkpoint_list, {"default": default_checkpoint}),
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
                "x_axis_values": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "y_axis_values": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "model_in": ("MODEL",),
                "clip_in": ("CLIP",),
                "vae_in": ("VAE",),
                "conditioning_in": ("CONDITIONING",),
                "latent_in": ("LATENT",),
                "image_in": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("grid_image", "last_latent", "execution_report")
    FUNCTION = "run_pipeline"
    CATEGORY = "h4 Toolkit/Generation"

    def __init__(self) -> None:
        self.logger = ToolkitLogger("PlotNode")
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

    def _load_checkpoint(self, checkpoint_name: str) -> Tuple[Any, Any, Any]:
        self.logger.trace(f"Loading checkpoint: {checkpoint_name}")
        ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
        model, clip, vae, _ = comfy_sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True
        )
        self.logger.info(f"Checkpoint loaded: {checkpoint_name}")
        return model, clip, vae

    def _encode_prompt(self, clip: Any, prompt: str) -> Any:
        self.logger.trace(f"Encoding prompt: {prompt[:60]}...")
        result = self.clip_encoder.encode(clip, prompt)
        return result[0] if isinstance(result, tuple) else result

    def _prepare_latent(
        self,
        vae: Any,
        width: int,
        height: int,
        latent_in: Optional[Dict[str, torch.Tensor]],
        image_in: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if latent_in is not None:
            self.logger.info("Using supplied latent as base")
            return clone_latent(latent_in)
        if image_in is not None:
            self.logger.info("Encoding supplied image into latent")
            if self.vae_encode is not None:
                latent = self.vae_encode.encode(vae, image_in)[0]
                return clone_latent(latent)
            if hasattr(vae, "encode"):
                try:
                    encoded = vae.encode(image_in)[0]
                except TypeError:
                    encoded = vae.encode(image_in)
                if isinstance(encoded, tuple):
                    encoded = encoded[0]
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
    ) -> Dict[str, torch.Tensor]:
        samples = latent["samples"]
        generator = torch.Generator(device=samples.device).manual_seed(int(seed))
        noise = torch.randn_like(samples, generator=generator)
        sampler_name = "dpmpp_2m_sde"
        scheduler = "karras"
        preview_token = f"plot-preview-{time.time():.0f}-{seed}"

        def callback(step: int, x0: torch.Tensor, *_args: Any) -> None:
            self.logger.trace(f"Preview step {step}/{steps}")
            if latent_preview is None:
                return
            try:
                decoded = latent_preview.decode_latent_preview(vae, x0)
                latent_preview.publish_preview(decoded, preview_token)
            except Exception as exc:  # pragma: no cover - best effort only
                self.logger.warn(f"Preview publishing failed: {exc}")

        self.logger.info(
            f"Sampling with {sampler_name} + {scheduler}, steps={steps}, cfg={cfg:.2f}, denoise={denoise:.2f}"
        )
        result = comfy_samplers.sample(
            model=model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
            seed=seed,
            denoise=denoise,
            disable_noise=False,
            start_step=0,
            last_step=-1,
            force_full_denoise=False,
            callback=callback,
        )
        return result

    def _decode(self, vae: Any, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = self.vae_decode.decode(vae, latent)[0]
        self.logger.trace("Decoded latent into image tensor")
        return images

    def _build_report(
        self,
        plans: List[GenerationPlan],
        results: List[Dict[str, Any]],
    ) -> str:
        payload = {
            "node": "PlotNode",
            "version": PLOT_NODE_VERSION,
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
        width: int,
        height: int,
        checkpoint: str,
        positive_prompt: str,
        negative_prompt: str,
        x_axis_values: str,
        y_axis_values: str,
        model_in: Optional[Any] = None,
        clip_in: Optional[Any] = None,
        vae_in: Optional[Any] = None,
        conditioning_in: Optional[Any] = None,
        latent_in: Optional[Dict[str, torch.Tensor]] = None,
        image_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        self.logger.info("Plot pipeline starting")
        model_cache: Dict[str, Tuple[Any, Any, Any]] = {}

        def get_model_bundle(name: Optional[str]) -> Tuple[Any, Any, Any]:
            if model_in is not None and name is None:
                return model_in, clip_in, vae_in
            lookup_name = name or checkpoint
            if lookup_name not in model_cache:
                model_cache[lookup_name] = self._load_checkpoint(lookup_name)
            return model_cache[lookup_name]

        plans = build_generation_matrix(
            base_checkpoint=checkpoint,
            x_descriptors=parse_axis_entries(x_axis_values),
            y_descriptors=parse_axis_entries(y_axis_values),
        )
        self.logger.info(f"Generated {len(plans)} execution plan(s)")

        base_model, base_clip, base_vae = get_model_bundle(None)
        latent_base = self._prepare_latent(base_vae, width, height, latent_in, image_in)
        negative = (
            conditioning_in
            if (conditioning_in is not None and cfg > 1.0)
            else (self._encode_prompt(base_clip, negative_prompt) if cfg > 1.0 else self._encode_prompt(base_clip, ""))
        )

        grid_images: List[torch.Tensor] = []
        last_latent: Dict[str, torch.Tensor] = latent_base
        run_summaries: List[Dict[str, Any]] = []

        for index, plan in enumerate(plans):
            self.logger.info(f"Executing plan {index + 1}/{len(plans)} :: {plan.label}")
            model_bundle = get_model_bundle(plan.checkpoint_name)
            active_model, active_clip, active_vae = model_bundle
            if conditioning_in is not None:
                positive = conditioning_in
            else:
                effective_prompt = positive_prompt
                if plan.prompt_suffix:
                    effective_prompt = f"{positive_prompt}\n{plan.prompt_suffix}".strip()
                positive = self._encode_prompt(active_clip, effective_prompt)
            patched_model, patched_clip = self._apply_loras(active_model, active_clip, plan.loras)
            latent_copy = clone_latent(latent_base)
            result_latent = self._run_sampler(
                model=patched_model,
                vae=active_vae,
                positive=positive,
                negative=negative,
                latent=latent_copy,
                seed=seed + index,
                steps=steps,
                cfg=cfg,
                denoise=1.0,
            )
            decoded = self._decode(active_vae, result_latent)
            grid_images.append(decoded)
            last_latent = result_latent
            run_summaries.append(
                {
                    "label": plan.label,
                    "seed": seed + index,
                    "steps": steps,
                    "cfg": cfg,
                    "loras": plan.loras,
                    "checkpoint": plan.checkpoint_name,
                }
            )
            if torch.cuda.is_available():  # pragma: no cover - device-specific
                torch.cuda.empty_cache()
        model_management.soft_empty_cache()

        grid = compose_image_grid(grid_images)
        report = self._build_report(plans, run_summaries)
        self.logger.info("Plot pipeline complete")
        return grid, last_latent, report


class DebugATron3000:
    """Adaptive router that mirrors signals per data type while logging everything."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:  # noqa: N802
        return {
            "required": {
                "mode": (
                    ["monitor", "passthrough"],
                    {"default": "monitor"},
                ),
                "orientation": (
                    ["horizontal", "vertical"],
                    {"default": "horizontal"},
                ),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "input_latent": ("LATENT",),
                "input_mask": ("MASK",),
                "input_conditioning": ("CONDITIONING",),
                "input_model": ("MODEL",),
                "input_clip": ("CLIP",),
                "input_vae": ("VAE",),
                "input_tensor": ("TENSOR",),
                "input_string": ("STRING",),
                "any_slot_0": ("*",),
                "any_slot_1": ("*",),
                "any_slot_2": ("*",),
                "any_slot_3": ("*",),
                "any_slot_4": ("*",),
                "any_slot_5": ("*",),
                "any_slot_6": ("*",),
                "any_slot_7": ("*",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "LATENT",
        "MASK",
        "CONDITIONING",
        "MODEL",
        "CLIP",
        "VAE",
        "TENSOR",
        "STRING",
        "*",
        "*",
        "*",
        "*",
        "*",
        "*",
        "*",
        "*",
        "STRING",
    )
    RETURN_NAMES = (
        "image_out",
        "latent_out",
        "mask_out",
        "conditioning_out",
        "model_out",
        "clip_out",
        "vae_out",
        "tensor_out",
        "string_out",
        "any_out_0",
        "any_out_1",
        "any_out_2",
        "any_out_3",
        "any_out_4",
        "any_out_5",
        "any_out_6",
        "any_out_7",
        "debug_log",
    )
    FUNCTION = "route"
    CATEGORY = "h4 Toolkit/Debug"

    def __init__(self) -> None:
        self.logger = ToolkitLogger("DebugATron3000")

    def _summarise_tensor(self, tensor: torch.Tensor) -> str:
        if tensor is None:
            return "<none>"
        return (
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )

    def _summarise_object(self, obj: Any) -> str:
        if obj is None:
            return "<none>"
        if torch.is_tensor(obj):
            return self._summarise_tensor(obj)
        return f"{type(obj).__name__}"

    def route(
        self,
        mode: str,
        orientation: str,
        input_image: Optional[torch.Tensor] = None,
        input_latent: Optional[Dict[str, torch.Tensor]] = None,
        input_mask: Optional[torch.Tensor] = None,
        input_conditioning: Optional[Any] = None,
        input_model: Optional[Any] = None,
        input_clip: Optional[Any] = None,
        input_vae: Optional[Any] = None,
        input_tensor: Optional[torch.Tensor] = None,
        input_string: Optional[str] = None,
        **dynamic_slots: Any,
    ) -> Tuple[Any, ...]:
        self.logger.info(
            f"Debug router engaged :: mode={mode}, orientation={orientation}"
        )
        log_entries = []
        payload_map = {
            "IMAGE": input_image,
            "LATENT": input_latent,
            "MASK": input_mask,
            "CONDITIONING": input_conditioning,
            "MODEL": input_model,
            "CLIP": input_clip,
            "VAE": input_vae,
            "TENSOR": input_tensor,
            "STRING": input_string,
        }
        for kind, value in payload_map.items():
            summary = self._summarise_object(value)
            log_entries.append(f"{kind}: {summary}")
            self.logger.trace(f"{kind} -> {summary}")
        for slot_name, slot_value in sorted(dynamic_slots.items()):
            descriptor = self._summarise_object(slot_value)
            log_entries.append(f"{slot_name}: {descriptor}")
            self.logger.trace(f"{slot_name} -> {descriptor}")
        debug_payload = json.dumps(
            {
                "node": "DebugATron3000",
                "version": DEBUG_NODE_VERSION,
                "mode": mode,
                "orientation": orientation,
                "signals": log_entries,
            },
            indent=2,
        )
        if mode == "monitor":
            passthroughs = (None,) * 9
            dynamic_passthroughs = (None,) * 8
        else:
            passthroughs = (
                input_image,
                input_latent,
                input_mask,
                input_conditioning,
                input_model,
                input_clip,
                input_vae,
                input_tensor,
                input_string,
            )
            dynamic_passthroughs = tuple(dynamic_slots.get(f"any_slot_{idx}") for idx in range(8))
        self.logger.info("Debug routing complete")
        return (*passthroughs, *dynamic_passthroughs, debug_payload)


NODE_CLASS_MAPPINGS = {
    "h4PlotNode": PlotNode,
    "h4DebugATron3000": DebugATron3000,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "h4PlotNode": f"h4 Plot (v{PLOT_NODE_VERSION})",
    "h4DebugATron3000": f"Debug-a-tron 3000 (v{DEBUG_NODE_VERSION})",
}

NODE_TOOLTIP_MAPPINGS = {
    "h4PlotNode": PLOT_WIDGET_TOOLTIPS,
    "h4DebugATron3000": DEBUG_WIDGET_TOOLTIPS,
}

__all__ = [
    "PlotNode",
    "DebugATron3000",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NODE_TOOLTIP_MAPPINGS",
]

# h4 Toolkit — Comprehensive Operations Manual

> Hyper-observable, cyberpunk-themed quality-of-life boosts for ComfyUI. This README is the definitive companion for installing, running, and mastering the **h4 Toolkit** nodes. Every control, feature, and internal safeguard is documented below.

---

## Table of Contents

1. [Project Snapshot](#project-snapshot)
2. [Installation & Environment](#installation--environment)
3. [Quick Start](#quick-start)
4. [Node Deep Dives](#node-deep-dives)
	1. [The Engine — `h4_PlotXY`](#the-engine--h4_plotxy)
	2. [Debug-a-Tron 3000 — `h4_DebugATron3000`](#debug-a-tron-3000--h4_debugatron3000)
	3. [Debug-a-Tron Router — `h4_DebugATronRouter`](#debug-a-tron-router--h4_debugatronrouter)
5. [Go Plus Ultra Diagnostics](#go-plus-ultra-diagnostics)
6. [UI Companion (`js/h4_ui.js`)](#ui-companion-jsh4_uijs)
7. [Operational Playbooks](#operational-playbooks)
8. [Troubleshooting & Support](#troubleshooting--support)
9. [Development Notes](#development-notes)
10. [Release Intel & Versioning](#release-intel--versioning)
11. [Appendix: Function Index](#appendix-function-index)
12. [License](#license)

---

## Project Snapshot

| Artifact | Value |
| --- | --- |
| **Toolkit Version** | `1.3.0` |
| **Engine (h4_PlotXY) Version** | `1.3.0` |
| **Debug Nodes Version** | `2.1.0` |
| **Supported Platform** | ComfyUI (Windows/Linux/macOS) |
| **Language Stack** | Python 3.10+ (nodes), JavaScript (UI overlay) |

### Key Pillars

- **The Engine (`h4_PlotXY`)** — production-grade sampler/grid orchestrator with plan matrices, LoRA layering, preview publishing, and labelled outputs.
- **Debug-a-Tron Suite** — real-time diagnostics nodes operating in monitor/passthrough modes, capable of deep latent inspection and branch-aware routing.
- **UI Enhancements** — dynamic sockets, contextual tooltips, orientation controls, and a polished Go Plus Ultra toggle experience.

---

## Installation & Environment

### Prerequisites

- ComfyUI checkout with write access to `custom_nodes/`
- Python 3.10 or 3.11 (mirrors ComfyUI baseline)
- Git (recommended) or manual download capabilities

### Dependency Footprint

`h4_Plot/__init__.py` auto-verifies and installs the following when missing:

- `colorama`
- `numpy`
- `torch`

> **Tip:** For air-gapped setups, install these manually into the ComfyUI environment first.

### Installation Steps

1. Navigate to the ComfyUI `custom_nodes` directory.
2. Clone or extract the toolkit:

	```powershell
	git clone https://github.com/m3rr/h4_ToolKit.git
	```

3. Restart ComfyUI. On launch, the toolkit prints a neon status banner enumerating loaded nodes.

### Optional Verification

Run a syntax gate to ensure your environment recognises the toolkit:

```powershell
python -m py_compile h4_Plot/nodes.py
```

Successful execution (exit code `0`) confirms the nodes file is syntax-tight for your interpreter.

---

## Quick Start

1. Drop **The Engine** node (`h4 : The Engine (Simple Sampler+Plot)`) into your ComfyUI graph.
2. Wire in prompts, sampler defaults, and (optionally) external latents or checkpoints.
3. Add **Debug-a-Tron 3000** downstream of any signal bundle you want to observe; flip `mode` to `monitor` during prototyping.
4. Kick off a run. Watch the ComfyUI console for colour-coded logs describing every stage.
5. Enable **GO PLUS ULTRA?!** when you need latent snapshots, anomaly detection, previews, JSON reporting, or model fingerprinting.
6. For dual-branch graphs (e.g., diffuser vs. refiner), insert the **Debug-a-Tron Router** to mirror and audit both branches simultaneously.

> **Workflow Pro-Tip:** Start with Debug-a-Tron disabled to validate your graph, then activate GO PLUS ULTRA for targeted forensic passes.

---

## Node Deep Dives

### The Engine — `h4_PlotXY`

**Purpose:** A full-stack sampler orchestrator that builds plan matrices, applies LoRAs, invokes samplers safely across ComfyUI versions, and outputs both per-image tensors and an annotated grid.

#### Inputs & Widgets

| Control | Description |
| --- | --- |
| `seed` | Base RNG seed. Axis plans can override per-cell values. |
| `steps` | Default sampler steps (integer). |
| `cfg` | Classifier-free guidance scale. |
| `denoise` | Denoise strength. Lower values adhere to input latent. |
| `clip_skip` | CLIP layer skip count; gracefully no-ops when the model lacks the hook. |
| `sampler_name` | Default sampler selection; axis presets can swap per cell. |
| `scheduler_name` | Noise schedule curve. |
| `width` / `height` | Dimensions for latent creation when no latent input is provided. |
| `checkpoint` | Baseline checkpoint to load. |
| `vae_name` / `clip_text_name` / `clip_vision_name` | Override modules with asset-specific replacements. |
| `positive_prompt` / `negative_prompt` | Base prompts; axis plans can append fragments via newline tokens. |
| `x_axis_mode` / `x_axis_values` | Row axis configuration (prompt, sampler, scheduler, CFG, steps, denoise, seed, LoRA, etc.). |
| `y_axis_mode` / `y_axis_values` | Column axis configuration, identical presets to the X axis. |

Optional inlets include `model`, `clip`, `clip_vision`, `vae`, generic/positive/negative conditioning payloads, `latent`, `image`, and `mask` tensors. When supplied, The Engine respects them; otherwise it provisions defaults.

#### Execution Pipeline

1. **Plan Construction** — Axis descriptors are parsed into `AxisDescriptor` dataclasses, each capturing checkpoint swaps, prompt suffixes, LoRAs, and overrides.
2. **Model Acquisition** — `get_model_bundle` fetches the checkpoint trio (UNet, CLIP, VAE). LoRAs are layered via `_apply_loras` with trace logging per injection.
3. **Latent Strategy** — `_prepare_latent` either clones incoming latents, encodes images, or fabricates an empty latent with the configured width/height.
4. **Sampler Invocation** — `_run_sampler` collaborates with `_invoke_sampler_with_compatibility` to adapt kwargs based on the active sampler signature. Sigma schedules are precomputed and reshaped to match device placement.
5. **Preview Publishing** — `_get_preview_handlers` picks the best decode/publish pair (preferring `latent_preview`, falling back to ComfyUI helpers or direct VAE decoding) and pushes JPEG snapshots mid-run.
6. **Grid Assembly** — `compose_image_grid` and `annotate_grid_image` stitch decoded tensors into a labelled panel. Annotations capture plan labels (e.g., `sampler: dpmpp_2m_sde | cfg: 6`).
7. **Output Packaging** — Returns a stacked tensor of individual images, the labelled grid, final model/clip/vae handles, the last latent, and positive/negative conditioning suitable for downstream reuse.

#### Unique Safeguards

- **Signature-Aware Shim:** `_invoke_sampler_with_compatibility` introspects the sampler function, renaming/dropping kwargs to match modern ComfyUI expectations. If a required parameter is still missing, it retries with raw kwargs so upstream errors carry context.
- **Sigma Generator Hardening:** All calls to `calculate_sigmas` are wrapped to supply only the keyword parameters it expects. Missing information results in logged warnings instead of hard crashes.
- **Clip-Skip Gracefulness:** `_set_clip_skip` hunts for the `set_clip_skip` hook across wrapped models, logging once if none is found.
- **Preview Fallbacks:** When `latent_preview` is unavailable, Engine falls back to direct VAE decoding. The toolkit warns once and continues without flooding logs.
- **GPU Cache Discipline:** If CUDA is detected, the Engine clears the cache after each plan to reduce VRAM spikes.

#### Operating The Engine

1. Configure baseline sampler settings (seed, steps, cfg, denoise).
2. Choose axis presets—example: `X Axis = sampler`, `Y Axis = cfg` with respective value lists.
3. Populate prompts or connect upstream conditioning.
4. Optionally connect a latent/image pair to start from a pre-existing source.
5. Execute the workflow. Inspect the console for per-plan logs and verify the annotated grid output.
6. Chain downstream to Debug-a-Tron nodes for post-run introspection.

---

### Debug-a-Tron 3000 — `h4_DebugATron3000`

**Purpose:** High-visibility inspector that mirrors signals while streaming structured logs. Designed for rapid troubleshooting of conditioning mismatches, latent anomalies, and model wiring.

#### Modes

- `monitor` — Outputs `None` for every slot but leaves logging/diagnostics active (perfect for non-invasive tapping).
- `passthrough` — Clones incoming payloads (with type-aware deep copies) and forwards them downstream while logging.

#### Slot Catalogue

| Slot | Type | Behaviour |
| --- | --- | --- |
| `model_in` | MODEL | Diffusion model handle; used for VAE fallback and fingerprinting. |
| `clip_in` | CLIP | Text encoder payload. |
| `clip_vision_in` | CLIP_VISION | Optional vision CLIP branch. |
| `vae_in` | VAE | Explicit VAE connection. If absent, node harvests VAE from the model payload. |
| `conditioning_in` | CONDITIONING | Generic list/dict of conditioning entries. |
| `conditioning_positive_in` | CONDITIONING | Positive branch; flagged when missing if diagnostics enabled. |
| `conditioning_negative_in` | CONDITIONING | Negative branch; flagged similarly. |
| `latent_in` | LATENT | Latent dict or tensor. Snapshots/anomaly checks focus here. |
| `image_in` | IMAGE | Image batch for logging. |
| `mask_in` | MASK | Optional mask tensor. |

#### Console & Telemetry

Each connected slot produces a multi-line summary:

- Object descriptor (type, label, file name).
- Tensor stats (shape, dtype, device, min/max/mean/std).
- Conditioning previews (first three entries, prompt snippets).
- Warnings when conditioning branches are absent or empty.

#### Go Plus Ultra Arsenal

All toggled via boolean/string widgets inside the node (see [Go Plus Ultra Diagnostics](#go-plus-ultra-diagnostics) for details):

- Latent snapshots (first/mid/last)
- Preview image generation + optional disk caching
- JSON session logs (`debugatron_ultra_*.json`)
- Conditioning gap callouts
- Token previews (first entries per branch)
- Latent anomaly detection (NaN/Inf/high std/extreme magnitude)
- Model fingerprinting (SHA-1 digest of state_dict metadata)
- Custom watch expressions evaluated against the slot context
- Artefact caching toggle to persist or discard preview outputs

#### Internal Architecture Highlights

- **Metadata-Driven Wiring:** Slot definitions, tooltips, fallbacks, and display names originate from class attributes, enabling subclass overrides without code duplication.
- **Context Snapshotting:** All inputs + GO PLUS ULTRA flags are captured into a dictionary for JSON logging and watch expression evaluation.
- **Safe Cloning:** `_clone_payload_safely` selects the correct deep-copy routine per slot type so passthrough mode never mutates inbound payloads.
- **Watch Expression Sandbox:** Evaluates user expressions with access to `torch`, `np`, and slot variables while blocking Python builtins for safety.

#### Usage Flow

1. Drop the node into your graph where visibility is needed.
2. Choose `monitor` during early prototyping to prevent downstream disruptions.
3. Flip GO PLUS ULTRA to `True` and enable desired diagnostics once the graph is stable.
4. Inspect console output and optional JSON/preview artefacts after each run.

---

### Debug-a-Tron Router — `h4_DebugATronRouter`

**Purpose:** Mirrors the Debug-a-Tron feature set for dual-stream pipelines, duplicating every slot with explicit branch suffixes (`_a`, `_b`). Ideal for diffuser/refiner splits, text/image dual tracks, or comparative conditioning paths.

#### Branch Slots

Each base inlet/outlet from the console node becomes two slots:

| Slot | Label | Notes |
| --- | --- | --- |
| `model_in_a` / `_b` | Branch A/B · Model | Allows independent model handles per branch. |
| `conditioning_positive_in_a` / `_b` | Branch A/B · Positive | Diagnostics track branch-specific conditioning health. |
| `latent_in_a` / `_b` | Branch A/B · Latent | Snapshots and anomaly checks annotate the originating branch. |
| … | … | All other slots follow the same pattern. |

#### Branch Intelligence

- **Per-Branch Summaries:** The router aggregates connection counts per branch and appends a human-readable report (e.g., `Branch A: 8/10 connected; missing -> Branch A · CLIP`).
- **Targeted VAE Fallbacks:** Each branch VAE inlet falls back to its corresponding model branch.
- **Passthrough-first Philosophy:** Default `mode` is `passthrough`, ensuring branch parity during live runs without extra configuration.
- **Branch-aware Diagnostics:** Latent snapshots, preview filenames, anomaly messages, and conditioning warnings all reference branch labels.

#### Operation

1. Insert the router where two lanes should be evaluated in parallel.
2. Leave `mode` in `passthrough` for branch mirroring, or switch to `monitor` if you only need telemetry.
3. Enable GO PLUS ULTRA to capture branch-tagged snapshots and JSON logs.
4. Consult console output for immediate branch health status.

---

## Go Plus Ultra Diagnostics

All Debug-a-Tron variants share the same control deck. Each toggle corresponds to a dedicated diagnostic routine inside `h4_DebugATron3000.route`.

| Control | Default | Effect |
| --- | --- | --- |
| `GO PLUS ULTRA?!` | `False` | Gatekeeper switch; when off, advanced widgets are physically removed from the UI and diagnostics are skipped. |
| `Snapshot: first step` | `True` | Capture the earliest latent tensor available. |
| `Snapshot: midpoint` | `False` | Capture the midpoint frame when history exists. |
| `Snapshot: final step` | `True` | Capture the most recent latent tensor. |
| `Generate preview image` | `True` | Decode snapshots into RGB previews whenever a VAE is connected. |
| `Write JSON session log` | `True` | Persist diagnostics to JSON (timestamped). |
| `Highlight missing conditioning` | `True` | Emit warnings when conditioning branches are absent/empty. |
| `Preview conditioning tokens` | `False` | Print first three token fragments per conditioning branch. |
| `Check for latent anomalies` | `True` | Scan tensors for NaN/Inf/extreme stats. |
| `Fingerprint attached models` | `False` | Hash model `state_dict` metadata for provenance tracking. |
| `Custom watch expression` | `""` | Python expression evaluated against slot context (`torch`/`np` available). |
| `Persist preview artifacts` | `True` | Keep preview PNG/JSON on disk (`False` deletes after logging). |

Artefacts land in the ComfyUI temp directory under `h4_debugatron_ultra`. Filenames include snapshot labels and branch suffixes.

---

## UI Companion (`js/h4_ui.js`)

The JavaScript extension auto-loads when the toolkit is present. Responsibilities include:

- **Widget Enhancement** — Applies descriptive placeholders/tooltips to Engine prompts, samplers, schedulers, etc.
- **Go Ultra UX** — Removes advanced widgets when GO PLUS ULTRA is disabled and restores them (with ordering intact) when re-enabled. Collapsed and expanded heights are preserved for consistent layout.
- **Orientation Toggle** — Adds a dropdown to Debug-a-Tron nodes allowing horizontal or vertical layout; recalculates size on the fly.
- **Dynamic Sockets** — Maintains eight wildcard sockets per node, auto-typing based on connected links and spawning fresh blanks when needed.
- **Event Hooking** — Intercepts `onNodeCreated` and `onConnectionsChange` to inject the above enhancements without modifying core ComfyUI files.

No configuration is required—simply ensure ComfyUI can read assets beneath `h4_Plot/js/`.

---

## Operational Playbooks

### Scenario A — Baseline Grid Generation

1. Configure Engine with prompts, sampler defaults, and axis presets (e.g., X axis = sampler, Y axis = CFG).
2. Leave Debug-a-Tron nodes disconnected for the initial smoke test.
3. Run once to verify outputs and labelled grid.
4. Introduce Debug-a-Tron 3000 post-run to inspect conditioning payloads; enable GO PLUS ULTRA for latent previews if discrepancies appear.

### Scenario B — Dual-Branch Diffusion

1. Duplicate your sampler branch (e.g., base diffuser vs. refiner) leading into separate pipelines.
2. Feed both into Debug-a-Tron Router, aligning branch A/B sockets accordingly.
3. Review branch summaries in the console; missing feeds are highlighted instantly.
4. Use JSON logs to archive branch-specific diagnostics alongside experiment metadata.

### Scenario C — Latent Forensics

1. Run Engine with GO PLUS ULTRA disabled to reproduce an anomaly.
2. Enable GO PLUS ULTRA on Debug-a-Tron 3000, toggle `Snapshot: final step`, `Generate preview image`, and `Check for latent anomalies`.
3. Re-run. Inspect console warnings and saved PNG/JSON artefacts in `h4_debugatron_ultra` to determine the fault.

---

## Troubleshooting & Support

| Symptom | Remedy |
| --- | --- |
| **Sampler rejects kwargs** | Ensure you are on the latest ComfyUI build; the compatibility shim adapts known signatures, but bleeding-edge changes may require toolkit updates. | 
| **Preview images missing** | Confirm a VAE is connected or that the model exposes a decodable VAE. If GO PLUS ULTRA caching is disabled, expect previews to be ephemeral. |
| **Conditioning warnings** | Check upstream graph wiring; the debugger flags empty conditioning lists to prevent silent CFG issues. |
| **Watch expression errors** | Expressions execute in a restricted environment; avoid referencing `__builtins__`. Console logs will echo the exception. |
| **UI widgets shift** | The UI script intentionally removes Go Ultra controls when disabled. Re-enable the toggle to restore them. |

If issues persist, gather console logs (the toolkit prints namespace-prefixed entries like `[h4 Toolkit::h4_DebugATron3000]`) and open a GitHub issue.

---

## Development Notes

- `_invoke_sampler_with_compatibility`, `_extract_model_sampling`, and `_ensure_ultra_directory` are prime extension points when aligning with future ComfyUI changes.
- Slot metadata is centralised; subclass overrides only need to redefine `SLOT_DEFINITIONS`, `SLOT_TOOLTIPS`, `VAE_MODEL_FALLBACKS`, and `BRANCH_DISPLAY_NAMES`.
- The router inherits `route` directly from the base class—branch behaviour is purely data-driven.
- JavaScript enhancements rely on standard ComfyUI extension hooks and do not monkey-patch core libraries.

### Local Validation Cycle

1. Modify Python modules.
2. Run `python -m py_compile h4_Plot/nodes.py`.
3. Optionally execute targeted ComfyUI workflows for runtime assurance.

---

## Release Intel & Versioning

- Update `TOOLKIT_VERSION`, `PLOT_NODE_VERSION`, and `DEBUG_NODE_VERSION` in `h4_Plot/nodes.py` when shipping releases.
- Document major changes inside `h4_Plot/docs/h4_toolkit_brief.html` (already acts as the canonical change log and architecture brief).
- Consider tagging Git releases aligned with version constants for traceability.

---

## Appendix: Function Index

| Module | Key Functions/Classes | Summary |
| --- | --- | --- |
| `h4_Plot/nodes.py` | `h4_PlotXY`, `h4_DebugATron3000`, `h4_DebugATronRouter`, `_invoke_sampler_with_compatibility`, `_collect_latent_snapshots`, `_tensor_anomalies`, etc. | Core logic for sampling, diagnostics, and metadata-driven routing. |
| `h4_Plot/js/h4_ui.js` | `toggleUltraWidgets`, `ensureDynamicSockets`, `applyOrientationLayout`, extension registration | UI/UX companion for nodes. |
| `h4_Plot/__init__.py` | `ensure_dependencies`, status banner, node exports | Dependency bootstrap and ComfyUI integration. |
| `h4_Plot/docs/h4_toolkit_brief.html` | HTML dossier | Narrative architecture and remediation history. |
import { app } from "/scripts/app.js";

const PLACEHOLDER_CLASS = "h4-placeholder";
const TOOLTIP_COPY = {
    positive_prompt: "Text that guides the sampler toward features you want in the image.",
    negative_prompt: "Terms to steer the sampler away from unwanted traits. Disabled when CFG=1.",
    x_axis_values: "Enter checkpoint:, lora:, or prompt fragments to test along the X axis.",
    y_axis_values: "Enter checkpoint:, lora:, or prompt fragments to test along the Y axis.",
    cfg: "Classifier-free guidance. 1 disables the negative prompt, higher values enforce it strongly.",
    steps: "How many denoising steps to run per image. More steps take longer but can improve detail.",
    width: "Output width in pixels. Leave at the base model resolution for best results.",
    height: "Output height in pixels. Leave at the base model resolution for best results.",
    seed: "Random seed. Use the same seed to compare variations consistently across the grid.",
    denoise: "Lower values hug the input image/latent, higher values lean on the prompt.",
    clip_skip: "How many CLIP layers to skip before conditioning. Higher values emphasise composition.",
    sampler_name: "Swap between ComfyUI samplers to see how each feels without rebuilding the graph.",
    scheduler_name: "Change the scheduler curve to test how noise levels progress over the steps.",
    vae_name: "Override the decoding VAE. Leave on <checkpoint> to reuse the checkpoint bundle.",
    clip_text_name: "Override the text CLIP model used for prompt encoding.",
    clip_vision_name: "Optional vision CLIP when using dual-CLIP checkpoints.",
    x_axis_mode: "Pick what the rows or columns change, like checkpoints, CFG, or samplers.",
    y_axis_mode: "Pick what the rows or columns change, like checkpoints, CFG, or samplers.",
};

const AXIS_PLACEHOLDER_SUFFIX = "Examples: checkpoint:xl/illustrious/h4_illustriousXL, lora:anime_style@0.8, cfg: 2.5, 4, 6";

const DYNAMIC_INPUT_PREFIX = "any_slot_";
const DYNAMIC_OUTPUT_PREFIX = "any_out_";
const DYNAMIC_SLOTS = 8;
const BASE_OUTPUT_COUNT = 8; // base routed payloads before dynamic slots
const ORIENTATION_HORIZONTAL = "horizontal";
const ULTRA_WIDGET_NAMES = [
    "ultra_capture_first_step",
    "ultra_capture_mid_step",
    "ultra_capture_last_step",
    "ultra_preview_images",
    "ultra_json_log",
    "ultra_highlight_missing_conditioning",
    "ultra_token_preview",
    "ultra_latent_anomaly_checks",
    "ultra_model_diff_tracking",
    "ultra_watch_expression",
    "ultra_cache_artifacts",
];

const ULTRA_WIDGET_LABELS = {
    ultra_capture_first_step: "Snapshot: first step",
    ultra_capture_mid_step: "Snapshot: midpoint",
    ultra_capture_last_step: "Snapshot: final step",
    ultra_preview_images: "Generate preview image",
    ultra_json_log: "Write JSON session log",
    ultra_highlight_missing_conditioning: "Highlight missing conditioning",
    ultra_token_preview: "Preview conditioning tokens",
    ultra_latent_anomaly_checks: "Check for latent anomalies",
    ultra_model_diff_tracking: "Fingerprint attached models",
    ultra_watch_expression: "Custom watch expression",
    ultra_cache_artifacts: "Persist preview artifacts",
};

const ULTRA_COLLAPSED_HEIGHT = 82;
const ULTRA_EXPANDED_MIN_HEIGHT = 184;
const ULTRA_MIN_WIDTH = 250;

const ensureUltraWidgetCache = (node) => {
    if (!node?.widgets || Array.isArray(node.h4UltraWidgetCache)) {
        return;
    }
    node.h4UltraWidgetCache = [];
    node.widgets.forEach((widget, index) => {
        if (ULTRA_WIDGET_NAMES.includes(widget.name)) {
            node.h4UltraWidgetCache.push({ widget, index });
        }
    });
    node.h4UltraBaseline = ULTRA_COLLAPSED_HEIGHT;
    node.h4UltraExpandedMin = ULTRA_EXPANDED_MIN_HEIGHT;
};

const applyUltraWidgetLabels = (node) => {
    if (!node?.widgets) {
        return;
    }
    for (const widget of node.widgets) {
        const label = ULTRA_WIDGET_LABELS[widget.name];
        if (label) {
            widget.label = label;
            if (widget.options) {
                widget.options.label = label;
            }
            if (widget.inputEl && typeof widget.inputEl.setAttribute === "function") {
                widget.inputEl.setAttribute("placeholder", label);
            }
        }
    }
};

const toggleUltraWidgets = (node, enabled) => {
    if (!node?.widgets) {
        return;
    }
    ensureUltraWidgetCache(node);
    let mutated = false;
    let minHeight = node.h4UltraBaseline ?? ULTRA_COLLAPSED_HEIGHT;
    if (!enabled) {
        const retained = [];
        for (const widget of node.widgets) {
            if (ULTRA_WIDGET_NAMES.includes(widget.name)) {
                mutated = true;
                if (!node.h4UltraWidgetCache.some((entry) => entry.widget === widget)) {
                    node.h4UltraWidgetCache.push({ widget, index: retained.length });
                }
            } else {
                retained.push(widget);
            }
        }
        if (mutated) {
            node.widgets = retained;
        }
    } else if (Array.isArray(node.h4UltraWidgetCache)) {
        minHeight = Math.max(node.h4UltraExpandedMin ?? ULTRA_EXPANDED_MIN_HEIGHT, ULTRA_EXPANDED_MIN_HEIGHT);
    const existing = new Set(node.widgets);
        const ordered = [...node.h4UltraWidgetCache].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
        for (const { widget } of ordered) {
            if (!existing.has(widget)) {
                widget.hidden = false;
                node.widgets.push(widget);
                existing.add(widget);
                mutated = true;
            }
        }
        applyUltraWidgetLabels(node);
    }
    if (mutated) {
        if (!Array.isArray(node.size)) {
            node.size = [ULTRA_MIN_WIDTH, minHeight];
        }
        if (enabled) {
            if (typeof node.computeSize === "function") {
                const size = node.computeSize();
                if (Array.isArray(size) && size.length === 2) {
                    node.size[0] = Math.max(size[0], ULTRA_MIN_WIDTH);
                    node.size[1] = Math.max(minHeight, size[1], ULTRA_EXPANDED_MIN_HEIGHT);
                    node.h4UltraExpandedMin = node.size[1];
                } else {
                    node.size[0] = Math.max(node.size[0], ULTRA_MIN_WIDTH);
                    node.size[1] = Math.max(node.size[1], minHeight, ULTRA_EXPANDED_MIN_HEIGHT);
                    node.h4UltraExpandedMin = node.size[1];
                }
            } else {
                node.size[0] = Math.max(node.size[0], ULTRA_MIN_WIDTH);
                node.size[1] = Math.max(node.size[1], minHeight, ULTRA_EXPANDED_MIN_HEIGHT);
                node.h4UltraExpandedMin = node.size[1];
            }
        } else {
            node.size[0] = Math.max(node.size[0], ULTRA_MIN_WIDTH);
            node.size[1] = ULTRA_COLLAPSED_HEIGHT;
            node.h4UltraBaseline = node.size[1];
        }
        node.setDirtyCanvas(true, true);
    }
};

const applyOrientationLayout = (node, orientation) => {
    node.h4Orientation = orientation;
    const isHorizontal = orientation === ORIENTATION_HORIZONTAL;
    node.flags = node.flags || {};
    node.flags.horizontal = isHorizontal;
    node.h4OrientationFlip = isHorizontal ? "row" : "column";
    if (typeof node.computeSize === "function") {
        const [w, h] = node.computeSize();
        if (Array.isArray(node.size)) {
            node.size[0] = w;
            node.size[1] = h;
        } else {
            node.size = [w, h];
        }
    }
    node.setDirtyCanvas(true, true);
};

const attachPlaceholderStyles = () => {
    if (document.querySelector("style[data-h4-toolkit]")) {
        return;
    }
    const styleTag = document.createElement("style");
    styleTag.dataset.h4Toolkit = "true";
    styleTag.textContent = `
        .${PLACEHOLDER_CLASS}::placeholder {
            color: #9aa0a6 !important;
            font-style: italic;
        }
    `;
    document.head.appendChild(styleTag);
};

const applyPlaceholder = (node, widgetName, placeholderText) => {
    const widget = node.widgets?.find((item) => item.name === widgetName);
    if (!widget || !widget.inputEl) {
        return;
    }
    widget.inputEl.placeholder = placeholderText;
    widget.inputEl.classList.add(PLACEHOLDER_CLASS);
    widget.inputEl.title = TOOLTIP_COPY[widgetName] ?? placeholderText;
    widget.desc = TOOLTIP_COPY[widgetName] ?? placeholderText;
};

const applySliderTooltip = (node, widgetName) => {
    const widget = node.widgets?.find((item) => item.name === widgetName);
    if (!widget) {
        return;
    }
    widget.desc = TOOLTIP_COPY[widgetName];
};

const AXIS_MODE_HINTS = {
    prompt: "Add prompt lines such as 'cinematic lighting'.",
    checkpoint: "checkpoint:xl/illustrious/h4_illustriousXL.safetensors",
    cfg: "Numbers like 2.5, 4.0, 6.5",
    steps: "Integers such as 20, 30, 40",
    sampler: "dpmpp_2m_sde, euler_a, ddim",
    scheduler: "karras, exponential, lognormal",
    denoise: "Values between 0.0 and 1.0",
    seed: "Seed numbers e.g. 12345",
    lora: "lora:anime_style@0.8",
    none: "Leave blank to keep the base settings.",
};

const updateAxisPlaceholder = (node, modeWidgetName, valuesWidgetName) => {
    const modeWidget = node.widgets?.find((item) => item.name === modeWidgetName);
    const valuesWidget = node.widgets?.find((item) => item.name === valuesWidgetName);
    if (!modeWidget || !valuesWidget || !valuesWidget.inputEl) {
        return;
    }
    const modeHint = AXIS_MODE_HINTS[modeWidget.value] ?? "Enter one value per line.";
    const detailedHint = `${modeHint} — ${AXIS_PLACEHOLDER_SUFFIX}`;
    valuesWidget.inputEl.placeholder = modeHint;
    valuesWidget.inputEl.title = detailedHint;
    valuesWidget.desc = detailedHint;
};

const ensureDynamicSockets = (node) => {
    if (!node.h4DynamicState) {
        node.h4DynamicState = {
            sockets: Array.from({ length: DYNAMIC_SLOTS }, () => ({ type: "*" })),
        };
    }
    for (let index = 0; index < DYNAMIC_SLOTS; index += 1) {
        updateDynamicSocketLabels(node, index, node.h4DynamicState.sockets[index].type);
    }
    ensureBlankSocket(node);
};

const updateDynamicSocketLabels = (node, index, typeName) => {
    const input = node.inputs?.find((item) => item.name === `${DYNAMIC_INPUT_PREFIX}${index}`);
    const output = node.outputs?.find((item) => item.name === `${DYNAMIC_OUTPUT_PREFIX}${index}`);
    const label = typeName && typeName !== "*" ? `${typeName} pass-through` : "(plug to arm)";
    if (input) {
        input.type = typeName || "*";
        input.label = label;
    }
    if (output) {
        output.type = typeName || "*";
        output.label = label;
    }
    if (node.h4DynamicState) {
        node.h4DynamicState.sockets[index].type = typeName || "*";
    }
};

const ensureBlankSocket = (node) => {
    if (!node.inputs) {
        return;
    }
    const hasBlank = node.inputs.some((item) => item.name?.startsWith(DYNAMIC_INPUT_PREFIX) && (!item.type || item.type === "*"));
    if (!hasBlank) {
        const lastIndex = node.inputs.filter((item) => item.name?.startsWith(DYNAMIC_INPUT_PREFIX)).length - 1;
        const nextIndex = Math.min(lastIndex + 1, DYNAMIC_SLOTS - 1);
        updateDynamicSocketLabels(node, nextIndex, "*");
    }
};

const handleDynamicConnectionChange = (node, slotIndex, isConnected, linkInfo, outputSocket) => {
    const input = node.inputs?.[slotIndex];
    if (!input || !input.name?.startsWith(DYNAMIC_INPUT_PREFIX)) {
        return;
    }
    const index = Number(input.name.replace(DYNAMIC_INPUT_PREFIX, ""));
    const candidateTypes = [linkInfo?.type, outputSocket?.type];
    const linkedType = isConnected ? (candidateTypes.find((value) => value && value !== "") || "*") : "*";
    updateDynamicSocketLabels(node, index, linkedType);
    ensureBlankSocket(node);
};

app.registerExtension({
    name: "h4.toolkit.ui",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData) {
            return;
        }
    if (nodeData.name === "h4PlotXY") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }
                attachPlaceholderStyles();
                applyPlaceholder(this, "positive_prompt", "POSITIVE PROMPT");
                applyPlaceholder(this, "negative_prompt", "NEGATIVE PROMPT");
                applyPlaceholder(this, "x_axis_values", "Axis modifiers (one per line)");
                applyPlaceholder(this, "y_axis_values", "Axis modifiers (one per line)");
                applySliderTooltip(this, "cfg");
                applySliderTooltip(this, "steps");
                applySliderTooltip(this, "denoise");
                applySliderTooltip(this, "width");
                applySliderTooltip(this, "height");
                applySliderTooltip(this, "seed");
                applySliderTooltip(this, "clip_skip");
                const vaeWidget = this.widgets?.find((item) => item.name === "vae_name");
                if (vaeWidget) {
                    vaeWidget.desc = TOOLTIP_COPY.vae_name;
                }
                const clipTextWidget = this.widgets?.find((item) => item.name === "clip_text_name");
                if (clipTextWidget) {
                    clipTextWidget.desc = TOOLTIP_COPY.clip_text_name;
                }
                const clipVisionWidget = this.widgets?.find((item) => item.name === "clip_vision_name");
                if (clipVisionWidget) {
                    clipVisionWidget.desc = TOOLTIP_COPY.clip_vision_name;
                }
                const samplerWidget = this.widgets?.find((item) => item.name === "sampler_name");
                if (samplerWidget) {
                    samplerWidget.desc = TOOLTIP_COPY.sampler_name;
                }
                const schedulerWidget = this.widgets?.find((item) => item.name === "scheduler_name");
                if (schedulerWidget) {
                    schedulerWidget.desc = TOOLTIP_COPY.scheduler_name;
                }
                updateAxisPlaceholder(this, "x_axis_mode", "x_axis_values");
                updateAxisPlaceholder(this, "y_axis_mode", "y_axis_values");
                const xModeWidget = this.widgets?.find((item) => item.name === "x_axis_mode");
                if (xModeWidget) {
                    const originalCallback = xModeWidget.callback;
                    xModeWidget.callback = (value) => {
                        if (originalCallback) {
                            originalCallback.call(this, value);
                        }
                        updateAxisPlaceholder(this, "x_axis_mode", "x_axis_values");
                    };
                }
                const yModeWidget = this.widgets?.find((item) => item.name === "y_axis_mode");
                if (yModeWidget) {
                    const originalYCallback = yModeWidget.callback;
                    yModeWidget.callback = (value) => {
                        if (originalYCallback) {
                            originalYCallback.call(this, value);
                        }
                        updateAxisPlaceholder(this, "y_axis_mode", "y_axis_values");
                    };
                }
            };
        }
        if (["h4DebugATron3000", "h4DebugATronRouter"].includes(nodeData.name)) {
            const baseOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (baseOnNodeCreated) {
                    baseOnNodeCreated.apply(this, arguments);
                }
                ensureDynamicSockets(this);
                ensureUltraWidgetCache(this);
            applyUltraWidgetLabels(this);
                const orientationWidget = this.widgets?.find((item) => item.name === "orientation");
                if (orientationWidget) {
                    const originalCallback = orientationWidget.callback;
                    orientationWidget.callback = (value) => {
                        if (originalCallback) {
                            originalCallback.call(this, value);
                        }
                        applyOrientationLayout(this, value);
                    };
                    applyOrientationLayout(this, orientationWidget.value ?? ORIENTATION_HORIZONTAL);
                }
                const modeWidget = this.widgets?.find((item) => item.name === "mode");
                if (modeWidget) {
                    modeWidget.desc = "Monitor taps the signal without forwarding it. Passthrough keeps the signal flowing.";
                }
                const goUltraWidget = this.widgets?.find((item) => item.name === "go_ultra");
                if (goUltraWidget) {
                    goUltraWidget.label = "GO PLUS ULTRA?!";
                    if (goUltraWidget.options) {
                        goUltraWidget.options.label = "GO PLUS ULTRA?!";
                    }
                    goUltraWidget.desc = "Delightfully dangerous diagnostics. Flip for full telemetry.";
                    const originalCallback = goUltraWidget.callback;
                    goUltraWidget.callback = (value) => {
                        if (originalCallback) {
                            originalCallback.call(this, value);
                        }
                        toggleUltraWidgets(this, Boolean(value));
                        applyUltraWidgetLabels(this);
                    };
                    toggleUltraWidgets(this, Boolean(goUltraWidget.value));
                    applyUltraWidgetLabels(this);
                }
            };

            const baseOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, link, input, output) {
                if (baseOnConnectionsChange) {
                    baseOnConnectionsChange.call(this, type, slotIndex, isConnected, link, input, output);
                }
                if (type === LiteGraph.INPUT) {
                    handleDynamicConnectionChange(this, slotIndex, isConnected, link || {}, output);
                }
            };
        }
    },
});

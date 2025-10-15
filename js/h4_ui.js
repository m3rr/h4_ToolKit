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

const DEFAULT_SAMPLERS = [
    "euler",
    "euler_ancestral",
    "lms",
    "heun",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_sde",
    "ddim",
    "ddpm",
];

const DEFAULT_SCHEDULERS = [
    "normal",
    "simple",
    "karras",
    "exponential",
    "sgm_uniform",
    "ddim_uniform",
];

const ASSET_ENDPOINT_BUILDERS = [
    // Standard ComfyUI endpoints
    (kind) => `/api/assets/${kind}`,
    (kind) => `/assets/${kind}`,
    (kind) => `/api/list?type=${kind}`,
    // ComfyUI object_info endpoints (most reliable)
    (kind) => {
        const typeMapping = {
            checkpoints: "CheckpointLoaderSimple",
            loras: "LoraLoader",
            vae: "VAELoader",
            clip: "CLIPLoader",
            clip_vision: "CLIPVisionLoader",
        };
        const nodeType = typeMapping[kind];
        return nodeType ? `/object_info/${nodeType}` : null;
    },
    // Alternative format
    (kind) => `/api/models/${kind}`,
];

const ASSET_KIND_MAPPING = {
    checkpoint: ["checkpoints"],
    lora: ["loras", "lycoris"],
    vae: ["vae"],
    clip: ["clip"],
    clip_vision: ["clip_vision"],
};

const assetCache = {};

const fetchAssetList = async (kind) => {
    if (assetCache[kind]) {
        return assetCache[kind];
    }
    for (const builder of ASSET_ENDPOINT_BUILDERS) {
        const endpoint = builder(kind);
        if (!endpoint) {
            continue;
        }
        try {
            const response = await fetch(endpoint, { cache: "no-store" });
            if (!response.ok) {
                continue;
            }
            const payload = await response.json();
            let names = [];
            
            // Handle different response formats
            if (Array.isArray(payload)) {
                names = payload;
            } else if (Array.isArray(payload?.items)) {
                names = payload.items.map((item) => item?.filename || item?.name || item).filter(Boolean);
            } else if (Array.isArray(payload?.files)) {
                names = payload.files.map((entry) => entry?.name || entry).filter(Boolean);
            } else if (payload && typeof payload === "object") {
                // Handle object_info format (ComfyUI native API)
                const firstKey = Object.keys(payload)[0];
                if (firstKey && payload[firstKey]?.input?.required) {
                    const inputKeys = Object.keys(payload[firstKey].input.required);
                    for (const key of inputKeys) {
                        const input = payload[firstKey].input.required[key];
                        if (Array.isArray(input) && Array.isArray(input[0])) {
                            names = input[0];
                            break;
                        }
                    }
                }
            }
            
            // Clean up names
            const unique = Array.from(new Set(names.map((name) => String(name)))).sort((a, b) => a.localeCompare(b));
            assetCache[kind] = unique;
            if (unique.length) {
                console.log(`h4 Axis Driver: Found ${unique.length} ${kind} from ${endpoint}`);
                return unique;
            }
        } catch (err) {
            // best-effort; continue to next endpoint
            console.debug(`h4 Axis Driver: Failed to fetch ${kind} from ${endpoint}`, err);
        }
    }
    console.warn(`h4 Axis Driver: No ${kind} found from any endpoint`);
    assetCache[kind] = [];
    return [];
};

const loadAssetKinds = async (kinds) => {
    const collected = [];
    for (const kind of kinds) {
        const entries = await fetchAssetList(kind);
        collected.push(...entries);
    }
    return Array.from(new Set(collected)).sort((a, b) => a.localeCompare(b));
};

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

const AXIS_DRIVER_PRESETS = [
    { value: "checkpoint", label: "Checkpoint" },
    { value: "prompt", label: "Prompt suffix" },
    { value: "lora", label: "LoRA" },
    { value: "sampler", label: "Sampler" },
    { value: "scheduler", label: "Scheduler" },
    { value: "steps", label: "Sampler steps" },
    { value: "cfg", label: "CFG" },
    { value: "denoise", label: "Denoise" },
    { value: "seed", label: "Seed" },
    { value: "none", label: "Disabled" },
];

const AXIS_DRIVER_SLOT_ORDER = ["X", "Y", "Z"];

const AXIS_DRIVER_DEFAULT_STYLE = {
    font_size: 22,
    font_family: "DejaVuSans",
    font_colour: "#FFFFFF",
    background: "black60",
    alignment: "center",
    label_position: "top_left",
    label_layout: "overlay",
    custom_label_x: "X",
    custom_label_y: "Y",
    custom_label_z: "Z",
    show_axis_headers: true,
};

const AXIS_DRIVER_DEFAULT_STATE = {
    axes: [
        { slot: "X", preset: "checkpoint", items: [] },
        { slot: "Y", preset: "prompt", items: [] },
        { slot: "Z", preset: "none", items: [] },
    ],
    style: { ...AXIS_DRIVER_DEFAULT_STYLE },
};

const AXIS_STYLE_ALIGNMENT_OPTIONS = [
    { value: "center", label: "Centre" },
    { value: "left", label: "Left" },
    { value: "right", label: "Right" },
];

const AXIS_STYLE_POSITION_OPTIONS = [
    { value: "top_left", label: "Top left" },
    { value: "top_right", label: "Top right" },
    { value: "bottom_left", label: "Bottom left" },
    { value: "bottom_right", label: "Bottom right" },
];

const AXIS_LABEL_LAYOUT_OPTIONS = [
    { value: "overlay", label: "Overlay (on images)" },
    { value: "border", label: "Border (outside images)" },
    { value: "none", label: "None (no labels)" },
];

const H4_NODE_NAMES = new Set([
    "h4PlotXY",
    "h4DebugATron3000",
    "h4DebugATronRouter",
    "h4DebugATron3000Console",
    "h4ExecutionLogger",
    "h4SeedBroadcaster",
    "h4AxisDriver",
]);

let h4NodeThemeEntry = null;

const ensureH4NodeThemeEntry = () => {
    const liteGraph = globalThis?.LiteGraph;
    const canvasProto = liteGraph?.LGraphCanvas;
    if (canvasProto?.node_colors) {
        if (!canvasProto.node_colors.h4ToolkitBlack) {
            const fallback = canvasProto.node_colors.default || {};
            canvasProto.node_colors.h4ToolkitBlack = {
                color: "#000000",
                bgcolor: fallback.bgcolor,
                groupcolor: fallback.groupcolor ?? fallback.bgcolor,
            };
        }
        h4NodeThemeEntry = canvasProto.node_colors.h4ToolkitBlack;
    }
    if (!h4NodeThemeEntry) {
        h4NodeThemeEntry = { color: "#000000" };
    }
    return h4NodeThemeEntry;
};

const applyH4NodeColor = (node) => {
    if (!node) {
        return;
    }
    const themeEntry = ensureH4NodeThemeEntry();
    node.color = themeEntry.color || "#000000";
    if (!node.bgcolor && themeEntry.bgcolor) {
        node.bgcolor = themeEntry.bgcolor;
    }
};

const AXIS_STYLE_BACKGROUND_OPTIONS = [
    { value: "black60", label: "Soft black" },
    { value: "none", label: "Transparent" },
    { value: "white40", label: "Soft white" },
];

const AXIS_DRIVER_MAX_ITEMS = 8;

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
        const isH4Node = H4_NODE_NAMES.has(nodeData.name);
        if (isH4Node) {
            const themeEntry = ensureH4NodeThemeEntry();
            if (themeEntry?.color) {
                nodeData.color = themeEntry.color;
            }
            if (!nodeData.bgcolor && themeEntry?.bgcolor) {
                nodeData.bgcolor = themeEntry.bgcolor;
            }
            nodeType.prototype.color = themeEntry?.color || "#000000";
            if (!nodeType.prototype.bgcolor && themeEntry?.bgcolor) {
                nodeType.prototype.bgcolor = themeEntry.bgcolor;
            }
        }
    if (nodeData.name === "h4PlotXY") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }
                applyH4NodeColor(this);
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
        if (nodeData.name === "h4AxisDriver") {
            const baseOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (baseOnNodeCreated) {
                    baseOnNodeCreated.apply(this, arguments);
                }
                applyH4NodeColor(this);
                const node = this;

                const applyTooltip = (element, text) => {
                    if (!element || !text) {
                        return;
                    }
                    element.title = text;
                    element.setAttribute("aria-label", text);
                };

                const axisOutputTips = {
                    axis_x: "Results while iterating X axis values (rows).",
                    axis_y: "Results while iterating Y axis values (columns).",
                    axis_z: "Results while iterating Z axis values (depth).",
                    legacy_summary: "Structured JSON summary of every axis combination rendered.",
                };

                if (Array.isArray(node.outputs)) {
                    node.outputs.forEach((output) => {
                        const tip = axisOutputTips[output?.name];
                        if (tip) {
                            output.desc = tip;
                        }
                    });
                }

                const installUi = () => {
                    if (!node?.widgets) {
                        return;
                    }
                    const configWidget = node.widgets.find((widget) => widget.name === "config");
                    if (!configWidget) {
                        return;
                    }
                    if (!configWidget.inputEl) {
                        requestAnimationFrame(installUi);
                        return;
                    }
                    if (node.h4AxisDriverUiAttached) {
                        return;
                    }
                    node.h4AxisDriverUiAttached = true;

                    try {
                        const hiddenHost = configWidget.inputEl;
                        const widgetWrapper =
                            hiddenHost.closest(".litegraph-widget") ||
                            hiddenHost.parentElement ||
                            configWidget.element ||
                            node?.html;
                        if (!widgetWrapper) {
                            console.warn("h4 Axis Driver: Unable to locate widget wrapper for config textarea");
                            return;
                        }

                        hiddenHost.style.display = "none";
                        hiddenHost.style.minHeight = "400px"; // Set reasonable height for JSON view
                        hiddenHost.style.fontFamily = "monospace";
                        hiddenHost.style.fontSize = "0.85rem";
                        widgetWrapper.style.width = "100%";
                        widgetWrapper.style.boxSizing = "border-box";

                        const rawWrapper = document.createElement("div");
                        rawWrapper.style.display = "none";
                        rawWrapper.style.marginTop = "6px";
                        rawWrapper.appendChild(hiddenHost);

                        const container = document.createElement("div");
                        container.className = "h4-axis-driver";
                        container.style.display = "flex";
                        container.style.flexDirection = "column";
                        container.style.gap = "12px";
                        container.style.marginTop = "6px";
                        container.style.padding = "8px";
                        container.style.width = "100%";
                        container.style.maxWidth = "100%";
                        container.style.boxSizing = "border-box";
                        container.style.overflowX = "hidden";
                        container.style.overflowY = "visible";
                        container.style.border = "1px solid rgba(255,255,255,0.08)";
                        container.style.borderRadius = "8px";
                        container.style.background = "rgba(0, 0, 0, 0.18)";

                        widgetWrapper.appendChild(container);
                        widgetWrapper.appendChild(rawWrapper);

                        const toolbar = document.createElement("div");
                        toolbar.style.display = "flex";
                        toolbar.style.gap = "8px";
                        toolbar.style.alignItems = "center";
                        toolbar.style.justifyContent = "space-between";

                        const axisStack = document.createElement("div");
                        axisStack.style.display = "flex";
                        axisStack.style.flexDirection = "column";
                        axisStack.style.gap = "10px";
                        axisStack.style.width = "100%";
                        axisStack.style.minWidth = "0";
                        axisStack.style.overflowX = "hidden";
                        axisStack.style.overflowY = "visible";

                        const originalComputeSize = typeof node.computeSize === "function" ? node.computeSize.bind(node) : null;
                        const originalSetSize = typeof node.setSize === "function" ? node.setSize.bind(node) : null;
                        const originalOnRemoved = typeof node.onRemoved === "function" ? node.onRemoved.bind(node) : null;
                        const originalOnResize = typeof node.onResize === "function" ? node.onResize.bind(node) : null;

                        node.resizable = true;
                        node.h4AxisDriverManualSize = node.h4AxisDriverManualSize || null;
                        if (!Number.isFinite(node.h4AxisDriverBaseScale) || node.h4AxisDriverBaseScale <= 0) {
                            node.h4AxisDriverBaseScale = resolveScale() || 1;
                        }
                        const DEFAULT_NODE_WIDTH = 560;
                        const DEFAULT_NODE_HEIGHT = 580;
                        node.h4AxisDriverDefaultSize = node.h4AxisDriverDefaultSize || [DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT];
                        if (!Array.isArray(node.size)) {
                            node.size = [DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT];
                        } else {
                            node.size[0] = Math.max(node.size[0] ?? 0, DEFAULT_NODE_WIDTH);
                            node.size[1] = Math.max(node.size[1] ?? 0, DEFAULT_NODE_HEIGHT);
                        }
                        node.h4AxisDriverUiReady = false;
                        let autoSizing = false;
                        let adjusting = false;
                        let lastManualResizeAt = 0;
                        const manualResizeHoldMs = 850;
                        let lastContentHash = "";
                        const MANUAL_MAX_WIDTH = 1400;
                        const MANUAL_MAX_HEIGHT = 1400;

                        const readManualSize = () => {
                            const record = node.h4AxisDriverManualSize;
                            if (!record) {
                                return { width: null, height: null };
                            }
                            if (Array.isArray(record)) {
                                const width = Number.isFinite(record[0]) ? record[0] : null;
                                const height = Number.isFinite(record[1]) ? record[1] : null;
                                node.h4AxisDriverManualSize = {
                                    width: width !== null ? Math.max(0, Math.min(width, MANUAL_MAX_WIDTH)) : null,
                                    height: height !== null ? Math.max(0, Math.min(height, MANUAL_MAX_HEIGHT)) : null,
                                };
                                return node.h4AxisDriverManualSize;
                            }
                            if (typeof record === "object") {
                                const width = Number.isFinite(record.width) ? record.width : null;
                                const height = Number.isFinite(record.height) ? record.height : null;
                                return {
                                    width: width !== null ? Math.max(0, Math.min(width, MANUAL_MAX_WIDTH)) : null,
                                    height: height !== null ? Math.max(0, Math.min(height, MANUAL_MAX_HEIGHT)) : null,
                                };
                            }
                            return { width: null, height: null };
                        };

                        const storeManualSize = (width, height) => {
                            const widthGraph = Number.isFinite(width) ? Math.max(0, Math.min(width, MANUAL_MAX_WIDTH)) : null;
                            const heightGraph = Number.isFinite(height) ? Math.max(0, Math.min(height, MANUAL_MAX_HEIGHT)) : null;
                            node.h4AxisDriverManualSize = {
                                width: widthGraph,
                                height: heightGraph,
                            };
                        };

                        node.onResize = function (width, height) {
                            // Just call the original, don't do any custom handling
                            if (originalOnResize) {
                                try {
                                    originalOnResize.apply(this, arguments);
                                } catch (_err) {
                                    /* ignore */
                                }
                            }
                        };

                        const ensureCleanupList = () => {
                            if (!Array.isArray(node.h4AxisDriverCleanup)) {
                                node.h4AxisDriverCleanup = [];
                            }
                            return node.h4AxisDriverCleanup;
                        };

                        const cleanup = ensureCleanupList();

                        const DEFAULT_MIN_WIDTH = 280;
                        const DEFAULT_MIN_HEIGHT = 220;
                        const DEFAULT_BASE_WIDTH = node.h4AxisDriverDefaultSize?.[0] ?? DEFAULT_NODE_WIDTH;
                        const DEFAULT_BASE_HEIGHT = node.h4AxisDriverDefaultSize?.[1] ?? DEFAULT_NODE_HEIGHT;
                        const AUTO_MIN_WIDTH = DEFAULT_BASE_WIDTH;
                        const AUTO_MIN_HEIGHT = DEFAULT_BASE_HEIGHT;
                        let loggedOriginalComputeError = false;

                        const normaliseSizeTuple = (maybeTuple, fallbackWidth = DEFAULT_MIN_WIDTH, fallbackHeight = DEFAULT_MIN_HEIGHT) => {
                            const width = Number.isFinite(maybeTuple?.[0]) ? maybeTuple[0] : fallbackWidth;
                            const height = Number.isFinite(maybeTuple?.[1]) ? maybeTuple[1] : fallbackHeight;
                            return [width, height];
                        };

                        const callOriginalComputeSize = (minWidth = DEFAULT_MIN_WIDTH, minHeight = DEFAULT_MIN_HEIGHT, outArray) => {
                            const widthFloor = Number.isFinite(minWidth) ? minWidth : DEFAULT_MIN_WIDTH;
                            const heightFloor = Number.isFinite(minHeight) ? minHeight : DEFAULT_MIN_HEIGHT;
                            if (!originalComputeSize) {
                                const base = Array.isArray(outArray) ? outArray : [widthFloor, heightFloor];
                                base[0] = widthFloor;
                                base[1] = heightFloor;
                                return base;
                            }
                            const target = Array.isArray(outArray) ? outArray : [widthFloor, heightFloor];
                            if (!Array.isArray(outArray)) {
                                target[0] = widthFloor;
                                target[1] = heightFloor;
                            }
                            try {
                                const maybeResult = originalComputeSize.call(node, target);
                                if (Array.isArray(maybeResult)) {
                                    return maybeResult;
                                }
                                if (typeof maybeResult === "number") {
                                    target[0] = Number.isFinite(maybeResult) ? maybeResult : widthFloor;
                                    target[1] = Number.isFinite(target[1]) ? target[1] : heightFloor;
                                    return target;
                                }
                                return Array.isArray(target) ? target : [widthFloor, heightFloor];
                            } catch (err) {
                                if (!loggedOriginalComputeError) {
                                    loggedOriginalComputeError = true;
                                    console.warn("h4 Axis Driver: original computeSize call failed", err);
                                }
                                target[0] = Number.isFinite(target[0]) ? target[0] : widthFloor;
                                target[1] = Number.isFinite(target[1]) ? target[1] : heightFloor;
                                return target;
                            }
                        };

                        function resolveScale() {
                            const graphCanvas = node.graph?.canvas || app?.canvas;
                            const raw = graphCanvas?.ds?.scale ?? graphCanvas?.scale ?? 1;
                            const parsed = Number(raw);
                            return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
                        }

                        const measurePixels = () => {
                            const containerHeight = container.scrollHeight || container.offsetHeight || 0;
                            const containerWidth = container.scrollWidth || container.offsetWidth || 0;
                            const rawHeight = rawWrapper.style.display !== "none"
                                ? (rawWrapper.scrollHeight || rawWrapper.offsetHeight || 0)
                                : 0;
                            const totalHeight = containerHeight + rawHeight + 12;
                            const totalWidth = containerWidth + 24;
                            return {
                                widthPx: Math.max(totalWidth, 280),
                                heightPx: Math.max(totalHeight, 220),
                                containerHeightPx: containerHeight,
                                rawHeightPx: rawHeight,
                            };
                        };

                        const convertPixelsToGraphUnits = (widthPx, heightPx) => {
                            const baseScale = node.h4AxisDriverBaseScale || 1;
                            const widthGraph = Math.max(DEFAULT_MIN_WIDTH, Math.round(widthPx / baseScale));
                            const heightGraph = Math.max(DEFAULT_MIN_HEIGHT, Math.round(heightPx / baseScale));
                            return [widthGraph, heightGraph];
                        };

                        const applyNodeSize = (width, height) => {
                            const attempts = [];
                            if (originalSetSize) {
                                attempts.push(() => originalSetSize([width, height]));
                                attempts.push(() => originalSetSize(width, height));
                            }
                            if (typeof node.setSize === "function") {
                                attempts.push(() => node.setSize([width, height]));
                                attempts.push(() => node.setSize(width, height));
                            }
                            for (const attempt of attempts) {
                                try {
                                    const result = attempt();
                                    if (result !== false && result !== undefined) {
                                        break;
                                    }
                                } catch (_err) {
                                    /* swallow sizing errors and fall through */
                                }
                            }
                            node.size = Array.isArray(node.size) ? node.size : [width, height];
                            node.size[0] = width;
                            node.size[1] = height;
                            autoSizing = true;
                            try {
                                if (typeof node.onResize === "function") {
                                    node.onResize(width, height);
                                }
                            } finally {
                                autoSizing = false;
                            }
                            node.h4AxisDriverLastSize = [width, height];
                        };

                        const updateWidgetMetrics = (pixelHeight, pixelWidth, graphWidthUnits, graphHeightUnits) => {
                            const scale = resolveScale();
                            const widgetGraphHeight = Math.max(graphHeightUnits, Math.ceil(pixelHeight / (scale || 1)));
                            const widgetGraphWidth = Math.max(graphWidthUnits, Math.ceil(pixelWidth / (scale || 1)));
                            configWidget.height = widgetGraphHeight;
                            configWidget.computeSize = () => [widgetGraphWidth, widgetGraphHeight];
                            node.widgets_height = widgetGraphHeight;
                            if (typeof node.widgets_start_y !== "number") {
                                node.widgets_start_y = node.h4AxisDriverOriginalWidgetStart ?? 60;
                            }
                            if (widgetWrapper) {
                                const boundedHeight = Math.max(pixelHeight, 0);
                                widgetWrapper.style.height = "auto";
                                widgetWrapper.style.minHeight = `${boundedHeight}px`;
                                widgetWrapper.style.maxHeight = "none";
                                widgetWrapper.style.overflow = "visible";
                            }
                        };

                        const scheduleList = [];
                        let pendingAnimation = null;
                        cleanup.push(() => {
                            if (pendingAnimation) {
                                cancelAnimationFrame(pendingAnimation);
                                pendingAnimation = null;
                            }
                            scheduleList.forEach((handle) => clearTimeout(handle));
                            scheduleList.length = 0;
                        });

                        const manualResizeActive = () => lastManualResizeAt && Date.now() - lastManualResizeAt < manualResizeHoldMs;

                        const performAdjust = () => {
                            // Disabled - auto-sizing causes infinite growth
                            // Node size is now fixed at initialization or manual resize only
                        };

                        const queueAdjust = () => {
                            if (pendingAnimation) {
                                cancelAnimationFrame(pendingAnimation);
                            }
                            pendingAnimation = requestAnimationFrame(() => {
                                pendingAnimation = null;
                                performAdjust();
                            });
                            scheduleList.forEach((id) => clearTimeout(id));
                            scheduleList.length = 0;
                            const fallbackDelays = [16, 48, 120, 240, 520];
                            fallbackDelays.forEach((delay) => {
                                const handle = setTimeout(performAdjust, delay);
                                scheduleList.push(handle);
                            });
                        };

                        node.h4AxisDriverAutoSize = {
                            compute(minWidth = 280, minHeight = 220) {
                                // Disabled - return the hints as-is without any measurement
                                const widthFloor = Number.isFinite(minWidth) && minWidth > 0 ? minWidth : DEFAULT_MIN_WIDTH;
                                const heightFloor = Number.isFinite(minHeight) && minHeight > 0 ? minHeight : DEFAULT_MIN_HEIGHT;
                                return [widthFloor, heightFloor];
                            },
                            adjust: queueAdjust,
                        };

                        node.computeSize = function (arg1, arg2) {
                            // Just pass through to original - no custom computation
                            if (originalComputeSize) {
                                return originalComputeSize.apply(this, arguments);
                            }
                            // Fallback to simple return
                            if (Array.isArray(arg1)) {
                                return arg1;
                            }
                            return [
                                Number.isFinite(arg1) ? arg1 : DEFAULT_MIN_WIDTH,
                                Number.isFinite(arg2) ? arg2 : DEFAULT_MIN_HEIGHT
                            ];
                        };

                        const adjustNodeSize = (force = false) => {
                            // Auto-adjust disabled - node stays at fixed size
                        };

                        node.h4AxisDriverAdjustSize = adjustNodeSize;
                        node.h4AxisDriverOriginalWidgetStart = node.h4AxisDriverOriginalWidgetStart ?? node.widgets_start_y;

                        const setupObservers = () => {
                            // Observers disabled - they were causing infinite growth loops
                        };

                        setupObservers();

                        node.onRemoved = function () {
                            cleanup.forEach((fn) => {
                                try {
                                    fn();
                                } catch (_err) {
                                    /* ignore */
                                }
                            });
                            cleanup.length = 0;
                            if (originalOnRemoved) {
                                originalOnRemoved.apply(this, arguments);
                            }
                        };

                        const toolbarLeft = document.createElement("div");
                        toolbarLeft.style.display = "flex";
                        toolbarLeft.style.gap = "8px";
                        toolbarLeft.style.alignItems = "center";

                        const toggleJsonBtn = document.createElement("button");
                        toggleJsonBtn.textContent = "Show JSON";
                        toggleJsonBtn.style.padding = "4px 8px";
                        toggleJsonBtn.style.fontSize = "0.8rem";
                        applyTooltip(toggleJsonBtn, "Display the raw Axis Driver configuration JSON for quick edits.");
                        toggleJsonBtn.onclick = () => {
                            // Force an immediate save before showing JSON to ensure latest state is visible
                            if (commitDebounceTimer) {
                                clearTimeout(commitDebounceTimer);
                                commitDebounceTimer = null;
                            }
                            commitState(false, true); // Immediate save without rerender
                            
                            const showing = rawWrapper.style.display !== "none";
                            rawWrapper.style.display = showing ? "none" : "block";
                            hiddenHost.style.display = showing ? "none" : "block";
                            if (widgetWrapper) {
                                widgetWrapper.style.maxHeight = "";
                            }
                            toggleJsonBtn.textContent = showing ? "Show JSON" : "Hide JSON";
                            node.h4AxisDriverManualSize = null;
                            adjustNodeSize(true);
                        };

                        toolbarLeft.appendChild(toggleJsonBtn);

                        const renderStatus = document.createElement("span");
                        renderStatus.style.fontSize = "0.7rem";
                        renderStatus.style.opacity = "0.66";
                        renderStatus.style.fontFamily = "monospace";
                        renderStatus.textContent = "axes:?";
                        applyTooltip(renderStatus, "Total active axes in this node (X, Y, and Z). All should read 3 when enabled.");

                        toolbar.appendChild(toolbarLeft);
                        toolbar.appendChild(renderStatus);
                        container.appendChild(toolbar);
                        container.appendChild(axisStack);

                        const stylePanel = document.createElement("details");
                        stylePanel.style.border = "1px solid rgba(255,255,255,0.08)";
                        stylePanel.style.borderRadius = "6px";
                        stylePanel.style.padding = "6px 8px";
                        stylePanel.style.background = "rgba(0,0,0,0.12)";
                        const styleSummary = document.createElement("summary");
                        styleSummary.textContent = "Axis styling";
                        styleSummary.style.cursor = "pointer";
                        styleSummary.style.fontWeight = "600";
                        applyTooltip(styleSummary, "Tweak how axis labels render in the generated grid image.");
                        stylePanel.appendChild(styleSummary);
                        const styleBody = document.createElement("div");
                        styleBody.style.display = "grid";
                        styleBody.style.gridTemplateColumns = "repeat(auto-fit, minmax(160px, 1fr))";
                        styleBody.style.gap = "8px";
                        styleBody.style.marginTop = "8px";
                        stylePanel.appendChild(styleBody);
                        stylePanel.addEventListener("toggle", () => {
                            node.h4AxisDriverManualSize = null;
                            adjustNodeSize(true);
                        });
                        container.appendChild(stylePanel);

                        const cloneDefault = (value) => JSON.parse(JSON.stringify(value));

                        const parseState = (rawText) => {
                            if (typeof rawText !== "string" || !rawText.trim()) {
                                return cloneDefault(AXIS_DRIVER_DEFAULT_STATE);
                            }
                            try {
                                const parsed = JSON.parse(rawText);
                                if (parsed && typeof parsed === "object") {
                                    return parsed;
                                }
                            } catch (_err) {
                                /* ignore parse errors */
                            }
                            return cloneDefault(AXIS_DRIVER_DEFAULT_STATE);
                        };

                const normaliseItem = (preset, rawItem) => {
                    const safe = rawItem && typeof rawItem === "object" ? rawItem : {};
                    let value = safe.value ?? safe.raw_value ?? "";
                    const label = typeof safe.label === "string" ? safe.label : typeof safe.source_label === "string" ? safe.source_label : "";
                    const overrides = safe.overrides && typeof safe.overrides === "object" && !Array.isArray(safe.overrides)
                        ? { ...safe.overrides }
                        : {};
                    let strength = safe.strength;
                    if (strength === undefined && typeof overrides.lora_strength === "number") {
                        strength = overrides.lora_strength;
                    }
                    if (["steps", "seed"].includes(preset)) {
                        const numeric = Number.parseInt(value, 10);
                        value = Number.isFinite(numeric) ? numeric : "";
                    } else if (["cfg", "denoise"].includes(preset)) {
                        const numeric = Number.parseFloat(value);
                        value = Number.isFinite(numeric) ? numeric : "";
                    } else if (typeof value !== "string" && value !== null && value !== undefined) {
                        value = String(value);
                    }
                    if (preset === "lora") {
                        const numericStrength = Number.parseFloat(strength);
                        const finalStrength = Number.isFinite(numericStrength) ? numericStrength : 0.75;
                        const valueText = typeof value === "string" ? value : String(value ?? "");
                        return {
                            label,
                            value: valueText,
                            strength: finalStrength,
                            overrides,
                        };
                    }
                    return {
                        label,
                        value,
                        strength: null,
                        overrides,
                    };
                };

                const normaliseState = (raw) => {
                    const base = raw && typeof raw === "object" ? raw : {};
                    const style = {
                        ...AXIS_DRIVER_DEFAULT_STYLE,
                        ...(base.style && typeof base.style === "object" ? base.style : {}),
                    };
                    const axisMap = {};
                    if (Array.isArray(base.axes)) {
                        for (const entry of base.axes) {
                            const slot = String(entry?.slot || "").toUpperCase();
                            if (!slot) {
                                continue;
                            }
                            axisMap[slot] = entry;
                        }
                    }
                    const axes = AXIS_DRIVER_SLOT_ORDER.map((slot, index) => {
                        const fallbackPreset = index === 0 ? "checkpoint" : index === 1 ? "prompt" : "none";
                        const source = axisMap[slot] && typeof axisMap[slot] === "object" ? axisMap[slot] : {};
                        const presetRaw = typeof source.preset === "string" ? source.preset.toLowerCase() : fallbackPreset;
                        const preset = AXIS_DRIVER_PRESETS.some((p) => p.value === presetRaw) ? presetRaw : fallbackPreset;
                        const rawItems = Array.isArray(source.items) ? source.items : [];
                        const items = rawItems.slice(0, AXIS_DRIVER_MAX_ITEMS).map((item) => normaliseItem(preset, item));
                        return {
                            slot,
                            preset,
                            items,
                        };
                    });
                    return { axes, style };
                };

                let state = normaliseState(parseState(configWidget.value ?? configWidget.inputEl.value ?? ""));
                let suppressCallback = false;
                let commitDebounceTimer = null;

                const ensureStateStructure = () => {
                    const fallback = cloneDefault(AXIS_DRIVER_DEFAULT_STATE);
                    if (!Array.isArray(state.axes)) {
                        state.axes = fallback.axes.map((axis) => ({ ...axis }));
                    } else {
                        // CRITICAL FIX: Don't recreate existing axes! Just ensure all slots exist
                        const axisLookup = new Map(state.axes.map((axis) => [String(axis?.slot || "").toUpperCase(), axis]));
                        
                        // Only add missing slots, preserve existing axes by reference
                        for (const slot of AXIS_DRIVER_SLOT_ORDER) {
                            if (!axisLookup.has(slot.toUpperCase())) {
                                const newAxis = { slot, preset: "none", items: [] };
                                state.axes.push(newAxis);
                                axisLookup.set(slot.toUpperCase(), newAxis);
                            }
                        }
                        
                        // Fix any invalid presets or items arrays without recreating
                        state.axes.forEach((axis) => {
                            if (!AXIS_DRIVER_PRESETS.some((entry) => entry.value === axis.preset)) {
                                axis.preset = "none";
                            }
                            if (!Array.isArray(axis.items)) {
                                axis.items = [];
                            }
                            // Trim to max items if needed
                            if (axis.items.length > AXIS_DRIVER_MAX_ITEMS) {
                                axis.items = axis.items.slice(0, AXIS_DRIVER_MAX_ITEMS);
                            }
                        });
                    }
                    if (!state.style || typeof state.style !== "object") {
                        state.style = { ...AXIS_DRIVER_DEFAULT_STYLE };
                    } else {
                        state.style = {
                            ...AXIS_DRIVER_DEFAULT_STYLE,
                            ...state.style,
                        };
                    }
                };

                const serialiseState = () => ({
                    axes: state.axes.map((axis) => ({
                        slot: axis.slot,
                        preset: axis.preset,
                        items: axis.items.map((item) => {
                            const payload = {
                                label: item.label || "",
                            };
                            if (axis.preset === "prompt") {
                                payload.value = typeof item.value === "string" ? item.value : String(item.value ?? "");
                            } else if (["steps", "seed"].includes(axis.preset)) {
                                const numeric = Number.parseInt(item.value, 10);
                                payload.value = Number.isFinite(numeric) ? numeric : "";
                            } else if (["cfg", "denoise"].includes(axis.preset)) {
                                const numeric = Number.parseFloat(item.value);
                                payload.value = Number.isFinite(numeric) ? numeric : "";
                            } else if (axis.preset === "lora") {
                                payload.value = typeof item.value === "string" ? item.value : String(item.value ?? "");
                                const strengthNumeric = Number.parseFloat(item.strength);
                                if (Number.isFinite(strengthNumeric)) {
                                    payload.strength = strengthNumeric;
                                }
                            } else {
                                payload.value = item.value ?? "";
                            }
                            if (item.overrides && typeof item.overrides === "object" && Object.keys(item.overrides).length) {
                                payload.overrides = { ...item.overrides };
                            }
                            return payload;
                        }),
                    })),
                    style: { ...state.style },
                });

                    const commitState = (shouldRerender = false, immediate = false) => {
                    // For non-rerender calls, debounce to batch rapid changes
                    if (!shouldRerender && !immediate) {
                        if (commitDebounceTimer) {
                            clearTimeout(commitDebounceTimer);
                        }
                        commitDebounceTimer = setTimeout(() => {
                            commitState(false, true); // Call immediately after delay
                        }, 150); // 150ms debounce
                        return;
                    }
                    
                    ensureStateStructure();
                    const stateObj = serialiseState();
                    console.log(`[h4 Axis Driver] Serializing state:`, stateObj);
                    console.log(`[h4 Axis Driver] X items: ${stateObj.axes[0].items.length}, Y items: ${stateObj.axes[1].items.length}, Z items: ${stateObj.axes[2].items.length}`);
                    const serialised = JSON.stringify(stateObj, null, 2);
                    
                    console.log(`[h4 Axis Driver] commitState executing, shouldRerender=${shouldRerender}`);
                    
                    // CRITICAL: Temporarily remove callback AND suppress all events
                    const savedCallback = configWidget.callback;
                    configWidget.callback = null;
                    suppressCallback = true;
                    
                    // Update widget values without triggering ANY events
                    const oldValue = configWidget.value;
                    
                    try {
                        configWidget.value = serialised;
                        if (configWidget.inputEl) {
                            // Remove and re-add event listeners to prevent any triggers
                            const inputEl = configWidget.inputEl;
                            const savedOnchange = inputEl.onchange;
                            const savedOninput = inputEl.oninput;
                            inputEl.onchange = null;
                            inputEl.oninput = null;
                            inputEl.value = serialised;
                            inputEl.onchange = savedOnchange;
                            inputEl.oninput = savedOninput;
                        }
                    } finally {
                        // Always restore callback and flag
                        configWidget.callback = savedCallback;
                        suppressCallback = false;
                    }
                    
                    if (shouldRerender) {
                        console.log("[h4 Axis Driver] Full rerender triggered");
                        render();
                    } else {
                        adjustNodeSize();
                    }
                    if (typeof node.setDirtyCanvas === "function") {
                        node.setDirtyCanvas(true, true);
                    }
                };

                const ensureAssetDatalist = (inputEl, preset, axisSlot, index) => {
                    const kinds = ASSET_KIND_MAPPING[preset];
                    if (!kinds) {
                        return;
                    }
                    const datalistId = `h4-axis-${node.id}-${axisSlot}-${preset}-${index}`;
                    let datalist = container.querySelector(`#${datalistId}`);
                    if (!datalist) {
                        datalist = document.createElement("datalist");
                        datalist.id = datalistId;
                        container.appendChild(datalist);
                    }
                    inputEl.setAttribute("list", datalistId);
                    loadAssetKinds(kinds).then((options) => {
                        if (!options.length) {
                            return;
                        }
                        datalist.innerHTML = "";
                        for (const option of options) {
                            const opt = document.createElement("option");
                            opt.value = option;
                            datalist.appendChild(opt);
                        }
                    });
                };

                const createItemRow = (axis, item, index) => {
                    const row = document.createElement("div");
                    row.style.display = "flex";
                    row.style.flexDirection = "column";
                    row.style.gap = "6px";
                    row.style.padding = "6px";
                    row.style.border = "1px solid rgba(255,255,255,0.08)";
                    row.style.borderRadius = "6px";

                    const top = document.createElement("div");
                    top.style.display = "flex";
                    top.style.gap = "6px";
                    top.style.alignItems = "center";

                    const labelInput = document.createElement("input");
                    labelInput.type = "text";
                    labelInput.id = `h4-axis-${node.id}-${axis.slot}-label-${index}`;
                    labelInput.placeholder = "Label";
                    labelInput.value = item.label || "";
                    labelInput.style.flex = "1";
                    applyTooltip(labelInput, "Friendly label for this entry. Appears in the summary output.");
                    labelInput.oninput = (ev) => {
                        item.label = ev.target.value;
                        commitState(false);
                    };

                    const removeBtn = document.createElement("button");
                    removeBtn.textContent = "Remove";
                    removeBtn.style.fontSize = "0.75rem";
                    removeBtn.style.padding = "3px 8px";
                    applyTooltip(removeBtn, "Delete this entry from the axis.");
                    removeBtn.onclick = () => {
                        // Remove item from data
                        axis.items.splice(index, 1);
                        
                        // Remove this row from DOM
                        row.remove();
                        
                        // Update counter in the parent card
                        const card = row.closest('[style*="border-radius: 8px"]');
                        if (card) {
                            const countLabel = card.querySelector('span[style*="font-size: 0.75rem"]');
                            if (countLabel) {
                                countLabel.textContent = `${axis.items.length}/${AXIS_DRIVER_MAX_ITEMS} items`;
                            }
                            
                            // Re-enable Add button if we're now under the limit
                            const addBtn = card.querySelector('button');
                            if (addBtn && axis.items.length < AXIS_DRIVER_MAX_ITEMS) {
                                addBtn.disabled = false;
                            }
                            
                            // If no items left, show placeholder
                            const itemsContainer = card.querySelector(".h4-axis-items");
                            if (itemsContainer && axis.items.length === 0) {
                                itemsContainer.innerHTML = "";
                                const empty = document.createElement("div");
                                empty.textContent = axis.preset === "none"
                                    ? "Axis disabled. Choose a preset to enable."
                                    : "No entries yet. Add items below.";
                                empty.style.fontSize = "0.8rem";
                                empty.style.opacity = "0.75";
                                itemsContainer.appendChild(empty);
                            }
                        }
                        
                        node.h4AxisDriverManualSize = null;
                        commitState(false); // Save state without full rerender
                        adjustNodeSize(); // Adjust size for removed item
                    };

                    top.appendChild(labelInput);
                    top.appendChild(removeBtn);
                    row.appendChild(top);

                    const valueRow = document.createElement("div");
                    valueRow.style.display = "flex";
                    valueRow.style.gap = "6px";
                    valueRow.style.alignItems = "center";

                    const preset = axis.preset;
                    if (preset === "prompt") {
                        const textarea = document.createElement("textarea");
                        textarea.id = `h4-axis-${node.id}-${axis.slot}-value-${index}`;
                        textarea.rows = 3;
                        textarea.style.width = "100%";
                        textarea.placeholder = "Prompt suffix";
                        textarea.value = typeof item.value === "string" ? item.value : String(item.value ?? "");
                        applyTooltip(textarea, "Prompt text appended to the base prompt for this iteration.");
                        textarea.oninput = (ev) => {
                            item.value = ev.target.value;
                            commitState(false);
                        };
                        valueRow.appendChild(textarea);
                    } else if (["steps", "seed"].includes(preset)) {
                        const input = document.createElement("input");
                        input.id = `h4-axis-${node.id}-${axis.slot}-value-${index}`;
                        input.type = "number";
                        input.step = "1";
                        input.min = preset === "steps" ? "1" : "0";
                        input.placeholder = preset === "steps" ? "Steps" : "Seed";
                        input.value = item.value === "" || item.value === null || item.value === undefined ? "" : item.value;
                        applyTooltip(input, preset === "steps"
                            ? "Number of sampler steps for this entry."
                            : "Seed to reproduce randomness for this entry.");
                        input.oninput = (ev) => {
                            const raw = ev.target.value;
                            item.value = raw === "" ? "" : Number.parseInt(raw, 10);
                            commitState(false);
                        };
                        valueRow.appendChild(input);
                    } else if (["cfg", "denoise"].includes(preset)) {
                        const input = document.createElement("input");
                        input.id = `h4-axis-${node.id}-${axis.slot}-value-${index}`;
                        input.type = "number";
                        input.step = preset === "cfg" ? "0.1" : "0.01";
                        input.min = "0";
                        if (preset === "denoise") {
                            input.max = "1";
                        }
                        input.placeholder = preset === "cfg" ? "CFG" : "Denoise";
                        input.value = item.value === "" || item.value === null || item.value === undefined ? "" : item.value;
                        applyTooltip(input, preset === "cfg"
                            ? "Classifier-Free Guidance scale for this entry."
                            : "Denoise strength (0-1) for this entry.");
                        input.oninput = (ev) => {
                            const raw = ev.target.value;
                            item.value = raw === "" ? "" : Number.parseFloat(raw);
                            commitState(false);
                        };
                        valueRow.appendChild(input);
                    } else if (preset === "sampler") {
                        const select = document.createElement("select");
                        select.id = `h4-axis-${node.id}-${axis.slot}-value-${index}`;
                        const placeholder = document.createElement("option");
                        placeholder.value = "";
                        placeholder.textContent = "Select sampler";
                        select.appendChild(placeholder);
                        DEFAULT_SAMPLERS.forEach((sampler) => {
                            const option = document.createElement("option");
                            option.value = sampler;
                            option.textContent = sampler;
                            select.appendChild(option);
                        });
                        select.value = item.value || "";
                        applyTooltip(select, "Choose which sampler to run for this entry.");
                        select.onchange = (ev) => {
                            item.value = ev.target.value;
                            commitState(false);
                        };
                        valueRow.appendChild(select);
                    } else if (preset === "scheduler") {
                        const select = document.createElement("select");
                        select.id = `h4-axis-${node.id}-${axis.slot}-value-${index}`;
                        const placeholder = document.createElement("option");
                        placeholder.value = "";
                        placeholder.textContent = "Select scheduler";
                        select.appendChild(placeholder);
                        DEFAULT_SCHEDULERS.forEach((scheduler) => {
                            const option = document.createElement("option");
                            option.value = scheduler;
                            option.textContent = scheduler;
                            select.appendChild(option);
                        });
                        select.value = item.value || "";
                        applyTooltip(select, "Choose which scheduler curve to use for this entry.");
                        select.onchange = (ev) => {
                            item.value = ev.target.value;
                            commitState(false);
                        };
                        valueRow.appendChild(select);
                    } else if (["checkpoint", "lora", "vae", "clip", "clip_vision"].includes(preset)) {
                        const select = document.createElement("select");
                        select.id = `h4-axis-${node.id}-${axis.slot}-value-${index}`;
                        select.style.flex = "1";
                        select.style.minWidth = "150px";
                        const placeholder = document.createElement("option");
                        placeholder.value = "";
                        placeholder.textContent = `Select ${preset}...`;
                        select.appendChild(placeholder);
                        
                        const currentValue = typeof item.value === "string" ? item.value : String(item.value ?? "");
                        select.value = "";
                        
                        const tooltipText = {
                            checkpoint: "Select a checkpoint model to test in this grid position.",
                            lora: "Select a LoRA model to apply for this grid position.",
                            vae: "Select a VAE model to use for decoding.",
                            clip: "Select a CLIP text encoder model.",
                            clip_vision: "Select a CLIP vision model."
                        }[preset] || "Select a model from the list.";
                        
                        applyTooltip(select, tooltipText);
                        
                        select.onchange = (ev) => {
                            item.value = ev.target.value;
                            commitState(false);
                        };
                        
                        valueRow.appendChild(select);
                        
                        const kinds = ASSET_KIND_MAPPING[preset];
                        if (kinds) {
                            loadAssetKinds(kinds).then((options) => {
                                select.innerHTML = "";
                                const newPlaceholder = document.createElement("option");
                                newPlaceholder.value = "";
                                newPlaceholder.textContent = `Select ${preset}...`;
                                select.appendChild(newPlaceholder);
                                
                                options.forEach((name) => {
                                    const option = document.createElement("option");
                                    option.value = name;
                                    option.textContent = name;
                                    select.appendChild(option);
                                });
                                
                                if (currentValue && options.includes(currentValue)) {
                                    select.value = currentValue;
                                } else if (currentValue) {
                                    const customOption = document.createElement("option");
                                    customOption.value = currentValue;
                                    customOption.textContent = `${currentValue} (custom)`;
                                    select.appendChild(customOption);
                                    select.value = currentValue;
                                }
                            });
                        }
                    } else {
                        const input = document.createElement("input");
                        input.type = "text";
                        input.placeholder = "Value";
                        input.value = typeof item.value === "string" ? item.value : String(item.value ?? "");
                        input.oninput = (ev) => {
                            item.value = ev.target.value;
                            commitState(false);
                        };
                        applyTooltip(input, "Value applied for this entry.");
                        valueRow.appendChild(input);
                    }

                    if (preset === "lora") {
                        const strengthInput = document.createElement("input");
                        strengthInput.type = "number";
                        strengthInput.step = "0.01";
                        strengthInput.min = "0";
                        strengthInput.max = "2";
                        strengthInput.placeholder = "Strength";
                        strengthInput.style.width = "90px";
                        strengthInput.value = Number.isFinite(Number(item.strength)) ? Number(item.strength) : 0.75;
                        strengthInput.oninput = (ev) => {
                            const raw = ev.target.value;
                            item.strength = raw === "" ? 0.75 : Number.parseFloat(raw);
                            commitState(false);
                        };
                        applyTooltip(strengthInput, "Adjust LoRA influence (0 disables, 1 is full strength).");
                        valueRow.appendChild(strengthInput);
                    }

                    row.appendChild(valueRow);
                    return row;
                };

                const renderAxisCard = (axis) => {
                    const card = document.createElement("div");
                    card.style.border = "1px solid rgba(255,255,255,0.08)";
                    card.style.borderRadius = "8px";
                    card.style.padding = "8px";
                    card.style.display = "flex";
                    card.style.flexDirection = "column";
                    card.style.gap = "8px";
                        card.style.background = "rgba(0,0,0,0.22)";
                        card.style.width = "100%";
                        card.style.boxSizing = "border-box";

                    const header = document.createElement("div");
                    header.style.display = "flex";
                    header.style.justifyContent = "space-between";
                    header.style.alignItems = "center";

                    const title = document.createElement("div");
                    title.textContent = `Axis ${axis.slot}`;
                    title.style.fontWeight = "600";
                    applyTooltip(title, `Configure what Axis ${axis.slot} changes across the grid.`);
                    header.appendChild(title);

                    const safePreset = AXIS_DRIVER_PRESETS.some((entry) => entry.value === axis.preset) ? axis.preset : "none";
                    axis.preset = safePreset;

                    const presetSelect = document.createElement("select");
                    presetSelect.style.minWidth = "140px";
                    presetSelect.style.maxWidth = "200px";
                    presetSelect.style.alignSelf = "flex-start";
                    AXIS_DRIVER_PRESETS.forEach((entry) => {
                        const option = document.createElement("option");
                        option.value = entry.value;
                        option.textContent = entry.label;
                        presetSelect.appendChild(option);
                    });
                    presetSelect.value = axis.preset;
                    applyTooltip(presetSelect, "Choose the parameter this axis will sweep through.");
                    presetSelect.onchange = (ev) => {
                        const next = ev.target.value;
                        axis.preset = next;
                        if (next === "none") {
                            axis.items = [];
                        } else {
                            axis.items = axis.items.slice(0, AXIS_DRIVER_MAX_ITEMS).map((item) => normaliseItem(next, item));
                        }
                        
                        // Update the items container incrementally instead of full rerender
                        const itemsContainer = card.querySelector(".h4-axis-items");
                        if (itemsContainer) {
                            itemsContainer.innerHTML = "";
                            if (axis.items.length === 0) {
                                const empty = document.createElement("div");
                                empty.textContent = next === "none"
                                    ? "Axis disabled. Choose a preset to enable."
                                    : "No entries yet. Add items below.";
                                empty.style.fontSize = "0.8rem";
                                empty.style.opacity = "0.75";
                                itemsContainer.appendChild(empty);
                            } else {
                                axis.items.forEach((item, idx) => {
                                    itemsContainer.appendChild(createItemRow(axis, item, idx));
                                });
                            }
                        }
                        
                        // Update Add button state
                        const addBtn = card.querySelector('button');
                        if (addBtn && addBtn.textContent === "Add item") {
                            addBtn.disabled = next === "none" || axis.items.length >= AXIS_DRIVER_MAX_ITEMS;
                        }
                        
                        // Update counter
                        const countLabel = card.querySelector('span[style*="font-size: 0.75rem"]');
                        if (countLabel) {
                            countLabel.textContent = `${axis.items.length}/${AXIS_DRIVER_MAX_ITEMS} items`;
                        }
                        
                        node.h4AxisDriverManualSize = null;
                        commitState(false); // Save state without rerender
                        adjustNodeSize(); // Adjust size for new items
                    };
                    header.appendChild(presetSelect);
                    card.appendChild(header);

                    const itemsContainer = document.createElement("div");
                    itemsContainer.className = "h4-axis-items";
                    itemsContainer.style.display = "flex";
                    itemsContainer.style.flexDirection = "column";
                    itemsContainer.style.gap = "6px";

                    const itemList = Array.isArray(axis.items) ? axis.items : [];
                    axis.items = itemList;

                    if (itemList.length === 0) {
                        const empty = document.createElement("div");
                        empty.textContent = axis.preset === "none"
                            ? "Axis disabled. Choose a preset to enable."
                            : "No entries yet. Add items below.";
                        empty.style.fontSize = "0.8rem";
                        empty.style.opacity = "0.75";
                        itemsContainer.appendChild(empty);
                    } else {
                        itemList.forEach((item, index) => {
                            itemsContainer.appendChild(createItemRow(axis, item, index));
                        });
                    }
                    card.appendChild(itemsContainer);

                    const footer = document.createElement("div");
                    footer.style.display = "flex";
                    footer.style.justifyContent = "space-between";
                    footer.style.alignItems = "center";

                    footer.style.flexWrap = "wrap";
                    footer.style.gap = "6px";

                    const countLabel = document.createElement("span");
                    countLabel.textContent = `${axis.items.length}/${AXIS_DRIVER_MAX_ITEMS} items`;
                    countLabel.style.fontSize = "0.75rem";
                    countLabel.style.opacity = "0.7";
                    footer.appendChild(countLabel);

                    const addBtn = document.createElement("button");
                    addBtn.textContent = "Add item";
                    addBtn.style.fontSize = "0.8rem";
                    addBtn.style.padding = "4px 10px";
                    addBtn.style.alignSelf = "flex-end";
                    applyTooltip(addBtn, "Append another entry to this axis.");
                    addBtn.disabled = axis.preset === "none" || axis.items.length >= AXIS_DRIVER_MAX_ITEMS;
                    addBtn.onclick = () => {
                        if (axis.items.length >= AXIS_DRIVER_MAX_ITEMS || axis.preset === "none") {
                            return;
                        }
                        const empty = {
                            label: "",
                            value: axis.preset === "prompt" ? "" : "",
                            strength: axis.preset === "lora" ? 0.75 : null,
                            overrides: {},
                        };
                        const newIndex = axis.items.length;
                        axis.items.push(empty);
                        
                        // Instead of full rerender, just append the new item row
                        const itemsContainer = card.querySelector(".h4-axis-items");
                        if (itemsContainer) {
                            // Remove "no entries" placeholder if this is the first item
                            if (newIndex === 0) {
                                itemsContainer.innerHTML = "";
                            }
                            const newRow = createItemRow(axis, empty, newIndex);
                            itemsContainer.appendChild(newRow);
                        }
                        
                        // Update counter
                        countLabel.textContent = `${axis.items.length}/${AXIS_DRIVER_MAX_ITEMS} items`;
                        addBtn.disabled = axis.items.length >= AXIS_DRIVER_MAX_ITEMS;
                        
                        node.h4AxisDriverManualSize = null;
                        commitState(false); // Don't rerender, just save state
                        adjustNodeSize(); // Adjust size for new item
                    };
                    footer.appendChild(addBtn);
                    card.appendChild(footer);
                    applyTooltip(card, `Entries for Axis ${axis.slot}. Each item becomes a row/column in the grid.`);

                    return card;
                };

                const renderStylePanel = () => {
                    styleBody.innerHTML = "";

                    const makeNumberField = (labelText, key, min, max, step, tooltip) => {
                        const wrapper = document.createElement("label");
                        wrapper.style.display = "flex";
                        wrapper.style.flexDirection = "column";
                        wrapper.style.gap = "4px";
                        const span = document.createElement("span");
                        span.textContent = labelText;
                        span.style.fontSize = "0.75rem";
                        applyTooltip(span, tooltip);
                        const input = document.createElement("input");
                        input.type = "number";
                        if (min !== undefined) input.min = String(min);
                        if (max !== undefined) input.max = String(max);
                        if (step !== undefined) input.step = String(step);
                        input.value = state.style[key];
                        input.oninput = (ev) => {
                            const raw = Number.parseFloat(ev.target.value);
                            if (Number.isFinite(raw)) {
                                state.style[key] = raw;
                                commitState(false);
                            }
                        };
                        applyTooltip(input, tooltip);
                        wrapper.appendChild(span);
                        wrapper.appendChild(input);
                        styleBody.appendChild(wrapper);
                    };

                    makeNumberField("Font size", "font_size", 8, 96, 1, "Set the font size (px) used for axis labels in the final grid.");

                    const fontFamily = document.createElement("label");
                    fontFamily.style.display = "flex";
                    fontFamily.style.flexDirection = "column";
                    fontFamily.style.gap = "4px";
                    const fontSpan = document.createElement("span");
                    fontSpan.textContent = "Font family";
                    fontSpan.style.fontSize = "0.75rem";
                    applyTooltip(fontSpan, "Provide a CSS font family stack for axis labels. Leave blank for default.");
                    const fontInput = document.createElement("input");
                    fontInput.type = "text";
                    fontInput.value = state.style.font_family || "";
                    fontInput.oninput = (ev) => {
                        state.style.font_family = ev.target.value;
                        commitState(false);
                    };
                    applyTooltip(fontInput, "Provide a CSS font family stack for axis labels. Leave blank for default.");
                    fontFamily.appendChild(fontSpan);
                    fontFamily.appendChild(fontInput);
                    styleBody.appendChild(fontFamily);

                    const colourWrapper = document.createElement("label");
                    colourWrapper.style.display = "flex";
                    colourWrapper.style.flexDirection = "column";
                    colourWrapper.style.gap = "4px";
                    const colourSpan = document.createElement("span");
                    colourSpan.textContent = "Font colour";
                    colourSpan.style.fontSize = "0.75rem";
                    applyTooltip(colourSpan, "Hex colour used for axis labels. Leave blank to inherit theme text colour.");
                    const colourInput = document.createElement("input");
                    colourInput.type = "text";
                    colourInput.placeholder = "#RRGGBB";
                    colourInput.value = state.style.font_colour || "";
                    colourInput.oninput = (ev) => {
                        state.style.font_colour = ev.target.value;
                        commitState(false);
                    };
                    applyTooltip(colourInput, "Hex colour used for axis labels. Leave blank to inherit theme text colour.");
                    colourWrapper.appendChild(colourSpan);
                    colourWrapper.appendChild(colourInput);
                    styleBody.appendChild(colourWrapper);

                    const makeSelect = (labelText, key, options, tooltip) => {
                        const wrapper = document.createElement("label");
                        wrapper.style.display = "flex";
                        wrapper.style.flexDirection = "column";
                        wrapper.style.gap = "4px";
                        const span = document.createElement("span");
                        span.textContent = labelText;
                        span.style.fontSize = "0.75rem";
                        applyTooltip(span, tooltip);
                        const select = document.createElement("select");
                        options.forEach((entry) => {
                            const option = document.createElement("option");
                            option.value = entry.value;
                            option.textContent = entry.label;
                            select.appendChild(option);
                        });
                        select.value = state.style[key];
                        select.onchange = (ev) => {
                            state.style[key] = ev.target.value;
                            commitState(false);
                        };
                        applyTooltip(select, tooltip);
                        wrapper.appendChild(span);
                        wrapper.appendChild(select);
                        styleBody.appendChild(wrapper);
                    };

                    makeSelect("Alignment", "alignment", AXIS_STYLE_ALIGNMENT_OPTIONS, "Align axis headers left, center, or right within their cell.");
                    makeSelect("Label position", "label_position", AXIS_STYLE_POSITION_OPTIONS, "Choose whether headers sit above, below, or overlay the grid.");
                    makeSelect("Label layout", "label_layout", AXIS_LABEL_LAYOUT_OPTIONS, "Choose how labels are rendered: overlay (on images), border (outside images in margins), or none (no labels).");
                    makeSelect("Background", "background", AXIS_STYLE_BACKGROUND_OPTIONS, "Toggle background styles for the axis header bar.");

                    const makeTextField = (labelText, key, tooltip) => {
                        const wrapper = document.createElement("label");
                        wrapper.style.display = "flex";
                        wrapper.style.flexDirection = "column";
                        wrapper.style.gap = "4px";
                        const span = document.createElement("span");
                        span.textContent = labelText;
                        span.style.fontSize = "0.75rem";
                        applyTooltip(span, tooltip);
                        const input = document.createElement("input");
                        input.type = "text";
                        input.value = state.style[key] || "";
                        input.oninput = (ev) => {
                            state.style[key] = ev.target.value;
                            commitState(false);
                        };
                        applyTooltip(input, tooltip);
                        wrapper.appendChild(span);
                        wrapper.appendChild(input);
                        styleBody.appendChild(wrapper);
                    };

                    makeTextField("Header label for X", "custom_label_x", "Override the displayed header text for the X axis.");
                    makeTextField("Header label for Y", "custom_label_y", "Override the displayed header text for the Y axis.");
                    makeTextField("Header label for Z", "custom_label_z", "Override the displayed header text for the Z axis.");

                    const headerToggle = document.createElement("label");
                    headerToggle.style.display = "flex";
                    headerToggle.style.gap = "6px";
                    headerToggle.style.alignItems = "center";
                    headerToggle.style.marginTop = "4px";
                    const checkbox = document.createElement("input");
                    checkbox.type = "checkbox";
                    checkbox.checked = Boolean(state.style.show_axis_headers);
                    checkbox.onchange = (ev) => {
                        state.style.show_axis_headers = Boolean(ev.target.checked);
                        commitState(false);
                    };
                    const checkboxLabel = document.createElement("span");
                    checkboxLabel.textContent = "Show axis headers";
                    checkboxLabel.style.fontSize = "0.8rem";
                    applyTooltip(checkbox, "Toggle whether the header row/column is rendered on the grid output.");
                    applyTooltip(checkboxLabel, "Toggle whether the header row/column is rendered on the grid output.");
                    headerToggle.appendChild(checkbox);
                    headerToggle.appendChild(checkboxLabel);
                    styleBody.appendChild(headerToggle);
                };

                        const render = () => {
                            console.debug("h4 Axis Driver render begin", node.id, { state });
                            
                            // Preserve focused element and scroll position before destroying DOM
                            const activeElement = document.activeElement;
                            const wasInThisNode = axisStack.contains(activeElement);
                            const focusedInputId = wasInThisNode && activeElement ? activeElement.id : null;
                            const focusedInputValue = wasInThisNode && activeElement && activeElement.value !== undefined ? activeElement.value : null;
                            const focusedSelectionStart = wasInThisNode && activeElement && activeElement.selectionStart !== undefined ? activeElement.selectionStart : null;
                            const focusedSelectionEnd = wasInThisNode && activeElement && activeElement.selectionEnd !== undefined ? activeElement.selectionEnd : null;
                            
                            ensureStateStructure();
                            const rawAxes = Array.isArray(state.axes) ? state.axes : [];
                            const axesList = rawAxes.filter((axis) => axis && typeof axis === "object");
                            renderStatus.textContent = `axes:${axesList.length}`;
                            if (!axesList.length) {
                                const fallback = normaliseState(cloneDefault(AXIS_DRIVER_DEFAULT_STATE));
                                axesList.push(...fallback.axes);
                                state.axes = fallback.axes;
                                if (!state.style || typeof state.style !== "object") {
                                    state.style = fallback.style;
                                }
                            } else if (!state.style || typeof state.style !== "object") {
                                state.style = { ...AXIS_DRIVER_DEFAULT_STYLE };
                            }
                            axisStack.innerHTML = "";
                            if (axesList.length) {
                                axesList.forEach((axis) => {
                                    try {
                                        axisStack.appendChild(renderAxisCard(axis));
                                    } catch (err) {
                                        console.warn("h4 Axis Driver: failed to render axis", axis?.slot, err);
                                    }
                                });
                                console.debug("h4 Axis Driver rendered cards", axesList.map((axis) => ({ slot: axis.slot, preset: axis.preset, items: axis.items?.length ?? 0 })));
                            } else {
                                const empty = document.createElement("div");
                                empty.textContent = "Axis configuration unavailable.";
                                empty.style.fontSize = "0.8rem";
                                empty.style.opacity = "0.7";
                                axisStack.appendChild(empty);
                            }
                            renderStylePanel();
                            adjustNodeSize(true);
                            
                            // Restore focus and cursor position if we destroyed a focused input
                            if (focusedInputId && wasInThisNode) {
                                // Give the DOM a moment to settle
                                requestAnimationFrame(() => {
                                    const elementToFocus = document.getElementById(focusedInputId);
                                    if (elementToFocus) {
                                        elementToFocus.focus();
                                        if (focusedInputValue !== null && elementToFocus.value !== undefined) {
                                            elementToFocus.value = focusedInputValue;
                                        }
                                        if (focusedSelectionStart !== null && focusedSelectionEnd !== null && elementToFocus.setSelectionRange) {
                                            try {
                                                elementToFocus.setSelectionRange(focusedSelectionStart, focusedSelectionEnd);
                                            } catch (e) {
                                                // Selection restoration failed, not critical
                                            }
                                        }
                                        console.debug("[h4 Axis Driver] Restored focus to", focusedInputId);
                                    }
                                });
                            }
                            
                            console.debug("h4 Axis Driver render end", node.id, {
                                size: [...node.size],
                                manual: node.h4AxisDriverManualSize,
                                min: node.h4AxisDriverMinSize,
                                measurement: node.h4AxisDriverLastMeasurement,
                            });
                        };

                    const originalCallback = configWidget.callback;
                    configWidget.callback = function (value) {
                        if (originalCallback) {
                            originalCallback.call(this, value);
                        }
                        if (suppressCallback) {
                            console.log("[h4 Axis Driver] Callback suppressed (internal update)");
                            return;
                        }
                        console.log("[h4 Axis Driver] External config change detected - full rerender");
                        state = normaliseState(parseState(value));
                        ensureStateStructure();
                        render();
                    };

                    render();
                    commitState(false);
                    render();
                    node.h4AxisDriverUiReady = true;
                } catch (error) {
                    console.error("h4 Axis Driver UI initialisation failed", error);
                    if (configWidget?.inputEl) {
                        configWidget.inputEl.style.display = "";
                    }
                    node.h4AxisDriverUiAttached = false;
                }
                };

                requestAnimationFrame(installUi);
            };
        }
        if (["h4DebugATron3000", "h4DebugATronRouter", "h4DebugATron3000Console"].includes(nodeData.name)) {
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

        if (isH4Node && nodeData.name !== "h4PlotXY" && nodeData.name !== "h4AxisDriver") {
            const priorOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (priorOnNodeCreated) {
                    priorOnNodeCreated.apply(this, arguments);
                }
                applyH4NodeColor(this);
            };
        }
    },
});

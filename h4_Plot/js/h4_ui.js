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
};

const DYNAMIC_INPUT_PREFIX = "any_slot_";
const DYNAMIC_OUTPUT_PREFIX = "any_out_";
const DYNAMIC_SLOTS = 8;
const BASE_OUTPUT_COUNT = 9; // image..string before dynamic slots
const ORIENTATION_HORIZONTAL = "horizontal";

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
        if (nodeData.name === "h4PlotNode") {
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
                applySliderTooltip(this, "width");
                applySliderTooltip(this, "height");
                applySliderTooltip(this, "seed");
            };
        }
        if (nodeData.name === "h4DebugATron3000") {
            const baseOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (baseOnNodeCreated) {
                    baseOnNodeCreated.apply(this, arguments);
                }
                ensureDynamicSockets(this);
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

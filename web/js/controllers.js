import { app } from "/scripts/app.js";

// NilorPreset dynamic inputs extension
// Adds a new empty _input_hook_N slot whenever the last slot gets connected, up to a hard cap

const MAX_INPUTS = 32;
const CLASS_TYPE = "NilorPreset";
const INPUT_PREFIX = "_preset_hook_";

function isTargetNode(node) {
    return node && (node.comfyClass === CLASS_TYPE || node.type === CLASS_TYPE);
}

function countHookInputs(node) {
    return (node.inputs || []).filter((i) => i && i.name?.startsWith(INPUT_PREFIX)).length;
}

function nextInputName(node) {
    let index = 1;
    while (index <= MAX_INPUTS) {
        const key = `${INPUT_PREFIX}${index}`;
        if (!node.inputs || !node.inputs.find((i) => i.name === key)) {
            return key;
        }
        index++;
    }
    return null;
}

function resizeNode(node) {
    try {
        const size = node.computeSize();
        node.onResize?.(size);
        app.graph?.setDirtyCanvas(true, true);
    } catch (_) {}
}

function ensureAtLeastOneSlot(node) {
    if (!isTargetNode(node)) return;
    if (countHookInputs(node) === 0) {
        const name = `${INPUT_PREFIX}1`;
        node.addInput(name, "PRESET_HOOK");
        resizeNode(node);
    }
}

function growIfLastLinked(node) {
    if (!isTargetNode(node)) return;
    const inputs = (node.inputs || []).filter((i) => i && i.name?.startsWith(INPUT_PREFIX));
    if (inputs.length === 0) return;
    const last = inputs[inputs.length - 1];
    const lastIsLinked = !!last.link;
    if (lastIsLinked && inputs.length < MAX_INPUTS) {
        const name = nextInputName(node);
        if (name) {
            node.addInput(name, "PRESET_HOOK");
            resizeNode(node);
        }
    }
}

function shrinkTrailingUnlinked(node) {
    if (!isTargetNode(node)) return;
    const allInputs = node.inputs || [];
    // Collect indices of hook inputs
    const hookIndices = [];
    for (let i = 0; i < allInputs.length; i++) {
        const inp = allInputs[i];
        if (inp && inp.name && inp.name.startsWith(INPUT_PREFIX)) {
            hookIndices.push(i);
        }
    }
    if (hookIndices.length <= 1) return; // always keep at least one

    // Find last linked among hook inputs (by position in hookIndices)
    let lastLinkedPos = -1;
    for (let pos = 0; pos < hookIndices.length; pos++) {
        const idx = hookIndices[pos];
        if (allInputs[idx]?.link) lastLinkedPos = pos;
    }

    const targetHookCount = lastLinkedPos >= 0 ? lastLinkedPos + 1 : 1;

    // Remove trailing unlinked beyond targetHookCount
    for (let pos = hookIndices.length - 1; pos >= targetHookCount; pos--) {
        const idx = hookIndices[pos];
        const input = node.inputs[idx];
        if (input && !input.link) {
            try {
                node.removeInput(idx);
            } catch (e) {
                console.warn("nilor-preset-dynamic-inputs removeInput error", e);
                break;
            }
        } else {
            break;
        }
    }
    resizeNode(node);
}

app.registerExtension({
    name: "comfy.nilor-nodes.userinputPreset",

    // Ensure compatibility with saved/loaded graphs
    afterConfigureGraph(graph) {
        try {
            (graph?._nodes || graph?.nodes || []).forEach((n) => {
                if (isTargetNode(n)) {
                    ensureAtLeastOneSlot(n);
                    shrinkTrailingUnlinked(n);
                    growIfLastLinked(n);
                }
            });
        } catch (e) {
            console.warn("nilor-preset-dynamic-inputs afterConfigureGraph error", e);
        }
    },

    // Patch the prototype so we always react to connection changes
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData?.name !== CLASS_TYPE) return;
        const original = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
            if (typeof original === "function") {
                original.apply(this, arguments);
            }
            try {
                shrinkTrailingUnlinked(this);
                growIfLastLinked(this);
            } catch (e) {
                console.warn("nilor-preset-dynamic-inputs onConnectionsChange error", e);
            }
        };
    },

    nodeCreated(node) {
        if (!isTargetNode(node)) return;
        ensureAtLeastOneSlot(node);
        shrinkTrailingUnlinked(node);
        growIfLastLinked(node);
    },
});



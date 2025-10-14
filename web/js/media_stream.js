import { app } from "/scripts/app.js";

function toggleFramerateWidget(node, show) {
    const framerateWidget = node.widgets.find((w) => w.name === "framerate");
    if (framerateWidget) {
        framerateWidget.hidden = !show;
        // This is a hack to force the node to redraw.
        //const size = node.computeSize();
        //node.onResize?.(size);
    }
}

function hideWidgets(node, widgetNames) {
    widgetNames.forEach(name => {
        const widget = node.widgets.find((w) => w.name === name);
        if (widget) {
            widget.hidden = true;
        }
    });
}

function setupMediaStreamOutput(node) {
    // Hide system inputs by default
    hideWidgets(node, [
        "content_id",
        "venue",
        "canvas",
        "scene",
        "job_type",
        "presigned_upload_url",
        "job_completions_queue_url",
        "output_object_keys",
    ]);

    const formatWidget = node.widgets.find((w) => w.name === "format");
    if (!formatWidget) return;

    // Apply current value
    toggleFramerateWidget(node, formatWidget.value === "mp4");
    try {
        const size = node.computeSize();
        node.onResize?.(size);
        app.graph?.setDirtyCanvas(true, true);
    } catch (_) {}

    // Chain the widget callback once
    if (!formatWidget.__nilorPatched) {
        const originalCallback = formatWidget.callback;
        formatWidget.callback = function (value) {
            toggleFramerateWidget(node, value === "mp4");
            try {
                const size = node.computeSize();
                node.onResize?.(size);
                app.graph?.setDirtyCanvas(true, true);
            } catch (_) {}
            if (originalCallback) return originalCallback.apply(this, arguments);
        };
        formatWidget.__nilorPatched = true;
    }
}

function setupMediaStreamInput(node) {
    hideWidgets(node, ["presigned_download_url"]);
}

app.registerExtension({
    name: "comfy.nilor-nodes.mediaStream",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData?.name === "MediaStreamOutput") {
            const origAdded = nodeType.prototype.onAdded;
            nodeType.prototype.onAdded = function () {
                if (typeof origAdded === "function") origAdded.apply(this, arguments);
                try { setTimeout(() => setupMediaStreamOutput(this), 0); } catch (_) {}
            };

            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (typeof origConfigure === "function") origConfigure.apply(this, arguments);
                try { setTimeout(() => setupMediaStreamOutput(this), 0); } catch (_) {}
            };
        }

        if (nodeData?.name === "MediaStreamInput") {
            const origAddedIn = nodeType.prototype.onAdded;
            nodeType.prototype.onAdded = function () {
                if (typeof origAddedIn === "function") origAddedIn.apply(this, arguments);
                try { setTimeout(() => setupMediaStreamInput(this), 0); } catch (_) {}
            };

            const origConfigureIn = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (typeof origConfigureIn === "function") origConfigureIn.apply(this, arguments);
                try { setTimeout(() => setupMediaStreamInput(this), 0); } catch (_) {}
            };
        }
    },

    afterConfigureGraph(graph) {
        try {
            (graph?._nodes || graph?.nodes || []).forEach((n) => {
                if (n?.comfyClass === "MediaStreamOutput") setupMediaStreamOutput(n);
                if (n?.comfyClass === "MediaStreamInput") setupMediaStreamInput(n);
            });
        } catch (e) {
            console.warn("nilor-media-stream afterConfigureGraph error", e);
        }
    },

    nodeCreated(node) {
        if (node.comfyClass === "MediaStreamOutput") setupMediaStreamOutput(node);
        if (node.comfyClass === "MediaStreamInput") setupMediaStreamInput(node);
    },
});

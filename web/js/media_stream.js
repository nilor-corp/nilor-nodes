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

app.registerExtension({
    name: "comfy.nilor-nodes.mediaStream",
    nodeCreated(node) {
        if (node.comfyClass === "MediaStreamOutput") {
            // Hide system inputs by default
            hideWidgets(node, [
                "job_id",
                "presigned_upload_url",
                "job_completions_queue_url",
                "output_object_keys"
            ]);

            const formatWidget = node.widgets.find((w) => w.name === "format");

            // Initial toggle for framerate based on the default format value
            toggleFramerateWidget(node, formatWidget.value === "mp4");

            // Store original callback to chain it
            const originalCallback = formatWidget.callback;

            formatWidget.callback = function (value) {
                toggleFramerateWidget(node, value === "mp4");

                // Recalculate node size after toggling widgets
                const size = node.computeSize();
                node.onResize?.(size);
                
                if (originalCallback) {
                    return originalCallback.apply(this, arguments);
                }
            };
        }

        if (node.comfyClass === "MediaStreamInput") {
            // Hide system inputs by default
            hideWidgets(node, ["presigned_download_url"]);
        }
    },
});

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Inline video preview + upload widget for the UniverSR video nodes.
// Adapted from HunyuanVideo-FoleyTune's FoleyTuneVideo.js.

const VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov", "avi", "flv", "wmv", "m4v", "mpg", "mpeg", "ts"];

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

function addVideoPreview(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        const node = this;
        const container = document.createElement("div");
        container.style.width = "100%";

        const videoEl = document.createElement("video");
        videoEl.controls = true;
        videoEl.loop = true;
        videoEl.muted = true;
        videoEl.style.width = "100%";
        videoEl.onmouseenter = () => { videoEl.muted = false; };
        videoEl.onmouseleave = () => { videoEl.muted = true; };
        container.appendChild(videoEl);

        const previewWidget = this.addDOMWidget("videopreview", "preview", container, {
            serialize: false,
            hideOnZoom: false,
            getValue() { return container.value; },
            setValue(v) { container.value = v; },
        });

        previewWidget.videoEl = videoEl;
        previewWidget.aspectRatio = null;

        previewWidget.computeSize = function (width) {
            if (this.aspectRatio && !container.hidden) {
                const height = (node.size[0] - 20) / this.aspectRatio + 10;
                return [width, Math.max(height, 0)];
            }
            return [width, -4];
        };

        videoEl.addEventListener("loadedmetadata", () => {
            previewWidget.aspectRatio = videoEl.videoWidth / videoEl.videoHeight;
            container.hidden = false;
            fitHeight(node);
        });

        videoEl.addEventListener("error", () => {
            container.hidden = true;
            fitHeight(node);
        });

        node._universrVideoPreview = previewWidget;

        const onExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            onExecuted?.apply(this, arguments);
            // custom key (see nodes_video.py) — core ignores it, so we render it once
            const g = output?.universr_videos?.[0];
            if (g) {
                const params = new URLSearchParams({
                    filename: g.filename,
                    type: g.type || "temp",
                    subfolder: g.subfolder || "",
                });
                videoEl.src = api.apiURL("/view?" + params.toString());
            }
        };
    };
}

function addUploadWidget(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        const node = this;
        const pathWidget = this.widgets.find((w) => w.name === "video");
        if (!pathWidget) return;

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "video/*,image/gif";
        fileInput.style.display = "none";
        document.body.appendChild(fileInput);

        async function uploadFile(file) {
            const body = new FormData();
            body.append("image", file);
            body.append("overwrite", "true");
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            if (resp.ok) {
                const data = await resp.json();
                if (!pathWidget.options.values.includes(data.name)) {
                    pathWidget.options.values.push(data.name);
                }
                pathWidget.value = data.name;
                pathWidget.callback?.(data.name);
            }
        }

        fileInput.onchange = () => {
            if (fileInput.files.length) uploadFile(fileInput.files[0]);
        };

        const uploadWidget = this.addWidget("button", "choose video to upload", null, () => {
            fileInput.click();
        });
        uploadWidget.serialize = false;

        this.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
        this.onDragDrop = async (e) => {
            const file = e?.dataTransfer?.files?.[0];
            if (!file) return false;
            const ext = file.name.split(".").pop()?.toLowerCase();
            if (!VIDEO_EXTENSIONS.includes(ext)) return false;
            await uploadFile(file);
            return true;
        };

        function showPreview(filename) {
            if (!filename) return;
            const pw = node._universrVideoPreview;
            if (!pw) return;
            const params = new URLSearchParams({ filename, type: "input", subfolder: "" });
            pw.videoEl.src = api.apiURL("/view?" + params.toString());
        }

        const origCallback = pathWidget.callback;
        pathWidget.callback = function (value) {
            origCallback?.apply(this, arguments);
            showPreview(value);
        };

        requestAnimationFrame(() => showPreview(pathWidget.value));
    };
}

app.registerExtension({
    name: "UniverSR.VideoNodes",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === "UniverSRLoadVideoAudio") {
            addVideoPreview(nodeType);
            addUploadWidget(nodeType);
        }
        if (nodeData?.name === "UniverSRLoadVideoAudioPath") {
            addVideoPreview(nodeType);
        }
        if (nodeData?.name === "UniverSRVideoCombiner") {
            addVideoPreview(nodeType);
        }
    },
});

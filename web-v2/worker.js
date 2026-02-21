// spoor — inference worker
// Runs ONNX Runtime Web in a Web Worker, off the main thread.

const ORT_VERSION = '1.21.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;
const MODEL_CACHE = 'spoor-models-v2';
let INPUT_SIZE = 640; // default to 640 for speed; 1280 for accuracy
const CONF_THRESHOLD = 0.1;
const IOU_THRESHOLD = 0.45;
const CATEGORIES = { 0: 'animal', 1: 'person', 2: 'vehicle' };

importScripts(`${ORT_CDN}/ort.min.js`);
ort.env.wasm.wasmPaths = `${ORT_CDN}/`;
ort.env.wasm.numThreads = navigator.hardwareConcurrency
    ? Math.min(navigator.hardwareConcurrency, 4)
    : 1;
ort.env.wasm.simd = true;

// Diagnostics
console.log('[spoor] SharedArrayBuffer available:', typeof SharedArrayBuffer !== 'undefined');
console.log('[spoor] Hardware concurrency:', navigator.hardwareConcurrency);
console.log('[spoor] WASM threads requested:', ort.env.wasm.numThreads);

let session = null;

// ── Model loading ──────────────────────────────────────────────

async function loadModel(modelUrl) {
    postMessage({ type: 'model-progress', stage: 'checking-cache' });

    const cache = await caches.open(MODEL_CACHE);
    let response = await cache.match(modelUrl);

    if (!response) {
        postMessage({ type: 'model-progress', stage: 'downloading', progress: 0 });
        const fetchResponse = await fetch(modelUrl);
        if (!fetchResponse.ok) throw new Error(`Model fetch failed: ${fetchResponse.status}`);

        // Stream the download so we can report progress
        const contentLength = +fetchResponse.headers.get('Content-Length') || 0;
        const reader = fetchResponse.body.getReader();
        const chunks = [];
        let received = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            received += value.length;
            if (contentLength > 0) {
                postMessage({ type: 'model-progress', stage: 'downloading', progress: received / contentLength });
            }
        }

        // Reassemble into a Response and cache it
        const blob = new Blob(chunks);
        response = new Response(blob, {
            headers: { 'Content-Type': 'application/octet-stream' }
        });
        await cache.put(modelUrl, response.clone());
    }

    postMessage({ type: 'model-progress', stage: 'creating-session' });

    const buffer = await response.arrayBuffer();
    session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm']
    });

    postMessage({ type: 'model-ready' });
}

// ── Preprocessing ──────────────────────────────────────────────
// Matches benchmark.py:load_and_preprocess() exactly.

function preprocess(imageData, origWidth, origHeight) {
    const targetH = INPUT_SIZE;
    const targetW = INPUT_SIZE;

    // Letterbox parameters
    const scale = Math.min(targetW / origWidth, targetH / origHeight);
    const newW = Math.trunc(origWidth * scale);
    const newH = Math.trunc(origHeight * scale);
    const padLeft = Math.trunc((targetW - newW) / 2);
    const padTop = Math.trunc((targetH - newH) / 2);

    // Resize with OffscreenCanvas (bilinear, like cv2.INTER_LINEAR)
    const srcCanvas = new OffscreenCanvas(origWidth, origHeight);
    const srcCtx = srcCanvas.getContext('2d');
    srcCtx.putImageData(imageData, 0, 0);

    // Letterbox canvas: fill gray, draw resized image centered
    const lbCanvas = new OffscreenCanvas(targetW, targetH);
    const lbCtx = lbCanvas.getContext('2d');
    lbCtx.fillStyle = 'rgb(114,114,114)';
    lbCtx.fillRect(0, 0, targetW, targetH);
    lbCtx.drawImage(srcCanvas, 0, 0, origWidth, origHeight, padLeft, padTop, newW, newH);

    // Extract RGBA, convert to CHW RGB float32
    const rgba = lbCtx.getImageData(0, 0, targetW, targetH).data;
    const numPixels = targetW * targetH;
    const chw = new Float32Array(3 * numPixels);

    for (let i = 0; i < numPixels; i++) {
        const ri = i * 4;
        chw[i]                  = rgba[ri]     / 255.0; // R
        chw[i + numPixels]      = rgba[ri + 1] / 255.0; // G
        chw[i + 2 * numPixels]  = rgba[ri + 2] / 255.0; // B
    }

    return { chw, scale, padLeft, padTop };
}

// ── Postprocessing ─────────────────────────────────────────────
// Matches benchmark.py:postprocess_yolo() + nms() exactly.

function computeIoU(a, b) {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const areaB = (b[2] - b[0]) * (b[3] - b[1]);
    const union = areaA + areaB - inter;
    return union > 0 ? inter / union : 0;
}

function nms(boxes, scores, iouThreshold) {
    if (boxes.length === 0) return [];

    const indices = Array.from(scores.keys());
    indices.sort((a, b) => scores[b] - scores[a]);

    const keep = [];
    const suppressed = new Set();

    for (const i of indices) {
        if (suppressed.has(i)) continue;
        keep.push(i);
        for (let j = indices.indexOf(i) + 1; j < indices.length; j++) {
            const k = indices[j];
            if (suppressed.has(k)) continue;
            if (computeIoU(boxes[i], boxes[k]) > iouThreshold) {
                suppressed.add(k);
            }
        }
    }
    return keep;
}

function postprocess(output) {
    const data = output.data; // flat Float32Array
    const numPreds = output.dims[1];
    const numCols = output.dims[2];

    // Filter by objectness, compute class scores
    const filtered = [];
    for (let i = 0; i < numPreds; i++) {
        const off = i * numCols;
        const objConf = data[off + 4];
        if (objConf <= CONF_THRESHOLD) continue;

        const cx = data[off], cy = data[off + 1], w = data[off + 2], h = data[off + 3];
        let bestCls = 0, bestScore = 0;
        for (let c = 0; c < 3; c++) {
            const s = objConf * data[off + 5 + c];
            if (s > bestScore) { bestScore = s; bestCls = c; }
        }

        filtered.push({
            box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            score: bestScore,
            cls: bestCls
        });
    }

    if (filtered.length === 0) return [];

    // NMS per class
    const detections = [];
    const classes = [...new Set(filtered.map(d => d.cls))];

    for (const cls of classes) {
        const items = filtered.filter(d => d.cls === cls);
        const boxes = items.map(d => d.box);
        const scores = items.map(d => d.score);
        const keep = nms(boxes, scores, IOU_THRESHOLD);
        for (const idx of keep) {
            detections.push({
                bbox: items[idx].box,
                confidence: items[idx].score,
                category: cls,
                categoryName: CATEGORIES[cls]
            });
        }
    }

    detections.sort((a, b) => b.confidence - a.confidence);
    return detections;
}

// ── Rescaling ──────────────────────────────────────────────────
// Matches evaluate.py:rescale_detections() — reverse letterbox transform.

function rescale(detections, scale, padLeft, padTop, origW, origH) {
    return detections.map(det => {
        let [x1, y1, x2, y2] = det.bbox;
        x1 = (x1 - padLeft) / scale;
        y1 = (y1 - padTop) / scale;
        x2 = (x2 - padLeft) / scale;
        y2 = (y2 - padTop) / scale;

        // Clamp
        x1 = Math.max(0, Math.min(x1, origW));
        y1 = Math.max(0, Math.min(y1, origH));
        x2 = Math.max(0, Math.min(x2, origW));
        y2 = Math.max(0, Math.min(y2, origH));

        // Normalized [x, y, w, h] for MegaDetector JSON export
        const bboxNorm = [
            x1 / origW,
            y1 / origH,
            (x2 - x1) / origW,
            (y2 - y1) / origH
        ];

        return { ...det, bbox: [x1, y1, x2, y2], bboxNorm };
    });
}

// ── Inference handler ──────────────────────────────────────────

async function infer(msg) {
    const { id, buffer, fileName, origWidth, origHeight } = msg;
    const t0 = performance.now();

    try {
        // Decode image from ArrayBuffer
        const blob = new Blob([buffer]);
        const bitmap = await createImageBitmap(blob);
        const tDecode = performance.now();

        // Get pixel data via OffscreenCanvas
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0);
        const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
        const w = bitmap.width;
        const h = bitmap.height;
        bitmap.close();

        // Preprocess
        const { chw, scale, padLeft, padTop } = preprocess(imageData, w, h);
        const tPreprocess = performance.now();

        // Run inference
        const inputTensor = new ort.Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
        const inputName = session.inputNames[0];
        const results = await session.run({ [inputName]: inputTensor });
        const output = results[session.outputNames[0]];
        const tInference = performance.now();

        // Postprocess
        let detections = postprocess(output);
        detections = rescale(detections, scale, padLeft, padTop, w, h);
        const tPost = performance.now();

        console.log(`[spoor] ${fileName}: decode=${Math.round(tDecode-t0)}ms preprocess=${Math.round(tPreprocess-tDecode)}ms inference=${Math.round(tInference-tPreprocess)}ms postprocess=${Math.round(tPost-tInference)}ms total=${Math.round(tPost-t0)}ms`);

        postMessage({
            type: 'result',
            id,
            fileName,
            origWidth: w,
            origHeight: h,
            detections,
            inferenceTimeMs: Math.round(tPost - t0)
        });
    } catch (err) {
        postMessage({ type: 'error', id, message: err.message });
    }
}

// ── Message handler ────────────────────────────────────────────

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'init') {
        if (msg.inputSize) INPUT_SIZE = msg.inputSize;
        try {
            await loadModel(msg.modelUrl);
        } catch (err) {
            postMessage({ type: 'error', id: null, message: `Model load failed: ${err.message}` });
        }
    } else if (msg.type === 'set-resolution') {
        INPUT_SIZE = msg.inputSize;
        console.log(`[spoor] Resolution set to ${INPUT_SIZE}x${INPUT_SIZE}`);
    } else if (msg.type === 'infer') {
        await infer(msg);
    }
};

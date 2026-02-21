// sieve — main thread
// UI, file handling, inference (native Rust in Tauri, WASM in browser), rendering, export.

const IS_TAURI = !!window.__TAURI_INTERNALS__;
const invoke = IS_TAURI ? window.__TAURI_INTERNALS__.invoke : null;
const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist';
const MODEL_URL = '/models/mdv5a_int8.onnx';
const MODEL_CACHE = 'spoor-models-v1';
const CONF_THRESHOLD = 0.1;
const IOU_THRESHOLD = 0.45;
const CATEGORIES = { 0: 'animal', 1: 'person', 2: 'vehicle' };
const CATEGORY_COLORS = { 0: '#c67b30', 1: '#4a6fa5', 2: '#6b8e6b' };

// Desktop defaults to 1280 (fast enough natively), web defaults to 640
let INPUT_SIZE = IS_TAURI ? 1280 : 640;
let MODEL_TYPE = 'thorough'; // 'quick' or 'thorough' — desktop only
let session = null;

// ── ONNX Runtime config (browser only) ─────────────────────────

if (!IS_TAURI && typeof ort !== 'undefined') {
    ort.env.wasm.wasmPaths = `${ORT_CDN}/`;
    ort.env.wasm.proxy = true;
    ort.env.wasm.numThreads = navigator.hardwareConcurrency
        ? Math.min(navigator.hardwareConcurrency, 4)
        : 1;
    ort.env.wasm.simd = true;
    console.log('[spoor] Browser mode, WASM threads:', ort.env.wasm.numThreads);
} else if (IS_TAURI) {
    console.log('[spoor] Desktop mode, native inference');
}

// ── State ──────────────────────────────────────────────────────

const state = {
    modelReady: false,
    queue: [],
    processing: false,
    results: [],
    totalImages: 0,
    processedImages: 0,
    sortBy: 'processing',
    displayThreshold: 0.2,
    visibleCategories: new Set(['animal', 'person', 'vehicle', 'empty']),
    folderName: null,
    batchActive: false,
    // Species classification
    speciesStatus: 'idle', // 'idle' | 'running' | 'done'
    speciesMap: [],        // request index → { result, detectionIndex }
    speciesTotal: 0,
    speciesProcessed: 0
};

// ── DOM refs ───────────────────────────────────────────────────

const sortControls = document.getElementById('sort-controls');
const filterControls = document.getElementById('filter-controls');
const thresholdControl = document.getElementById('threshold-control');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdValue = document.getElementById('threshold-value');
const exportCsvBtn = document.getElementById('export-csv-btn');
const speciesBtn = document.getElementById('species-btn');
const clearBtn = document.getElementById('clear-btn');
const folderBtn = document.getElementById('folder-btn');
const cancelBtn = document.getElementById('cancel-btn');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const statusCounts = document.getElementById('status-counts');
const resultsSection = document.getElementById('results');
const resultsSummary = document.getElementById('results-summary');
const resultsGrid = document.getElementById('results-grid');
const exportBtn = document.getElementById('export-btn');
const objectUrls = new Map();

// ── Model loading ──────────────────────────────────────────────

async function loadModel() {
    statusBar.hidden = false;

    const cache = await caches.open(MODEL_CACHE);
    let response = await cache.match(MODEL_URL);

    if (!response) {
        statusText.textContent = 'Downloading model (134 MB, one time only)\u2026';
        const fetchResponse = await fetch(MODEL_URL);
        if (!fetchResponse.ok) throw new Error(`Model fetch failed: ${fetchResponse.status}`);

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
                const pct = Math.round((received / contentLength) * 100);
                statusText.textContent = `Downloading model\u2026 ${pct}%`;
                progressFill.style.width = `${pct}%`;
            }
        }

        const blob = new Blob(chunks);
        response = new Response(blob, { headers: { 'Content-Type': 'application/octet-stream' } });
        await cache.put(MODEL_URL, response.clone());
    }

    statusText.textContent = 'Starting up\u2026';
    progressFill.style.width = '90%';

    const buffer = await response.arrayBuffer();

    session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm']
    });

    state.modelReady = true;
    statusBar.hidden = true;
    progressFill.style.width = '0%';
    console.log('[spoor] model ready');

    if (state.queue.length > 0 && !state.processing) processNext();
}

if (IS_TAURI) {
    // Model is loaded in Rust on startup — ready immediately
    state.modelReady = true;
    console.log('[spoor] native model ready');
} else {
    loadModel().catch(err => {
        console.error('Model load failed:', err);
        statusText.textContent = `Failed to load model: ${err.message}`;
    });
}

// ── Preprocessing ──────────────────────────────────────────────

function preprocess(img) {
    const origW = img.naturalWidth;
    const origH = img.naturalHeight;
    const targetH = INPUT_SIZE;
    const targetW = INPUT_SIZE;

    const scale = Math.min(targetW / origW, targetH / origH);
    const newW = Math.trunc(origW * scale);
    const newH = Math.trunc(origH * scale);
    const padLeft = Math.trunc((targetW - newW) / 2);
    const padTop = Math.trunc((targetH - newH) / 2);

    // Letterbox: gray pad, draw resized image centered
    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, targetW, targetH);
    ctx.drawImage(img, 0, 0, origW, origH, padLeft, padTop, newW, newH);

    const rgba = ctx.getImageData(0, 0, targetW, targetH).data;
    const numPixels = targetW * targetH;
    const chw = new Float32Array(3 * numPixels);

    for (let i = 0; i < numPixels; i++) {
        const ri = i * 4;
        chw[i]                  = rgba[ri]     / 255.0;
        chw[i + numPixels]      = rgba[ri + 1] / 255.0;
        chw[i + 2 * numPixels]  = rgba[ri + 2] / 255.0;
    }

    return { chw, scale, padLeft, padTop, origW, origH };
}

// ── Postprocessing ─────────────────────────────────────────────

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
            if (computeIoU(boxes[i], boxes[k]) > iouThreshold) suppressed.add(k);
        }
    }
    return keep;
}

function postprocess(output) {
    const data = output.data;
    const numPreds = output.dims[1];
    const numCols = output.dims[2];
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
            score: bestScore, cls: bestCls
        });
    }

    if (filtered.length === 0) return [];

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

function rescale(detections, scale, padLeft, padTop, origW, origH) {
    return detections.map(det => {
        let [x1, y1, x2, y2] = det.bbox;
        x1 = Math.max(0, Math.min((x1 - padLeft) / scale, origW));
        y1 = Math.max(0, Math.min((y1 - padTop) / scale, origH));
        x2 = Math.max(0, Math.min((x2 - padLeft) / scale, origW));
        y2 = Math.max(0, Math.min((y2 - padTop) / scale, origH));
        const bboxNorm = [x1 / origW, y1 / origH, (x2 - x1) / origW, (y2 - y1) / origH];
        return { ...det, bbox: [x1, y1, x2, y2], bboxNorm };
    });
}

// ── Inference ──────────────────────────────────────────────────

async function inferNative(id, file) {
    const t0 = performance.now();

    // Read file as bytes for Rust
    const buffer = await file.arrayBuffer();
    const bytes = Array.from(new Uint8Array(buffer));
    const tRead = performance.now();

    // Call Rust inference
    const result = await window.__TAURI_INTERNALS__.invoke('detect_image_bytes', {
        imageBytes: bytes,
        fileName: file.name,
        inputSize: INPUT_SIZE,
        model: MODEL_TYPE
    });
    const tInfer = performance.now();

    console.log(`[spoor] ${file.name}: read=${Math.round(tRead-t0)}ms native_inference=${result.inferenceTimeMs}ms ipc_total=${Math.round(tInfer-t0)}ms`);

    return {
        id,
        fileName: result.fileName,
        origWidth: result.origWidth,
        origHeight: result.origHeight,
        detections: result.detections,
        inferenceTimeMs: result.inferenceTimeMs
    };
}

async function inferWasm(id, file) {
    const t0 = performance.now();

    // Decode image
    const objectUrl = objectUrls.get(id);
    const img = new Image();
    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = objectUrl;
    });
    const tDecode = performance.now();

    // Preprocess
    const { chw, scale, padLeft, padTop, origW, origH } = preprocess(img);
    const tPreprocess = performance.now();

    // Run inference (async — proxy sends to ONNX's own worker)
    const inputTensor = new ort.Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const inputName = session.inputNames[0];
    const results = await session.run({ [inputName]: inputTensor });
    const output = results[session.outputNames[0]];
    const tInference = performance.now();

    // Postprocess
    let detections = postprocess(output);
    detections = rescale(detections, scale, padLeft, padTop, origW, origH);
    const tPost = performance.now();

    console.log(`[spoor] ${file.name}: decode=${Math.round(tDecode-t0)}ms preprocess=${Math.round(tPreprocess-tDecode)}ms inference=${Math.round(tInference-tPreprocess)}ms postprocess=${Math.round(tPost-tInference)}ms total=${Math.round(tPost-t0)}ms`);

    return {
        id, fileName: file.name, origWidth: origW, origHeight: origH,
        detections, inferenceTimeMs: Math.round(tPost - t0)
    };
}

const infer = IS_TAURI ? inferNative : inferWasm;

// ── File handling ──────────────────────────────────────────────

function addFiles(files) {
    const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
    if (imageFiles.length === 0) return;

    for (const file of imageFiles) {
        const id = crypto.randomUUID();
        objectUrls.set(id, URL.createObjectURL(file));
        state.queue.push({ id, file });
    }

    state.totalImages += imageFiles.length;
    statusBar.hidden = false;
    resultsSection.hidden = false;
    dropZone.classList.add('compact');
    updateProgress();

    if (state.modelReady && !state.processing) processNext();
}

async function processNext() {
    if (state.queue.length === 0) {
        state.processing = false;
        return;
    }

    state.processing = true;
    const item = state.queue.shift();

    try {
        const result = await infer(item.id, item.file);
        handleResult(result);
    } catch (err) {
        console.error('Inference error:', item.file.name, err);
        state.processedImages++;
        updateProgress();
    }
    processNext();
}

// ── Drop zone events ───────────────────────────────────────────

dropZone.addEventListener('click', async () => {
    if (IS_TAURI) {
        // Native file dialog — returns paths, no byte serialization overhead
        const result = await invoke('plugin:dialog|open', {
            options: {
                multiple: true,
                title: 'Select images',
                filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp', 'tiff', 'tif'] }]
            }
        });
        if (!result) return;
        const paths = Array.isArray(result) ? result : [result];
        if (paths.length === 0) return;

        const total = await invoke('process_paths', { paths, inputSize: INPUT_SIZE, model: MODEL_TYPE });
        if (total === 0) return;
        startBatch(total, paths.length === 1 ? paths[0].split('/').filter(Boolean).pop() : `${paths.length} files`);
    } else {
        fileInput.click();
    }
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) addFiles(fileInput.files);
    fileInput.value = '';
});

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

// ── Browser folder drop support ─────────────────────────────────

async function readAllEntries(reader) {
    const entries = [];
    while (true) {
        const batch = await new Promise((resolve, reject) =>
            reader.readEntries(resolve, reject));
        if (batch.length === 0) break;
        entries.push(...batch);
    }
    return entries;
}

async function collectFilesFromEntries(entries) {
    const files = [];
    for (const entry of entries) {
        if (entry.isFile) {
            const file = await new Promise(resolve => entry.file(resolve));
            if (file.type.startsWith('image/')) files.push(file);
        } else if (entry.isDirectory) {
            const reader = entry.createReader();
            const children = await readAllEntries(reader);
            const subFiles = await collectFilesFromEntries(children);
            files.push(...subFiles);
        }
    }
    return files;
}

async function handleDrop(e) {
    e.preventDefault();
    if (IS_TAURI) return; // Handled by native-drop event

    // Check for directories via webkitGetAsEntry
    const items = e.dataTransfer.items ? [...e.dataTransfer.items] : [];
    const entries = items.map(i => i.webkitGetAsEntry?.()).filter(Boolean);
    const hasDirectory = entries.some(ent => ent.isDirectory);

    if (hasDirectory) {
        const files = await collectFilesFromEntries(entries);
        if (files.length > 0) addFiles(files);
    } else if (e.dataTransfer.files.length > 0) {
        addFiles(e.dataTransfer.files);
    }
}

dropZone.addEventListener('drop', (e) => {
    dropZone.classList.remove('drag-over');
    handleDrop(e);
});

document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', handleDrop);

// ── Result rendering ───────────────────────────────────────────

function handleResult(result) {
    state.results.push(result);
    state.processedImages++;
    renderCard(result);
    updateProgress();
}

function visibleDetections(result) {
    return result.detections.filter(d => d.confidence >= state.displayThreshold);
}

function cardCategory(result) {
    const dets = visibleDetections(result);
    return dets.length > 0 ? dets[0].categoryName : 'empty';
}

function applyFilters() {
    for (const result of state.results) {
        const cat = cardCategory(result);
        result.cardElement.hidden = !state.visibleCategories.has(cat);
    }
    updateProgress();
}

function drawBboxes(result) {
    const canvas = result.bboxCanvas;
    const img = result.imgElement;
    if (!canvas || !img) return;

    const rect = img.getBoundingClientRect();
    const dw = rect.width, dh = rect.height;
    if (dw === 0 || dh === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = dw * dpr;
    canvas.height = dh * dpr;
    canvas.style.width = dw + 'px';
    canvas.style.height = dh + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const scaleX = dw / result.origWidth;
    const scaleY = dh / result.origHeight;

    for (const det of visibleDetections(result)) {
        const [x1, y1, x2, y2] = det.bbox;
        const color = CATEGORY_COLORS[det.category] || '#888';

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

        // Species name replaces "animal" on bbox label; detection confidence stays
        const displayName = det.species ? det.species.commonName : det.categoryName;
        const label = `${displayName} ${Math.round(det.confidence * 100)}%`;
        ctx.font = '11px system-ui';
        const tw = ctx.measureText(label).width;
        const lx = x1 * scaleX;
        const ly = y1 * scaleY - 16;
        ctx.fillStyle = color;
        ctx.fillRect(lx, ly, tw + 6, 16);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, lx + 3, ly + 12);
    }
}

function updateCard(result) {
    const dets = visibleDetections(result);
    const card = result.cardElement;
    card.classList.toggle('empty', dets.length === 0);
    const info = card.querySelector('.result-detections');
    if (info) info.textContent = summarizeDetections(result.detections, dets);
    drawBboxes(result);
}

function renderCard(result) {
    const objectUrl = objectUrls.get(result.id);
    const card = document.createElement('div');
    card.className = 'result-card';

    const dets = visibleDetections(result);
    if (dets.length === 0) card.classList.add('empty');

    const container = document.createElement('div');
    container.className = 'image-container';

    const img = document.createElement('img');
    img.src = objectUrl;
    img.alt = result.fileName;

    const bboxCanvas = document.createElement('canvas');
    bboxCanvas.className = 'bbox-overlay';

    container.appendChild(img);
    container.appendChild(bboxCanvas);

    result.imgElement = img;
    result.bboxCanvas = bboxCanvas;

    img.onload = () => {
        img.classList.add('loaded');
        drawBboxes(result);
    };

    const info = document.createElement('div');
    info.className = 'result-info';
    info.innerHTML = `
        <div class="result-filename">${result.fileName}</div>
        <div class="result-detections">${summarizeDetections(result.detections, dets)}</div>
        <div class="result-time">${result.inferenceTimeMs}ms</div>
    `;

    card.appendChild(container);
    card.appendChild(info);
    card.addEventListener('animationend', () => card.classList.add('settled'), { once: true });
    result.cardElement = card;
    resultsGrid.appendChild(card);
}

function summarizeDetections(allDetections, visibleDets) {
    if (visibleDets.length === 0) {
        return allDetections.length > 0 ? 'Below threshold' : 'No detections';
    }

    // If any detection has species data, use species-aware summary
    const hasSpecies = visibleDets.some(d => d.species);
    if (hasSpecies) {
        const parts = [];
        for (const d of visibleDets) {
            if (d.species) {
                let name = d.species.commonName;
                if (d.speciesRunnerUp) {
                    name += ` or ${d.speciesRunnerUp.commonName}`;
                }
                parts.push(name);
            } else {
                parts.push(d.categoryName);
            }
        }
        return parts.join(', ');
    }

    const counts = {};
    for (const d of visibleDets) counts[d.categoryName] = (counts[d.categoryName] || 0) + 1;
    return Object.entries(counts).map(([name, n]) => `${n} ${name}${n > 1 ? 's' : ''}`).join(', ');
}

function countVisibleAnimals() {
    let count = 0;
    for (const r of state.results) {
        count += visibleDetections(r).filter(d => d.category === 0).length;
    }
    return count;
}

function updateProgress() {
    const { processedImages, totalImages } = state;
    const done = processedImages >= totalImages && totalImages > 0;

    if (done) {
        statusText.textContent = 'Done';
        progressFill.style.width = '100%';
        cancelBtn.hidden = true;
        exportBtn.hidden = false;
        exportCsvBtn.hidden = false;
        clearBtn.hidden = false;
        sortControls.hidden = false;
        filterControls.hidden = false;
        thresholdControl.hidden = false;
        // Show species button if Tauri + animal detections exist + not already done
        if (IS_TAURI && state.speciesStatus === 'idle' && hasAnimalDetections()) {
            speciesBtn.hidden = false;
        }
    } else if (totalImages > 0) {
        const name = state.folderName ? ` \u2014 ${state.folderName}` : '';
        statusText.textContent = `Processing ${processedImages + 1} of ${totalImages}${name}\u2026`;
        progressFill.style.width = `${(processedImages / totalImages) * 100}%`;
        if (state.batchActive) cancelBtn.hidden = false;
    }

    const animalCount = countVisibleAnimals();
    const parts = [`${processedImages} of ${totalImages} images`];
    if (animalCount > 0) parts.push(`${animalCount} animal${animalCount !== 1 ? 's' : ''}`);
    if (state.speciesStatus === 'done') {
        parts.push(`${state.speciesProcessed} species identified`);
    }
    resultsSummary.textContent = parts.join(' \u00b7 ');

    if (done) {
        statusCounts.textContent = '';
    } else {
        statusCounts.textContent = parts.join(' \u00b7 ');
    }
}

// ── Sorting ─────────────────────────────────────────────────────

const sortComparators = {
    processing: null, // original order — uses index
    confidence: (a, b) => {
        const aConf = a.detections.length > 0 ? a.detections[0].confidence : 0;
        const bConf = b.detections.length > 0 ? b.detections[0].confidence : 0;
        return bConf - aConf;
    },
    filename: (a, b) => a.fileName.localeCompare(b.fileName, undefined, { numeric: true })
};

function reorderGrid() {
    const sorted = [...state.results];
    const cmp = sortComparators[state.sortBy];
    if (cmp) sorted.sort(cmp);
    for (const result of sorted) resultsGrid.appendChild(result.cardElement);
}

for (const btn of document.querySelectorAll('.sort-btn')) {
    btn.addEventListener('click', () => {
        for (const b of document.querySelectorAll('.sort-btn')) b.classList.remove('active');
        btn.classList.add('active');
        state.sortBy = btn.dataset.sort;
        reorderGrid();
    });
}

// ── Filtering ───────────────────────────────────────────────────

for (const btn of document.querySelectorAll('.filter-btn')) {
    btn.addEventListener('click', () => {
        const cat = btn.dataset.category;
        if (state.visibleCategories.has(cat)) {
            state.visibleCategories.delete(cat);
            btn.classList.remove('active');
        } else {
            state.visibleCategories.add(cat);
            btn.classList.add('active');
        }
        applyFilters();
    });
}

// ── Export: MegaDetector JSON v1.5 ─────────────────────────────

function exportJSON() {
    const isQuick = MODEL_TYPE === 'quick';
    const output = {
        info: {
            format_version: '1.5',
            detector: isQuick ? 'mdv6-yolov10-c' : 'mdv5a',
            detector_metadata: isQuick
                ? { megadetector_version: 'v6-yolov10-c', typical_detection_threshold: 0.2, conservative_detection_threshold: 0.05 }
                : { megadetector_version: 'v5a.0.0', typical_detection_threshold: 0.2, conservative_detection_threshold: 0.05 }
        },
        detection_categories: { '1': 'animal', '2': 'person', '3': 'vehicle' },
        images: state.results.map(r => {
            const dets = visibleDetections(r);
            return {
                file: r.fileName,
                max_detection_conf: dets.length > 0
                    ? parseFloat(dets[0].confidence.toFixed(4)) : 0,
                detections: dets.map(d => {
                    const det = {
                        category: String(d.category + 1),
                        conf: parseFloat(d.confidence.toFixed(4)),
                        bbox: d.bboxNorm
                    };
                    if (d.species) {
                        det.species = {
                            common_name: d.species.commonName,
                            scientific_name: d.species.scientificName,
                            conf: parseFloat(d.species.probability.toFixed(4)),
                            class: d.species.class,
                            order: d.species.order,
                            family: d.species.family
                        };
                    }
                    return det;
                })
            };
        })
    };

    const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'spoor_detections.json';
    a.click();
    URL.revokeObjectURL(url);
}

function exportCSV() {
    const hasAnySpecies = state.results.some(r =>
        r.detections.some(d => d.species));
    const header = hasAnySpecies
        ? 'file,max_confidence,category,num_detections,species'
        : 'file,max_confidence,category,num_detections';
    const rows = state.results.map(r => {
        const dets = visibleDetections(r);
        const maxConf = dets.length > 0
            ? dets[0].confidence.toFixed(4) : '0';
        const topCategory = dets.length > 0
            ? dets[0].categoryName : 'empty';
        const name = r.fileName.includes(',') ? `"${r.fileName}"` : r.fileName;
        const topSpecies = dets.find(d => d.species)?.species.commonName || '';
        const speciesCol = hasAnySpecies ? `,${topSpecies.includes(',') ? `"${topSpecies}"` : topSpecies}` : '';
        return `${name},${maxConf},${topCategory},${dets.length}${speciesCol}`;
    });

    const blob = new Blob([header + '\n' + rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'spoor_detections.csv';
    a.click();
    URL.revokeObjectURL(url);
}

exportBtn.addEventListener('click', exportJSON);
exportCsvBtn.addEventListener('click', exportCSV);

// ── Species identification (Tauri only) ────────────────────────

function hasAnimalDetections() {
    return state.results.some(r =>
        r.detections.some(d => d.category === 0 && d.confidence >= state.displayThreshold)
    );
}

function startSpeciesClassification() {
    // Build requests: every animal detection above threshold with a native path
    const requests = [];
    const map = [];

    for (let ri = 0; ri < state.results.length; ri++) {
        const result = state.results[ri];
        if (!result.nativePath) continue;
        for (let di = 0; di < result.detections.length; di++) {
            const det = result.detections[di];
            if (det.category !== 0 || det.confidence < state.displayThreshold) continue;
            requests.push({
                imagePath: result.nativePath,
                bbox: det.bbox
            });
            map.push({ resultIndex: ri, detectionIndex: di });
        }
    }

    if (requests.length === 0) return;

    state.speciesStatus = 'running';
    state.speciesMap = map;
    state.speciesTotal = requests.length;
    state.speciesProcessed = 0;

    speciesBtn.disabled = true;
    speciesBtn.textContent = 'Identifying\u2026';
    statusBar.hidden = false;
    statusText.textContent = `Identifying species 1 of ${requests.length}\u2026`;
    progressFill.style.width = '0%';

    invoke('classify_detections', { requests }).catch(err => {
        console.error('[spoor] Species classification failed:', err);
        state.speciesStatus = 'idle';
        speciesBtn.disabled = false;
        speciesBtn.textContent = 'Identify species';
        statusText.textContent = `Species identification failed: ${err}`;
    });
}

speciesBtn.addEventListener('click', startSpeciesClassification);

// ── Clear / reset ───────────────────────────────────────────────

function clearResults() {
    // Cancel any active batch first
    if (state.batchActive && IS_TAURI) {
        invoke('cancel_processing');
    }

    // Revoke blob URLs (browser mode only — Tauri uses asset protocol)
    for (const [id, url] of objectUrls) {
        if (url.startsWith('blob:')) URL.revokeObjectURL(url);
    }
    objectUrls.clear();

    // Cancel any in-progress species classification
    if (state.speciesStatus === 'running') {
        invoke('cancel_processing');
    }

    // Reset state
    state.results = [];
    state.queue = [];
    state.processing = false;
    state.totalImages = 0;
    state.processedImages = 0;
    state.sortBy = 'processing';
    state.folderName = null;
    state.batchActive = false;
    state.speciesStatus = 'idle';
    state.speciesMap = [];
    state.speciesTotal = 0;
    state.speciesProcessed = 0;

    // Clear DOM
    resultsGrid.innerHTML = '';
    resultsSection.hidden = true;
    statusBar.hidden = true;
    dropZone.classList.remove('compact');
    progressFill.style.width = '0%';
    statusText.textContent = '';
    statusCounts.textContent = '';
    resultsSummary.textContent = '';

    // Hide controls
    sortControls.hidden = true;
    filterControls.hidden = true;
    thresholdControl.hidden = true;
    exportBtn.hidden = true;
    exportCsvBtn.hidden = true;
    speciesBtn.hidden = true;
    speciesBtn.disabled = false;
    speciesBtn.textContent = 'Identify species';
    clearBtn.hidden = true;
    cancelBtn.hidden = true;
    folderBtn.disabled = false;

    // Reset sort buttons
    for (const btn of document.querySelectorAll('.sort-btn')) {
        btn.classList.toggle('active', btn.dataset.sort === 'processing');
    }
}

clearBtn.addEventListener('click', clearResults);

// ── Model toggle (desktop only) ─────────────────────────────────

const modelToggle = document.getElementById('model-toggle');
const modelQuickBtn = document.getElementById('model-quick');
const modelThoroughBtn = document.getElementById('model-thorough');
const resolutionToggle = document.getElementById('resolution-toggle');

if (IS_TAURI) {
    modelToggle.hidden = false;
}

modelQuickBtn.addEventListener('click', () => {
    modelQuickBtn.classList.add('active');
    modelThoroughBtn.classList.remove('active');
    MODEL_TYPE = 'quick';
    INPUT_SIZE = 1280; // Quick always runs at 1280 — it's fast enough
    resolutionToggle.hidden = true;
});

modelThoroughBtn.addEventListener('click', () => {
    modelThoroughBtn.classList.add('active');
    modelQuickBtn.classList.remove('active');
    MODEL_TYPE = 'thorough';
    resolutionToggle.hidden = false;
    // Restore resolution from toggle state
    INPUT_SIZE = resFast.classList.contains('active') ? 640 : 1280;
});

// ── Resolution toggle ──────────────────────────────────────────

const resFast = document.getElementById('res-fast');
const resAccurate = document.getElementById('res-accurate');

// Desktop defaults to Accurate (1280) since native is fast enough
if (IS_TAURI) {
    resFast.classList.remove('active');
    resAccurate.classList.add('active');
}

resFast.addEventListener('click', () => {
    resFast.classList.add('active');
    resAccurate.classList.remove('active');
    INPUT_SIZE = 640;
});

resAccurate.addEventListener('click', () => {
    resAccurate.classList.add('active');
    resFast.classList.remove('active');
    INPUT_SIZE = 1280;
});

// ── Confidence threshold ────────────────────────────────────────

thresholdSlider.addEventListener('input', () => {
    state.displayThreshold = parseFloat(thresholdSlider.value);
    thresholdValue.textContent = Math.round(state.displayThreshold * 100) + '%';
    for (const result of state.results) updateCard(result);
    applyFilters();
});

// ── Batch helpers ───────────────────────────────────────────────

function startBatch(total, folderName) {
    state.totalImages += total;
    state.batchActive = true;
    state.folderName = folderName;
    statusBar.hidden = false;
    resultsSection.hidden = false;
    dropZone.classList.add('compact');
    folderBtn.disabled = true;
    updateProgress();
}

function endBatch() {
    state.batchActive = false;
    state.folderName = null;
    cancelBtn.hidden = true;
    folderBtn.disabled = false;
}

// ── Folder selection & batch events (Tauri only) ────────────────

if (IS_TAURI) {
    // Helper to listen for Tauri events without npm packages
    function tauriListen(event, handler) {
        const id = window.__TAURI_INTERNALS__.transformCallback((e) => handler(e.payload));
        return invoke('plugin:event|listen', {
            event,
            target: { kind: 'Any' },
            handler: id
        });
    }

    folderBtn.hidden = false;

    // ── Folder button ───────────────────────────────────────────
    folderBtn.addEventListener('click', async () => {
        const folder = await invoke('plugin:dialog|open', {
            options: { directory: true, title: 'Select image folder' }
        });
        if (!folder) return;

        const folderName = folder.split('/').filter(Boolean).pop() || folder;
        const total = await invoke('process_folder', {
            folderPath: folder,
            inputSize: INPUT_SIZE,
            model: MODEL_TYPE
        });
        if (total === 0) return;

        startBatch(total, folderName);
    });

    // ── Cancel button ───────────────────────────────────────────
    cancelBtn.addEventListener('click', async () => {
        await invoke('cancel_processing');
    });

    // ── Native drag-and-drop (files & folders with paths) ───────
    tauriListen('native-drop', async (payload) => {
        const paths = payload.paths;
        if (!paths?.length || state.batchActive) return;

        const total = await invoke('process_paths', {
            paths,
            inputSize: INPUT_SIZE,
            model: MODEL_TYPE
        });
        if (total === 0) return;

        // Derive a display name from the dropped items
        let name;
        if (paths.length === 1) {
            name = paths[0].split('/').filter(Boolean).pop() || null;
        } else {
            name = `${paths.length} items`;
        }
        startBatch(total, name);
    });

    // ── Batch events ────────────────────────────────────────────
    tauriListen('folder-result', (payload) => {
        const result = payload.result;
        result.id = crypto.randomUUID();
        result.nativePath = payload.filePath; // needed for species classification
        const imageUrl = window.__TAURI_INTERNALS__.convertFileSrc(payload.filePath);
        objectUrls.set(result.id, imageUrl);
        handleResult(result);
    });

    tauriListen('folder-error', (payload) => {
        console.error('[spoor] Failed:', payload.filePath, payload.error);
        state.processedImages++;
        updateProgress();
    });

    tauriListen('folder-complete', (payload) => {
        console.log('[spoor] Folder complete:', payload.total, 'images');
        endBatch();
    });

    tauriListen('folder-cancelled', () => {
        const skipped = state.totalImages - state.processedImages;
        state.totalImages = state.processedImages;
        endBatch();

        progressFill.style.width = '100%';
        statusText.textContent = `Cancelled \u2014 ${skipped} image${skipped !== 1 ? 's' : ''} skipped`;

        if (state.results.length > 0) {
            exportBtn.hidden = false;
            exportCsvBtn.hidden = false;
            clearBtn.hidden = false;
            sortControls.hidden = false;
            filterControls.hidden = false;
            thresholdControl.hidden = false;
            if (state.speciesStatus === 'idle' && hasAnimalDetections()) {
                speciesBtn.hidden = false;
            }
        }
    });

    // ── Species classification events ───────────────────────────
    tauriListen('species-result', (payload) => {
        const { index, result } = payload;
        const mapping = state.speciesMap[index];
        if (!mapping) return;

        const det = state.results[mapping.resultIndex].detections[mapping.detectionIndex];
        const top = result.predictions[0];

        // Attach species data to the detection
        det.species = {
            commonName: top.commonName,
            scientificName: top.scientificName,
            probability: top.probability,
            class: top.class,
            order: top.order,
            family: top.family
        };

        // If uncertain, store runner-up
        if (top.probability < 0.8 && result.predictions.length > 1) {
            det.speciesRunnerUp = {
                commonName: result.predictions[1].commonName,
                scientificName: result.predictions[1].scientificName,
                probability: result.predictions[1].probability
            };
        }

        state.speciesProcessed++;
        const pct = (state.speciesProcessed / state.speciesTotal) * 100;
        progressFill.style.width = `${pct}%`;
        statusText.textContent = state.speciesProcessed < state.speciesTotal
            ? `Identifying species ${state.speciesProcessed + 1} of ${state.speciesTotal}\u2026`
            : 'Finishing species identification\u2026';

        // Update the card
        const cardResult = state.results[mapping.resultIndex];
        updateCard(cardResult);
    });

    tauriListen('species-error', (payload) => {
        console.error('[spoor] Species error:', payload.imagePath, payload.error);
        state.speciesProcessed++;
        const pct = (state.speciesProcessed / state.speciesTotal) * 100;
        progressFill.style.width = `${pct}%`;
    });

    tauriListen('species-complete', () => {
        state.speciesStatus = 'done';
        speciesBtn.textContent = 'Species identified';
        speciesBtn.disabled = true;
        statusBar.hidden = true;
        updateProgress(); // refresh resultsSummary with species count
    });

    tauriListen('species-cancelled', () => {
        state.speciesStatus = 'idle';
        speciesBtn.disabled = false;
        speciesBtn.textContent = 'Identify species';
        statusText.textContent = 'Species identification cancelled';
    });
}

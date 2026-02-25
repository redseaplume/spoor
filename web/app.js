// spoor — main thread
// UI, file handling, inference (native Rust in Tauri, WASM in browser), rendering, export.

const IS_TAURI = !!window.__TAURI_INTERNALS__;
const invoke = IS_TAURI ? window.__TAURI_INTERNALS__.invoke : null;
const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist';
const MODEL_URL = '../models/mdv6-yolov10-c.onnx';
const MODEL_CACHE = 'spoor-models-v2';
const CONF_THRESHOLD = 0.1;
const IOU_THRESHOLD = 0.45;
const CATEGORIES = { 0: 'animal', 1: 'person', 2: 'vehicle' };
const CATEGORY_COLORS = { 0: '#5a7a52', 1: '#4a6a9a', 2: '#b06a28' };

// Redraw bbox canvases when image containers resize
const bboxObserver = new ResizeObserver((entries) => {
    if (bboxObserver._rafPending) return;
    bboxObserver._rafPending = true;
    requestAnimationFrame(() => {
        bboxObserver._rafPending = false;
        for (const entry of entries) {
            const result = entry.target._spoorResult;
            if (result) drawBboxes(result);
        }
    });
});

// MDv6 is fast enough at 1280 for both native and web
let INPUT_SIZE = 1280;
let MODEL_TYPE = IS_TAURI ? 'thorough' : 'quick'; // Browser always uses MDv6
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
    folderPath: null,
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
const confidenceGroup = document.querySelector('[data-testid="confidence-group"]');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdValue = document.getElementById('threshold-value');
const exportCsvBtn = document.getElementById('export-csv-btn');
const speciesBtn = document.getElementById('species-btn');
const clearBtn = document.getElementById('clear-btn');
const cancelBtn = document.getElementById('cancel-btn');
const dropZone = document.getElementById('drop-zone');
const dropInner = document.getElementById('drop-inner');
const dropText = document.getElementById('drop-text');
const dropZonePrimary = dropZone.querySelector('.drop-zone-primary');
const dropZoneSecondary = dropZone.querySelector('.drop-zone-secondary');
const convergenceContainer = document.getElementById('convergence');
const fileInput = document.getElementById('file-input');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const countAnimal = document.getElementById('count-animal');
const countPerson = document.getElementById('count-person');
const countVehicle = document.getElementById('count-vehicle');
const countEmpty = document.getElementById('count-empty');
const labelAnimal = document.getElementById('label-animal');
const labelPerson = document.getElementById('label-person');
const labelVehicle = document.getElementById('label-vehicle');
const resultsSection = document.getElementById('results');
const resultsSummary = document.getElementById('results-summary');
const resultsGrid = document.getElementById('results-grid');
const exportBtn = document.getElementById('export-btn');
const exportImagesBtn = document.getElementById('export-images-btn');
const objectUrls = new Map();

// ── Desktop-only DOM refs ─────────────────────────────────────────
const procRow = document.getElementById('proc-row');
const convCore = document.getElementById('conv-core');
const procText = document.getElementById('proc-text');

// ── Convergence particles ───────────────────────────────────────

const PARTICLE_COUNT = 28;
for (let i = 0; i < PARTICLE_COUNT; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const angle = Math.random() * Math.PI * 2;
    const dist = IS_TAURI ? 18 + Math.random() * 14 : 35 + Math.random() * 25;
    const dur = IS_TAURI ? 1.2 + Math.random() * 0.8 : 1.8 + Math.random() * 1.2;
    p.style.cssText = `
        --sx: ${Math.cos(angle) * dist}px;
        --sy: ${Math.sin(angle) * dist}px;
        --delay: ${IS_TAURI ? -Math.random() * 2 : Math.random() * 2.5}s;
        --dur: ${dur}s;
    `;
    convergenceContainer.appendChild(p);
}

let convergenceTimers = [];

function clearConvergenceTimers() {
    for (const id of convergenceTimers) clearTimeout(id);
    convergenceTimers = [];
}

function startConvergence() {
    clearConvergenceTimers();

    if (IS_TAURI) {
        // Desktop: fade out drop zone, show proc-row with convergence
        dropZone.classList.add('hidden');
        procRow.classList.add('visible');
        convergenceContainer.classList.remove('fading');
        convergenceTimers.push(
            setTimeout(() => {
                convCore.classList.add('pulsing');
                convergenceContainer.classList.add('active');
            }, 200)
        );
    } else {
        // Web: existing behavior
        dropText.classList.add('hidden');
        convergenceContainer.classList.remove('fading');
        convergenceTimers.push(
            setTimeout(() => dropZone.classList.add('processing'), 100),
            setTimeout(() => {
                dropInner.classList.add('pulsing');
                convergenceContainer.classList.add('active');
            }, 700)
        );
    }
}

function stopConvergence() {
    clearConvergenceTimers();

    dropInner?.classList.remove('pulsing');
    if (IS_TAURI) convCore?.classList.remove('pulsing');
    convergenceContainer.classList.add('fading');

    convergenceTimers.push(
        setTimeout(() => {
            convergenceContainer.classList.remove('active', 'fading');
            if (IS_TAURI) {
                // Desktop: proc-row stays visible, just stop animation
            } else {
                dropZone.classList.remove('processing');
                convergenceTimers.push(
                    setTimeout(() => dropText.classList.remove('hidden'), 650)
                );
            }
        }, 450)
    );
}

// ── Model loading ──────────────────────────────────────────────

async function loadModel() {
    const cache = await caches.open(MODEL_CACHE);
    let response = await cache.match(MODEL_URL);

    if (!response) {
        let showProgress = false;
        const slowTimer = setTimeout(() => {
            showProgress = true;
            dropZonePrimary.textContent = 'Downloading model\u2026';
            dropZoneSecondary.textContent = '10 MB, one time only';
            dropZoneSecondary.style.display = '';
        }, 400);

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
            if (showProgress && contentLength > 0) {
                const pct = Math.min(100, Math.round((received / contentLength) * 100));
                dropZonePrimary.textContent = `Downloading model\u2026 ${pct}%`;
            }
        }

        clearTimeout(slowTimer);
        const blob = new Blob(chunks);
        response = new Response(blob, { headers: { 'Content-Type': 'application/octet-stream' } });
        await cache.put(MODEL_URL, response.clone());
    }

    const buffer = await response.arrayBuffer();

    session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm']
    });

    state.modelReady = true;
    statusText.textContent = 'Ready';
    console.log('[spoor] model ready');

    if (state.queue.length > 0 && !state.processing) processNext();
}

if (IS_TAURI) {
    // Model is loaded in Rust on startup — ready immediately
    state.modelReady = true;
    statusText.textContent = 'Ready';
    console.log('[spoor] native model ready');
} else {
    loadModel().catch(err => {
        console.error('Model load failed:', err);
        dropZonePrimary.textContent = 'Failed to load model';
        dropZoneSecondary.textContent = err.message;
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

// MDv6 YOLOv10-c: NMS-free, output is [1, 300, 6] = [x1, y1, x2, y2, conf, class]
function postprocessV6(output) {
    const data = output.data;
    const numPreds = output.dims[1];
    const detections = [];

    for (let i = 0; i < numPreds; i++) {
        const off = i * 6;
        const conf = data[off + 4];
        if (conf <= CONF_THRESHOLD) continue;

        const cls = Math.round(data[off + 5]);
        detections.push({
            bbox: [data[off], data[off + 1], data[off + 2], data[off + 3]],
            confidence: conf,
            category: cls,
            categoryName: CATEGORIES[cls]
        });
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

    // Postprocess — MDv6 is NMS-free
    let detections = postprocessV6(output);
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
    if (!IS_TAURI) statusBar.classList.add('visible');
    resultsSection.hidden = false;
    startConvergence();
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
        result.file = item.file; // retain for image export
        handleResult(result);
    } catch (err) {
        console.error('Inference error:', item.file.name, err);
        state.processedImages++;
        updateProgress();
    }
    processNext();
}

// ── Drop zone events ───────────────────────────────────────────

dropZone.addEventListener('click', async (e) => {
    // Only respond to clicks on the drop zone inner area, not the convergence container
    const isProcessing = IS_TAURI ? dropZone.classList.contains('hidden') : dropZone.classList.contains('processing');
    if (isProcessing) return;

    if (IS_TAURI) {
        // Folder picker — drag-and-drop covers files, click = folder
        const folder = await invoke('plugin:dialog|open', {
            options: { directory: true, title: 'Select image folder' }
        });
        if (!folder) return;

        state.folderPath = folder;
        const folderName = folder.split('/').filter(Boolean).pop() || folder;
        const total = await invoke('process_paths', { paths: [folder], inputSize: INPUT_SIZE, model: MODEL_TYPE });
        if (total === 0) return;
        startBatch(total, folderName);
    } else {
        fileInput.click();
    }
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) addFiles(fileInput.files);
    fileInput.value = '';
});

const dragTarget = dropZone;
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dragTarget.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dragTarget.classList.remove('drag-over'));

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
    dragTarget.classList.remove('drag-over');
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
    return result.detections.length > 0 ? result.detections[0].categoryName : 'empty';
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

function buildExpandedContent(result) {
    const dets = visibleDetections(result);
    if (dets.length === 0) return '';

    let html = '';
    for (let i = 0; i < dets.length; i++) {
        const det = dets[i];

        // Species info (when available)
        if (det.species) {
            html += `<div class="expanded-row">
                <span class="expanded-label">Species</span>
                <span class="expanded-value"><span class="species-tag">${det.species.commonName} <span class="species-prob">${det.species.probability.toFixed(2)}</span></span></span>
            </div>`;
            const taxParts = [det.species.class, det.species.order, det.species.family].filter(Boolean);
            if (taxParts.length > 0) {
                html += `<div class="expanded-row">
                    <span class="expanded-label">Taxonomy</span>
                    <span class="expanded-value">${taxParts.join(' \u00b7 ')}</span>
                </div>`;
            }
        }

        // Box coordinates + confidence
        const boxLabel = dets.length > 1 ? `Box ${i + 1}` : 'Box';
        if (det.bboxNorm) {
            const bboxStr = det.bboxNorm.map(v => v.toFixed(2)).join(', ');
            html += `<div class="expanded-row">
                <span class="expanded-label">${boxLabel}</span>
                <span class="expanded-value">[${bboxStr}] \u00b7 ${(det.confidence * 100).toFixed(1)}% ${det.categoryName}</span>
            </div>`;
        }
    }

    return html;
}

function updateCard(result) {
    const dets = visibleDetections(result);
    const card = result.cardElement;
    card.classList.toggle('empty', dets.length === 0);
    // Update category class for left border
    card.classList.remove('cat-animal', 'cat-person', 'cat-vehicle', 'cat-empty');
    card.classList.add(`cat-${cardCategory(result)}`);
    const info = card.querySelector('.result-detections');
    if (info) info.textContent = summarizeDetections(result.detections, dets);
    drawBboxes(result);
    // Update expandable state and content
    card.classList.toggle('expandable', dets.length > 0);
    if (dets.length === 0) card.classList.remove('expanded');
    const expandedDiv = card.querySelector('.result-expanded');
    if (expandedDiv) expandedDiv.innerHTML = buildExpandedContent(result);
}

function renderCard(result) {
    const objectUrl = objectUrls.get(result.id);
    const card = document.createElement('div');
    card.className = 'result-card';

    const dets = visibleDetections(result);
    if (dets.length === 0) card.classList.add('empty');

    // Category class for left border color
    const cat = cardCategory(result);
    card.classList.add(`cat-${cat}`);

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

    container._spoorResult = result;
    bboxObserver.observe(container);

    img.onload = () => {
        img.classList.add('loaded');
        drawBboxes(result);
    };

    const info = document.createElement('div');
    info.className = 'result-info';
    info.innerHTML = `
        <div class="result-filename">${result.fileName}</div>
        <div class="result-meta">
            <span class="result-detections">${summarizeDetections(result.detections, dets)}</span>
            <span class="result-time">${result.inferenceTimeMs.toLocaleString()}ms</span>
        </div>
    `;

    // Expanded detail panel
    const expanded = document.createElement('div');
    expanded.className = 'result-expanded';
    expanded.innerHTML = buildExpandedContent(result);
    if (dets.length > 0) card.classList.add('expandable');

    card.appendChild(container);
    card.appendChild(info);
    card.appendChild(expanded);
    card.addEventListener('click', () => {
        if (card.classList.contains('expandable')) {
            card.classList.toggle('expanded');
        }
    });
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

function updateCategoryCounts() {
    const counts = { animal: 0, person: 0, vehicle: 0, empty: 0 };
    for (const r of state.results) {
        const vis = visibleDetections(r);
        counts.animal += vis.filter(d => d.category === 0).length;
        counts.person += vis.filter(d => d.category === 1).length;
        counts.vehicle += vis.filter(d => d.category === 2).length;
        if (vis.length === 0) counts.empty++;
    }
    countAnimal.textContent = counts.animal;
    countPerson.textContent = counts.person;
    countVehicle.textContent = counts.vehicle;
    countEmpty.textContent = counts.empty;
    if (labelAnimal) labelAnimal.textContent = counts.animal === 1 ? 'animal' : 'animals';
    if (labelPerson) labelPerson.textContent = counts.person === 1 ? 'person' : 'people';
    if (labelVehicle) labelVehicle.textContent = counts.vehicle === 1 ? 'vehicle' : 'vehicles';
}

function updateProgress() {
    const { processedImages, totalImages } = state;
    const done = processedImages >= totalImages && totalImages > 0;

    if (done) {
        const doneText = state.speciesStatus === 'done'
            ? `Done \u00b7 ${state.speciesProcessed} species identified`
            : 'Done';
        statusText.textContent = doneText;
        cancelBtn.classList.add('hidden');
        exportBtn.hidden = false;
        exportCsvBtn.hidden = false;
        exportImagesBtn.hidden = false;
        clearBtn.hidden = false;
        sortControls.hidden = false;
        filterControls.hidden = false;
        confidenceGroup.hidden = false;
        stopConvergence();
        if (IS_TAURI) {
            if (procText) procText.textContent = state.speciesStatus === 'done'
                ? `Done \u2014 ${totalImages} images \u00b7 ${state.speciesProcessed} species identified`
                : `Done \u2014 ${totalImages} images`;
            cancelBtn.style.display = 'none';
            if (state.speciesStatus === 'idle' && hasAnimalDetections()) {
                speciesBtn.hidden = false;
            }
        }
    } else if (totalImages > 0) {
        const name = state.folderName ? ` \u2014 ${state.folderName}` : '';
        const progressMsg = `Processing ${processedImages + 1} of ${totalImages}${name}\u2026`;
        statusText.textContent = progressMsg;
        if (IS_TAURI && procText) procText.textContent = progressMsg;
        if (state.batchActive) cancelBtn.classList.remove('hidden');
    }

    updateCategoryCounts();

    const animalCount = countVisibleAnimals();
    const parts = [`${processedImages} of ${totalImages} images`];
    if (animalCount > 0) parts.push(`${animalCount} animal${animalCount !== 1 ? 's' : ''}`);
    resultsSummary.textContent = parts.join(' \u00b7 ');
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

async function exportJSON() {
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
            const dets = r.detections;
            return {
                file: IS_TAURI ? r.nativePath : r.fileName,
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

    const json = JSON.stringify(output, null, 2);

    if (IS_TAURI) {
        const path = await invoke('plugin:dialog|save', {
            options: {
                title: 'Save detections as JSON',
                defaultPath: 'spoor_detections.json',
                filters: [{ name: 'JSON', extensions: ['json'] }],
                canCreateDirectories: true
            }
        });
        if (!path) return;
        await invoke('write_text_file', { path, content: json });
    } else {
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'spoor_detections.json';
        a.click();
        URL.revokeObjectURL(url);
    }
}

async function exportCSV() {
    const hasAnySpecies = state.results.some(r =>
        r.detections.some(d => d.species));
    const header = hasAnySpecies
        ? 'file,confidence,category,bbox_x,bbox_y,bbox_w,bbox_h,species'
        : 'file,confidence,category,bbox_x,bbox_y,bbox_w,bbox_h';
    const csvQuote = s => s.includes(',') || s.includes('"') ? `"${s.replace(/"/g, '""')}"` : s;
    const rows = [];
    for (const r of state.results) {
        const file = IS_TAURI ? r.nativePath : r.fileName;
        const name = csvQuote(file);
        const dets = r.detections;
        if (dets.length === 0) {
            rows.push(hasAnySpecies
                ? `${name},0,empty,0,0,0,0,`
                : `${name},0,empty,0,0,0,0`);
        } else {
            for (const d of dets) {
                const [bx, by, bw, bh] = d.bboxNorm;
                const species = d.species ? csvQuote(d.species.commonName) : '';
                const speciesCol = hasAnySpecies ? `,${species}` : '';
                rows.push(`${name},${d.confidence.toFixed(4)},${d.categoryName || 'unknown'},${bx.toFixed(6)},${by.toFixed(6)},${bw.toFixed(6)},${bh.toFixed(6)}${speciesCol}`);
            }
        }
    }

    const csv = header + '\n' + rows.join('\n');

    if (IS_TAURI) {
        const path = await invoke('plugin:dialog|save', {
            options: {
                title: 'Save detections as CSV',
                defaultPath: 'spoor_detections.csv',
                filters: [{ name: 'CSV', extensions: ['csv'] }],
                canCreateDirectories: true
            }
        });
        if (!path) return;
        await invoke('write_text_file', { path, content: csv });
    } else {
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'spoor_detections.csv';
        a.click();
        URL.revokeObjectURL(url);
    }
}

exportBtn.addEventListener('click', exportJSON);
exportCsvBtn.addEventListener('click', exportCSV);

// ── Export: Annotated images ───────────────────────────────────

async function exportImages() {
    if (IS_TAURI) {
        await exportImagesTauri();
    } else {
        await exportImagesWeb();
    }
}

async function exportImagesTauri() {
    // Pick output directory
    const outputDir = await invoke('plugin:dialog|open', {
        options: { directory: true, title: 'Choose output folder for annotated images' }
    });
    if (!outputDir) return;

    const items = state.results
        .filter(r => r.nativePath && !r.cardElement.hidden)
        .map(r => ({
            filePath: r.nativePath,
            detections: visibleDetections(r).map(d => ({
                bbox: d.bboxNorm,
                confidence: d.confidence,
                category: d.categoryName || CATEGORIES[d.category] || 'unknown',
                species: d.species ? d.species.commonName : null
            }))
        }));

    if (items.length === 0) return;

    exportImagesBtn.disabled = true;

    let total;
    try {
        total = await invoke('export_images', {
            outputDir,
            items,
            baseDir: state.folderPath || null
        });
    } catch (e) {
        console.error('[spoor] Export invoke failed:', e);
        exportImagesBtn.disabled = false;
        return;
    }

    if (total === 0) {
        exportImagesBtn.disabled = false;
    }
    // Progress and completion handled by event listeners (set up in Tauri block below)
}

async function exportImagesWeb() {
    const results = state.results
        .filter(r => r.file && !r.cardElement.hidden)
        .map(r => ({
            file: r.file,
            fileName: r.fileName,
            origWidth: r.origWidth,
            origHeight: r.origHeight,
            detections: visibleDetections(r).map(d => ({
                bbox: d.bbox,
                confidence: d.confidence,
                category: d.category,
                categoryName: d.categoryName,
                species: d.species ? { commonName: d.species.commonName } : null
            }))
        }));

    if (results.length === 0) return;

    exportImagesBtn.disabled = true;

    const worker = new Worker('export-worker.js');
    worker.onmessage = (e) => {
        const msg = e.data;
        if (msg.type === 'progress') {
            // progress shown in status bar, button stays stable
        } else if (msg.type === 'done') {
            const url = URL.createObjectURL(msg.blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = state.folderName
                ? `spoor_${state.folderName}.zip`
                : 'spoor_images.zip';
            a.click();
            URL.revokeObjectURL(url);

            exportImagesBtn.disabled = false;
            worker.terminate();
        } else if (msg.type === 'error') {
            console.error('[spoor] Image export failed:', msg.error);
            exportImagesBtn.disabled = false;
            worker.terminate();
        }
    };

    worker.postMessage({ results });
}

exportImagesBtn.addEventListener('click', exportImages);

// ── Species identification (Tauri only) ────────────────────────

function hasAnimalDetections() {
    return state.results.some(r =>
        r.detections.some(d => d.category === 0 && d.confidence >= state.displayThreshold)
    );
}

function startSpeciesClassification() {
    // Build requests: every animal detection with a native path
    const requests = [];
    const map = [];

    for (let ri = 0; ri < state.results.length; ri++) {
        const result = state.results[ri];
        if (!result.nativePath) continue;
        for (let di = 0; di < result.detections.length; di++) {
            const det = result.detections[di];
            if (det.category !== 0) continue;
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
    if (!IS_TAURI) statusBar.classList.add('visible');
    statusText.textContent = `Identifying species 1 of ${requests.length}\u2026`;
    if (IS_TAURI && procText) procText.textContent = `Identifying species 1 of ${requests.length}\u2026`;
    startConvergence();

    invoke('classify_detections', { requests }).catch(err => {
        console.error('[spoor] Species classification failed:', err);
        state.speciesStatus = 'idle';
        speciesBtn.disabled = false;
        statusText.textContent = `Species identification failed: ${err}`;
    });
}

speciesBtn.addEventListener('click', startSpeciesClassification);

// ── Clear / reset ───────────────────────────────────────────────

function resetState() {
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
    state.folderPath = null;
    state.batchActive = false;
    state.speciesStatus = 'idle';
    state.speciesMap = [];
    state.speciesTotal = 0;
    state.speciesProcessed = 0;

    // Stop observing resizes
    bboxObserver.disconnect();

    // Reset sort buttons
    for (const btn of document.querySelectorAll('.sort-btn')) {
        btn.classList.toggle('active', btn.dataset.sort === 'processing');
    }
}

function resetDOM() {
    resultsGrid.innerHTML = '';
    resultsSection.hidden = true;
    resultsSection.classList.remove('fading');
    resultsSummary.textContent = '';

    clearConvergenceTimers();
    convergenceContainer.classList.remove('active', 'fading');
    dropInner?.classList.remove('pulsing');
    if (IS_TAURI) convCore?.classList.remove('pulsing');

    countAnimal.textContent = '0';
    countPerson.textContent = '0';
    countVehicle.textContent = '0';
    countEmpty.textContent = '0';
    if (labelAnimal) labelAnimal.textContent = 'animals';
    if (labelPerson) labelPerson.textContent = 'people';
    if (labelVehicle) labelVehicle.textContent = 'vehicles';

    sortControls.hidden = true;
    filterControls.hidden = true;
    confidenceGroup.hidden = true;
    exportBtn.hidden = true;
    exportCsvBtn.hidden = true;
    exportImagesBtn.hidden = true;
    speciesBtn.hidden = true;
    speciesBtn.disabled = false;
    clearBtn.hidden = true;
    cancelBtn.classList.add('hidden');
    cancelBtn.disabled = false;
}

function clearResults() {
    resetState();

    if (IS_TAURI) {
        // Phase 1: fade out proc-row + results (300ms)
        procRow.classList.remove('visible');
        resultsSection.classList.add('fading');
        confidenceGroup.classList.add('fading');
        countAnimal.textContent = '0';
        countPerson.textContent = '0';
        countVehicle.textContent = '0';
        countEmpty.textContent = '0';
        if (labelAnimal) labelAnimal.textContent = 'animals';
        if (labelPerson) labelPerson.textContent = 'people';
        if (labelVehicle) labelVehicle.textContent = 'vehicles';
        statusText.textContent = 'Ready';

        // Phase 2: after fade completes, clean up and snap drop zone in
        setTimeout(() => {
            resetDOM();
            confidenceGroup.classList.remove('fading');
            if (procText) procText.textContent = '';
            cancelBtn.style.display = '';

            // Snap: show drop zone instantly, no fade
            dropZone.classList.add('no-transition');
            dropZone.classList.remove('hidden');
            requestAnimationFrame(() => {
                dropZone.classList.remove('no-transition');
            });
        }, 300);
    } else {
        // Web: instant clear
        resetDOM();
        statusBar.classList.remove('visible');
        statusText.textContent = '';
        dropZone.classList.remove('processing');
        dropText.classList.remove('hidden');
    }
}

clearBtn.addEventListener('click', clearResults);

// ── Model toggle (desktop only) ─────────────────────────────────

const modelGroup = document.querySelector('[data-testid="model-group"]');
const modelQuickBtn = document.getElementById('model-quick');
const modelThoroughBtn = document.getElementById('model-thorough');
const resGroup = document.getElementById('res-group');

if (IS_TAURI) {
    modelThoroughBtn.hidden = false;
    modelThoroughBtn.classList.add('active');
    modelQuickBtn.classList.remove('active');
    resGroup.classList.remove('hidden');
}

modelQuickBtn.addEventListener('click', () => {
    modelQuickBtn.classList.add('active');
    modelThoroughBtn.classList.remove('active');
    MODEL_TYPE = 'quick';
    INPUT_SIZE = 1280; // Quick always runs at 1280 — it's fast enough
    resGroup.classList.add('hidden');
});

modelThoroughBtn.addEventListener('click', () => {
    modelThoroughBtn.classList.add('active');
    modelQuickBtn.classList.remove('active');
    MODEL_TYPE = 'thorough';
    resGroup.classList.remove('hidden');
    // Restore resolution from toggle state
    INPUT_SIZE = resFast.classList.contains('active') ? 640 : 1280;
});

// ── Resolution toggle ──────────────────────────────────────────

const resFast = document.getElementById('res-fast');
const resAccurate = document.getElementById('res-accurate');

// Both desktop and web default to Accurate (1280) — MDv6 is fast enough
resFast.classList.remove('active');
resAccurate.classList.add('active');

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
    for (const result of state.results) {
        const allDets = result.detections;
        const visDets = visibleDetections(result);
        updateCard(result);
    }
    applyFilters();
});

// ── Batch helpers ───────────────────────────────────────────────

function startBatch(total, folderName) {
    state.totalImages += total;
    state.batchActive = true;
    state.folderName = folderName;
    if (!IS_TAURI) statusBar.classList.add('visible');
    resultsSection.hidden = false;
    startConvergence();
    updateProgress();
}

function endBatch() {
    state.batchActive = false;
    state.folderName = null;
    state.folderPath = null;
    cancelBtn.classList.add('hidden');
    cancelBtn.disabled = false;
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

    // ── Cancel button ───────────────────────────────────────────
    cancelBtn.addEventListener('click', async () => {
        cancelBtn.disabled = true;
        statusText.textContent = 'Cancelling\u2026';
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

        stopConvergence();
        const cancelMsg = `Cancelled \u2014 ${skipped} image${skipped !== 1 ? 's' : ''} skipped`;
        statusText.textContent = cancelMsg;
        if (procText) procText.textContent = cancelMsg;
        cancelBtn.style.display = 'none';

        if (state.results.length > 0) {
            exportBtn.hidden = false;
            exportCsvBtn.hidden = false;
            exportImagesBtn.hidden = false;
            clearBtn.hidden = false;
            sortControls.hidden = false;
            filterControls.hidden = false;
            confidenceGroup.hidden = false;
            if (state.speciesStatus === 'idle' && hasAnimalDetections()) {
                speciesBtn.hidden = false;
            }
        }
    });

    // ── Image export events ────────────────────────────────────
    tauriListen('export-progress', (payload) => {
        statusText.textContent = `Exporting ${payload.current}/${payload.total}\u2026`;
    });

    tauriListen('export-complete', (payload) => {
        exportImagesBtn.disabled = false;
        const msg = payload.errors > 0
            ? `Exported ${payload.total - payload.errors}/${payload.total} images (${payload.errors} failed)`
            : `Exported ${payload.total} images`;
        statusText.textContent = msg;
    });

    tauriListen('export-error', (payload) => {
        console.error('[spoor] Export error:', payload.filePath, payload.error);
    });

    tauriListen('export-cancelled', () => {
        exportImagesBtn.disabled = false;
        statusText.textContent = 'Export cancelled';
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
        const speciesMsg = state.speciesProcessed < state.speciesTotal
            ? `Identifying species ${state.speciesProcessed + 1} of ${state.speciesTotal}\u2026`
            : 'Finishing species identification\u2026';
        statusText.textContent = speciesMsg;
        if (IS_TAURI && procText) procText.textContent = speciesMsg;

        // Update the card
        const cardResult = state.results[mapping.resultIndex];
        updateCard(cardResult);
    });

    tauriListen('species-error', (payload) => {
        console.error('[spoor] Species error:', payload.imagePath, payload.error);
        state.speciesProcessed++;
    });

    tauriListen('species-complete', () => {
        state.speciesStatus = 'done';
        speciesBtn.disabled = true;
        stopConvergence();
        updateProgress(); // refresh resultsSummary with species count
    });

    tauriListen('species-cancelled', () => {
        state.speciesStatus = 'idle';
        speciesBtn.disabled = false;
        statusText.textContent = 'Species identification cancelled';
        stopConvergence();
    });
}

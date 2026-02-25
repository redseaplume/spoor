// spoor — image export worker
// Renders annotated images at full resolution via OffscreenCanvas, builds a zip.

importScripts('https://cdn.jsdelivr.net/npm/fflate@0.8.2/umd/index.js');

const CATEGORY_COLORS = { 0: '#5a7a52', 1: '#4a6a9a', 2: '#b06a28' };
const JPEG_QUALITY = 0.92;

self.onmessage = async (e) => {
    const { results } = e.data;
    const total = results.length;
    const files = {};

    for (let i = 0; i < total; i++) {
        const r = results[i];
        try {
            const bitmap = await createImageBitmap(r.file);
            const w = r.origWidth;
            const h = r.origHeight;

            const canvas = new OffscreenCanvas(w, h);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(bitmap, 0, 0, w, h);
            bitmap.close();

            // Scale annotations relative to image size
            const dim = Math.min(w, h);
            const lineWidth = Math.max(2, Math.round(dim * 0.003));
            const fontSize = Math.max(12, Math.round(dim * 0.012));
            const labelPad = Math.round(fontSize * 0.3);
            const labelHeight = fontSize + labelPad * 2;

            ctx.font = `${fontSize}px monospace`;
            ctx.textBaseline = 'top';

            for (const det of r.detections) {
                const [x1, y1, x2, y2] = det.bbox;
                const color = CATEGORY_COLORS[det.category] || '#888';

                // Bounding box
                ctx.strokeStyle = color;
                ctx.lineWidth = lineWidth;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Label
                const displayName = det.species ? det.species.commonName : det.categoryName;
                const label = `${displayName} ${Math.round(det.confidence * 100)}%`;
                const tw = ctx.measureText(label).width;

                // Position label above box, or inside if it would clip
                const lx = x1;
                const ly = (y1 - labelHeight < 0) ? y1 : y1 - labelHeight;

                ctx.fillStyle = color;
                ctx.fillRect(lx, ly, tw + labelPad * 2, labelHeight);
                ctx.fillStyle = '#fff';
                ctx.fillText(label, lx + labelPad, ly + labelPad);
            }

            // Export as JPEG — no zip compression (JPEG is already compressed)
            const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: JPEG_QUALITY });
            const jpegData = new Uint8Array(await blob.arrayBuffer());

            files[r.fileName] = [jpegData, { level: 0 }];
        } catch (err) {
            console.error(`[export-worker] Failed: ${r.fileName}:`, err);
        }

        self.postMessage({ type: 'progress', current: i + 1, total });
    }

    const zipData = fflate.zipSync(files);
    const blob = new Blob([zipData], { type: 'application/zip' });
    self.postMessage({ type: 'done', blob });
};

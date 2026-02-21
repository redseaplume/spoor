use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;

const CONF_THRESHOLD: f32 = 0.1;
const IOU_THRESHOLD: f32 = 0.45;
const LETTERBOX_GRAY: u8 = 114;
const SPECIES_INPUT_SIZE: u32 = 480;
const SPECIES_TOP_K: usize = 5;

static SESSION_THOROUGH: OnceLock<Mutex<Session>> = OnceLock::new();
static SESSION_QUICK: OnceLock<Mutex<Session>> = OnceLock::new();
static SESSION_SPECIES: OnceLock<Mutex<Session>> = OnceLock::new();
static SPECIES_LABELS: OnceLock<Vec<SpeciesLabel>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Quick,    // MDv6 YOLOv10-c — fast, NMS-free
    Thorough, // MDv5a INT8 — slower, tighter boxes
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::Thorough
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Detection {
    pub bbox: [f32; 4],      // [x1, y1, x2, y2] in pixel coords
    pub bbox_norm: [f32; 4], // [x, y, w, h] normalized 0-1 (MegaDetector format)
    pub confidence: f32,
    pub category: u32,       // 0=animal, 1=person, 2=vehicle
    pub category_name: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionResult {
    pub file_name: String,
    pub orig_width: u32,
    pub orig_height: u32,
    pub detections: Vec<Detection>,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct SpeciesLabel {
    pub class: String,
    pub order: String,
    pub family: String,
    pub genus: String,
    pub species: String,
    pub common_name: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeciesPrediction {
    pub label_index: usize,
    pub probability: f32,
    pub common_name: String,
    pub scientific_name: String,
    pub class: String,
    pub order: String,
    pub family: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationResult {
    pub predictions: Vec<SpeciesPrediction>,
    pub inference_time_ms: u64,
}

fn category_name(cls: u32) -> String {
    match cls {
        0 => "animal".into(),
        1 => "person".into(),
        2 => "vehicle".into(),
        _ => "unknown".into(),
    }
}

// ── Model loading ────────────────────────────────────────────────

pub fn load_model(model_path: &Path, model_type: ModelType) -> Result<(), String> {
    let session = Session::builder()
        .and_then(|b| b.with_intra_threads(4))
        .and_then(|b| b.commit_from_file(model_path))
        .map_err(|e| format!("Failed to load model: {e}"))?;

    let target = match model_type {
        ModelType::Thorough => &SESSION_THOROUGH,
        ModelType::Quick => &SESSION_QUICK,
    };

    target
        .set(Mutex::new(session))
        .map_err(|_| format!("{:?} model already loaded", model_type))
}

pub fn model_available(model_type: ModelType) -> bool {
    match model_type {
        ModelType::Thorough => SESSION_THOROUGH.get().is_some(),
        ModelType::Quick => SESSION_QUICK.get().is_some(),
    }
}

pub fn load_species_model(model_path: &Path) -> Result<(), String> {
    let session = Session::builder()
        .and_then(|b| b.with_intra_threads(4))
        .and_then(|b| b.commit_from_file(model_path))
        .map_err(|e| format!("Failed to load species model: {e}"))?;

    SESSION_SPECIES
        .set(Mutex::new(session))
        .map_err(|_| "Species model already loaded".to_string())
}

pub fn load_species_labels(labels_path: &Path) -> Result<(), String> {
    let content = std::fs::read_to_string(labels_path)
        .map_err(|e| format!("Failed to read labels file: {e}"))?;

    let labels: Vec<SpeciesLabel> = content
        .lines()
        .map(|line| {
            let parts: Vec<&str> = line.split(';').collect();
            SpeciesLabel {
                class: parts.get(1).unwrap_or(&"").to_string(),
                order: parts.get(2).unwrap_or(&"").to_string(),
                family: parts.get(3).unwrap_or(&"").to_string(),
                genus: parts.get(4).unwrap_or(&"").to_string(),
                species: parts.get(5).unwrap_or(&"").to_string(),
                common_name: parts.get(6).unwrap_or(&"").to_string(),
            }
        })
        .collect();

    eprintln!("[spoor] Loaded {} species labels", labels.len());

    SPECIES_LABELS
        .set(labels)
        .map_err(|_| "Species labels already loaded".to_string())
}

pub fn species_model_available() -> bool {
    SESSION_SPECIES.get().is_some() && SPECIES_LABELS.get().is_some()
}

// ── Preprocessing (shared) ───────────────────────────────────────

/// Letterbox resize + normalize to CHW float32.
/// Returns (tensor_data, scale, pad_left, pad_top, orig_w, orig_h).
fn preprocess(
    img: &DynamicImage,
    input_size: u32,
) -> (Vec<f32>, f32, u32, u32, u32, u32) {
    let (orig_w, orig_h) = img.dimensions();

    let scale = f32::min(
        input_size as f32 / orig_w as f32,
        input_size as f32 / orig_h as f32,
    );
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;
    let pad_left = (input_size - new_w) / 2;
    let pad_top = (input_size - new_h) / 2;

    // Resize
    let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

    // Create letterbox canvas filled with gray
    let mut letterbox =
        image::RgbImage::from_pixel(input_size, input_size, image::Rgb([LETTERBOX_GRAY; 3]));

    // Paste resized image centered
    image::imageops::overlay(
        &mut letterbox,
        &resized.to_rgb8(),
        pad_left as i64,
        pad_top as i64,
    );

    // Convert to CHW float32 normalized to [0, 1]
    let num_pixels = (input_size * input_size) as usize;
    let mut chw = vec![0.0f32; 3 * num_pixels];

    for (i, pixel) in letterbox.pixels().enumerate() {
        chw[i] = pixel[0] as f32 / 255.0;
        chw[i + num_pixels] = pixel[1] as f32 / 255.0;
        chw[i + 2 * num_pixels] = pixel[2] as f32 / 255.0;
    }

    (chw, scale, pad_left, pad_top, orig_w, orig_h)
}

// ── Postprocessing: MDv5a (Thorough) ─────────────────────────────

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

fn nms(boxes: &[[f32; 4]], scores: &[f32], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; scores.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &indices {
            if !suppressed[j] && j != i && iou(&boxes[i], &boxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// Rescale a letterboxed [x1, y1, x2, y2] back to original image coordinates,
/// and produce both pixel and normalized bbox formats.
fn rescale_bbox(
    x1: f32, y1: f32, x2: f32, y2: f32,
    scale: f32, pad_left: u32, pad_top: u32,
    orig_w: u32, orig_h: u32,
) -> ([f32; 4], [f32; 4]) {
    let rx1 = ((x1 - pad_left as f32) / scale).clamp(0.0, orig_w as f32);
    let ry1 = ((y1 - pad_top as f32) / scale).clamp(0.0, orig_h as f32);
    let rx2 = ((x2 - pad_left as f32) / scale).clamp(0.0, orig_w as f32);
    let ry2 = ((y2 - pad_top as f32) / scale).clamp(0.0, orig_h as f32);

    let bbox = [rx1, ry1, rx2, ry2];
    let bbox_norm = [
        rx1 / orig_w as f32,
        ry1 / orig_h as f32,
        (rx2 - rx1) / orig_w as f32,
        (ry2 - ry1) / orig_h as f32,
    ];
    (bbox, bbox_norm)
}

/// Postprocess MDv5a output: objectness * class_conf, per-class NMS.
fn postprocess_v5(
    output_data: &[f32],
    num_preds: usize,
    num_cols: usize,
    scale: f32,
    pad_left: u32,
    pad_top: u32,
    orig_w: u32,
    orig_h: u32,
) -> Vec<Detection> {
    struct RawDet {
        bbox: [f32; 4],
        score: f32,
        cls: u32,
    }

    let mut filtered = Vec::new();

    for i in 0..num_preds {
        let off = i * num_cols;
        let obj_conf = output_data[off + 4];
        if obj_conf <= CONF_THRESHOLD {
            continue;
        }

        let cx = output_data[off];
        let cy = output_data[off + 1];
        let w = output_data[off + 2];
        let h = output_data[off + 3];

        let mut best_cls = 0u32;
        let mut best_score = 0.0f32;
        for c in 0..3u32 {
            let s = obj_conf * output_data[off + 5 + c as usize];
            if s > best_score {
                best_score = s;
                best_cls = c;
            }
        }

        filtered.push(RawDet {
            bbox: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            score: best_score,
            cls: best_cls,
        });
    }

    if filtered.is_empty() {
        return vec![];
    }

    // NMS per class
    let mut detections = Vec::new();
    for cls in 0..3u32 {
        let items: Vec<&RawDet> = filtered.iter().filter(|d| d.cls == cls).collect();
        if items.is_empty() {
            continue;
        }

        let boxes: Vec<[f32; 4]> = items.iter().map(|d| d.bbox).collect();
        let scores: Vec<f32> = items.iter().map(|d| d.score).collect();
        let keep = nms(&boxes, &scores, IOU_THRESHOLD);

        for idx in keep {
            let det = items[idx];
            let (bbox, bbox_norm) = rescale_bbox(
                det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                scale, pad_left, pad_top, orig_w, orig_h,
            );

            detections.push(Detection {
                bbox,
                bbox_norm,
                confidence: det.score,
                category: det.cls,
                category_name: category_name(det.cls),
            });
        }
    }

    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    detections
}

// ── Postprocessing: MDv6 YOLOv10-c (Quick) ──────────────────────

/// Postprocess MDv6 YOLOv10-c output: NMS-free, output is [1, 300, 6] = [x1, y1, x2, y2, conf, class].
fn postprocess_v6(
    output_data: &[f32],
    num_preds: usize,
    scale: f32,
    pad_left: u32,
    pad_top: u32,
    orig_w: u32,
    orig_h: u32,
) -> Vec<Detection> {
    let mut detections = Vec::new();

    for i in 0..num_preds {
        let off = i * 6;
        let conf = output_data[off + 4];
        if conf <= CONF_THRESHOLD {
            continue;
        }

        let cls = output_data[off + 5] as u32;
        let (bbox, bbox_norm) = rescale_bbox(
            output_data[off], output_data[off + 1],
            output_data[off + 2], output_data[off + 3],
            scale, pad_left, pad_top, orig_w, orig_h,
        );

        detections.push(Detection {
            bbox,
            bbox_norm,
            confidence: conf,
            category: cls,
            category_name: category_name(cls),
        });
    }

    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    detections
}

// ── Inference ────────────────────────────────────────────────────

/// Run inference on an already-decoded image.
fn run_inference(
    img: &DynamicImage,
    input_size: u32,
    model_type: ModelType,
) -> Result<(Vec<Detection>, u32, u32, u64), String> {
    let session_lock = match model_type {
        ModelType::Thorough => SESSION_THOROUGH.get(),
        ModelType::Quick => SESSION_QUICK.get(),
    };

    let mut session = session_lock
        .ok_or(format!("{:?} model not loaded", model_type))?
        .lock()
        .map_err(|e| format!("Session lock poisoned: {e}"))?;

    let t0 = std::time::Instant::now();

    let (orig_w, orig_h) = img.dimensions();
    let (chw, scale, pad_left, pad_top, _, _) = preprocess(img, input_size);

    // Build input tensor
    let sz = input_size as usize;
    let input_tensor = ort::value::Value::from_array(([1usize, 3, sz, sz], chw))
        .map_err(|e| format!("Failed to create tensor: {e}"))?;

    // Run inference
    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| format!("Inference failed: {e}"))?;

    let (output_shape, output_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("Failed to extract output: {e}"))?;

    let num_preds = output_shape[1] as usize;

    let detections = match model_type {
        ModelType::Thorough => {
            let num_cols = output_shape[2] as usize;
            postprocess_v5(
                output_data, num_preds, num_cols, scale, pad_left, pad_top, orig_w, orig_h,
            )
        }
        ModelType::Quick => {
            postprocess_v6(
                output_data, num_preds, scale, pad_left, pad_top, orig_w, orig_h,
            )
        }
    };

    let elapsed = t0.elapsed().as_millis() as u64;
    Ok((detections, orig_w, orig_h, elapsed))
}

// ── Public API ───────────────────────────────────────────────────

/// Detect from a file path.
pub fn detect(
    image_path: &Path,
    input_size: u32,
    model_type: ModelType,
) -> Result<DetectionResult, String> {
    let img = image::open(image_path).map_err(|e| format!("Failed to open image: {e}"))?;
    let (detections, orig_w, orig_h, elapsed) = run_inference(&img, input_size, model_type)?;

    let file_name = image_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    Ok(DetectionResult {
        file_name,
        orig_width: orig_w,
        orig_height: orig_h,
        detections,
        inference_time_ms: elapsed,
    })
}

/// Check if a file has a supported image extension (case-insensitive).
pub fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "tiff" | "tif"
            )
        })
        .unwrap_or(false)
}

/// Recursively collect image files from a directory, sorted by path.
pub fn collect_images(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut images = Vec::new();
    collect_images_inner(dir, &mut images)?;
    images.sort();
    Ok(images)
}

fn collect_images_inner(dir: &Path, images: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read {}: {e}", dir.display()))?;

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_images_inner(&path, images)?;
        } else if is_image_file(&path) {
            images.push(path);
        }
    }
    Ok(())
}

/// Detect from raw image bytes (JPEG/PNG/WebP). Used by the Tauri IPC path.
pub fn detect_bytes(
    bytes: &[u8],
    file_name: &str,
    input_size: u32,
    model_type: ModelType,
) -> Result<DetectionResult, String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("Failed to decode image: {e}"))?;
    let (detections, orig_w, orig_h, elapsed) = run_inference(&img, input_size, model_type)?;

    Ok(DetectionResult {
        file_name: file_name.to_string(),
        orig_width: orig_w,
        orig_height: orig_h,
        detections,
        inference_time_ms: elapsed,
    })
}

// ── Species classification ──────────────────────────────────────

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

/// Crop bbox from image, resize to 480x480, convert to NHWC float32 [0, 1].
fn preprocess_species(img: &DynamicImage, bbox: &[f32; 4]) -> Vec<f32> {
    let (orig_w, orig_h) = img.dimensions();

    // bbox is [x1, y1, x2, y2] in pixel coords
    let x1 = (bbox[0] as u32).min(orig_w);
    let y1 = (bbox[1] as u32).min(orig_h);
    let x2 = (bbox[2] as u32).min(orig_w);
    let y2 = (bbox[3] as u32).min(orig_h);
    let crop_w = x2.saturating_sub(x1).max(1);
    let crop_h = y2.saturating_sub(y1).max(1);

    let crop = img.crop_imm(x1, y1, crop_w, crop_h);
    let resized = crop.resize_exact(
        SPECIES_INPUT_SIZE,
        SPECIES_INPUT_SIZE,
        FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();

    // NHWC layout: [1, 480, 480, 3]
    let num_pixels = (SPECIES_INPUT_SIZE * SPECIES_INPUT_SIZE) as usize;
    let mut hwc = Vec::with_capacity(num_pixels * 3);
    for pixel in rgb.pixels() {
        hwc.push(pixel[0] as f32 / 255.0);
        hwc.push(pixel[1] as f32 / 255.0);
        hwc.push(pixel[2] as f32 / 255.0);
    }
    hwc
}

/// Classify species for a single detection crop.
pub fn classify_species(
    image_path: &Path,
    bbox: &[f32; 4],
) -> Result<ClassificationResult, String> {
    let mut session = SESSION_SPECIES
        .get()
        .ok_or("Species model not loaded")?
        .lock()
        .map_err(|e| format!("Species session lock poisoned: {e}"))?;

    let labels = SPECIES_LABELS.get().ok_or("Species labels not loaded")?;

    let img = image::open(image_path).map_err(|e| format!("Failed to open image: {e}"))?;

    let t0 = std::time::Instant::now();

    let hwc = preprocess_species(&img, bbox);
    let sz = SPECIES_INPUT_SIZE as usize;
    let input_tensor = ort::value::Value::from_array(([1usize, sz, sz, 3usize], hwc))
        .map_err(|e| format!("Failed to create species tensor: {e}"))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| format!("Species inference failed: {e}"))?;

    let (_, output_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("Failed to extract species output: {e}"))?;

    let probs = softmax(output_data);

    // Top-K by probability
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let predictions: Vec<SpeciesPrediction> = indexed
        .iter()
        .take(SPECIES_TOP_K)
        .filter_map(|&(idx, prob)| {
            labels.get(idx).map(|label| SpeciesPrediction {
                label_index: idx,
                probability: prob,
                common_name: label.common_name.clone(),
                scientific_name: if label.species.is_empty() {
                    String::new()
                } else {
                    format!("{} {}", label.genus, label.species)
                },
                class: label.class.clone(),
                order: label.order.clone(),
                family: label.family.clone(),
            })
        })
        .collect();

    let elapsed = t0.elapsed().as_millis() as u64;

    Ok(ClassificationResult {
        predictions,
        inference_time_ms: elapsed,
    })
}

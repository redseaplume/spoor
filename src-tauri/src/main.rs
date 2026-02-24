// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod detect;
mod export;

use detect::ModelType;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::path::BaseDirectory;
use tauri::{Emitter, Manager};

static CANCEL_FLAG: AtomicBool = AtomicBool::new(false);

// ── Commands ────────────────────────────────────────────────────

#[tauri::command]
fn detect_image(
    image_path: String,
    input_size: Option<u32>,
    model: Option<ModelType>,
) -> Result<detect::DetectionResult, String> {
    let size = input_size.unwrap_or(1280);
    let model_type = model.unwrap_or_default();
    detect::detect(&PathBuf::from(&image_path), size, model_type)
}

#[tauri::command]
fn detect_image_bytes(
    image_bytes: Vec<u8>,
    file_name: String,
    input_size: Option<u32>,
    model: Option<ModelType>,
) -> Result<detect::DetectionResult, String> {
    let size = input_size.unwrap_or(1280);
    let model_type = model.unwrap_or_default();
    detect::detect_bytes(&image_bytes, &file_name, size, model_type)
}

// ── Batch processing ────────────────────────────────────────────

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FolderResultEvent {
    file_path: String,
    result: detect::DetectionResult,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FolderErrorEvent {
    file_path: String,
    error: String,
}

#[derive(Clone, serde::Serialize)]
struct FolderCompleteEvent {
    total: usize,
}

// ── Species classification ────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClassifyRequest {
    image_path: String,
    bbox: [f32; 4], // [x1, y1, x2, y2] pixel coords
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SpeciesResultEvent {
    index: usize,
    image_path: String,
    result: detect::ClassificationResult,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SpeciesErrorEvent {
    index: usize,
    image_path: String,
    error: String,
}

#[tauri::command]
fn classify_detections(app: tauri::AppHandle, requests: Vec<ClassifyRequest>) -> Result<(), String> {
    if !detect::species_model_available() {
        return Err("Species model not available".into());
    }

    CANCEL_FLAG.store(false, Ordering::Relaxed);
    std::thread::spawn(move || {
        for (i, req) in requests.iter().enumerate() {
            if CANCEL_FLAG.load(Ordering::Relaxed) {
                let _ = app.emit("species-cancelled", serde_json::json!({}));
                return;
            }
            match detect::classify_species(&PathBuf::from(&req.image_path), &req.bbox) {
                Ok(result) => {
                    let _ = app.emit(
                        "species-result",
                        SpeciesResultEvent {
                            index: i,
                            image_path: req.image_path.clone(),
                            result,
                        },
                    );
                }
                Err(e) => {
                    let _ = app.emit(
                        "species-error",
                        SpeciesErrorEvent {
                            index: i,
                            image_path: req.image_path.clone(),
                            error: e,
                        },
                    );
                }
            }
        }
        let _ = app.emit("species-complete", serde_json::json!({}));
    });

    Ok(())
}

fn run_batch(app: tauri::AppHandle, images: Vec<PathBuf>, size: u32, model_type: ModelType) {
    let total = images.len();
    for image_path in &images {
        if CANCEL_FLAG.load(Ordering::Relaxed) {
            let _ = app.emit("folder-cancelled", serde_json::json!({}));
            return;
        }
        match detect::detect(image_path, size, model_type) {
            Ok(result) => {
                let _ = app.emit("folder-result", FolderResultEvent {
                    file_path: image_path.to_string_lossy().into_owned(),
                    result,
                });
            }
            Err(e) => {
                let _ = app.emit("folder-error", FolderErrorEvent {
                    file_path: image_path.to_string_lossy().into_owned(),
                    error: e,
                });
            }
        }
    }
    let _ = app.emit("folder-complete", FolderCompleteEvent { total });
}

#[tauri::command]
fn process_folder(
    app: tauri::AppHandle,
    folder_path: String,
    input_size: Option<u32>,
    model: Option<ModelType>,
) -> Result<usize, String> {
    let size = input_size.unwrap_or(1280);
    let model_type = model.unwrap_or_default();
    let path = PathBuf::from(&folder_path);
    let images = detect::collect_images(&path)?;
    let total = images.len();
    if total == 0 {
        return Ok(0);
    }

    CANCEL_FLAG.store(false, Ordering::Relaxed);
    std::thread::spawn(move || run_batch(app, images, size, model_type));
    Ok(total)
}

#[tauri::command]
fn process_paths(
    app: tauri::AppHandle,
    paths: Vec<String>,
    input_size: Option<u32>,
    model: Option<ModelType>,
) -> Result<usize, String> {
    let size = input_size.unwrap_or(1280);
    let model_type = model.unwrap_or_default();
    let mut images = Vec::new();

    for path_str in &paths {
        let path = PathBuf::from(path_str);
        if path.is_dir() {
            images.extend(detect::collect_images(&path)?);
        } else if detect::is_image_file(&path) {
            images.push(path);
        }
    }

    images.sort();
    let total = images.len();
    if total == 0 {
        return Ok(0);
    }

    CANCEL_FLAG.store(false, Ordering::Relaxed);
    std::thread::spawn(move || run_batch(app, images, size, model_type));
    Ok(total)
}

#[tauri::command]
fn cancel_processing() {
    CANCEL_FLAG.store(true, Ordering::Relaxed);
}

// ── Image export ───────────────────────────────────────────────

static FONT_BYTES: &[u8] = include_bytes!("../resources/JetBrainsMono-Regular.ttf");

#[tauri::command]
fn export_images(
    app: tauri::AppHandle,
    output_dir: String,
    items: Vec<export::ExportItem>,
    base_dir: Option<String>,
) -> Result<usize, String> {
    let total = items.len();
    if total == 0 {
        return Ok(0);
    }

    let output = PathBuf::from(&output_dir);
    if !output.exists() {
        std::fs::create_dir_all(&output)
            .map_err(|e| format!("Create output dir: {}", e))?;
    }

    let base = base_dir.map(PathBuf::from);

    CANCEL_FLAG.store(false, Ordering::Relaxed);
    std::thread::spawn(move || {
        export::export_images(app, output, items, base, FONT_BYTES, &CANCEL_FLAG);
    });

    Ok(total)
}

// ── Drag-and-drop event ─────────────────────────────────────────

#[derive(Clone, serde::Serialize)]
struct NativeDropEvent {
    paths: Vec<String>,
}

// ── Model loading ────────────────────────────────────────────────

struct ModelInfo {
    filename: &'static str,
    model_type: ModelType,
}

const MODELS: &[ModelInfo] = &[
    ModelInfo { filename: "mdv5a_int8.onnx", model_type: ModelType::Thorough },
    ModelInfo { filename: "mdv6-yolov10-c.onnx", model_type: ModelType::Quick },
];

/// Try to find a model via CWD-relative path (dev / CLI).
fn dev_model_path(filename: &str) -> Option<PathBuf> {
    std::env::current_dir()
        .ok()
        .map(|cwd| cwd.join("../models").join(filename))
        .filter(|p| p.exists())
}

fn load_and_log(path: &std::path::Path, model_type: ModelType) {
    match detect::load_model(path, model_type) {
        Ok(()) => eprintln!("[spoor] {:?} model loaded: {}", model_type, path.display()),
        Err(e) => eprintln!("[spoor] Warning: {e}"),
    }
}

const SPECIES_MODEL_FILENAME: &str = "speciesnet-v4.0.2a.onnx";
const SPECIES_LABELS_FILENAME: &str = "speciesnet-labels.txt";

fn load_species(model_path: &std::path::Path, labels_path: &std::path::Path) {
    match detect::load_species_model(model_path) {
        Ok(()) => eprintln!("[spoor] Species model loaded: {}", model_path.display()),
        Err(e) => eprintln!("[spoor] Warning: {e}"),
    }
    match detect::load_species_labels(labels_path) {
        Ok(()) => {}
        Err(e) => eprintln!("[spoor] Warning: {e}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // CLI test mode: spoor --test <image_path|dir> [input_size] [quick|thorough]
    if args.len() >= 3 && args[1] == "--test" {
        // Load detection models for CLI
        for info in MODELS {
            match dev_model_path(info.filename) {
                Some(p) => load_and_log(&p, info.model_type),
                None => eprintln!("[spoor] {:?} model not found at dev path", info.model_type),
            }
        }

        // Load species model for CLI
        let species_model = dev_model_path(SPECIES_MODEL_FILENAME);
        let species_labels = dev_model_path(SPECIES_LABELS_FILENAME);
        if let (Some(mp), Some(lp)) = (&species_model, &species_labels) {
            load_species(mp, lp);
        } else {
            eprintln!("[spoor] Species model not found at dev path (optional)");
        }

        let path = PathBuf::from(&args[2]);
        let input_size = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1280);
        let model_type = args.get(4)
            .map(|s| match s.as_str() {
                "quick" => ModelType::Quick,
                _ => ModelType::Thorough,
            })
            .unwrap_or_default();

        if !detect::model_available(model_type) {
            eprintln!("[spoor] {:?} model not available", model_type);
            std::process::exit(1);
        }

        let images = if path.is_dir() {
            detect::collect_images(&path).unwrap_or_else(|e| {
                eprintln!("[spoor] Error: {e}");
                std::process::exit(1);
            })
        } else {
            vec![path]
        };

        let has_species = detect::species_model_available();

        for image_path in &images {
            match detect::detect(image_path, input_size, model_type) {
                Ok(result) => {
                    let n = result.detections.len();
                    let top = if n > 0 {
                        format!("{} {:.0}%", result.detections[0].category_name, result.detections[0].confidence * 100.0)
                    } else {
                        "empty".into()
                    };
                    eprintln!("  {} → {}ms, {} det ({})", result.file_name, result.inference_time_ms, n, top);

                    // Run species classification on animal detections
                    if has_species {
                        for det in &result.detections {
                            if det.category == 0 && det.confidence > 0.2 {
                                match detect::classify_species(image_path, &det.bbox) {
                                    Ok(cls) => {
                                        if let Some(top) = cls.predictions.first() {
                                            eprintln!(
                                                "    species: {} ({}) {:.1}% ({}ms)",
                                                top.common_name,
                                                top.scientific_name,
                                                top.probability * 100.0,
                                                cls.inference_time_ms
                                            );
                                        }
                                    }
                                    Err(e) => eprintln!("    species error: {e}"),
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  {} → error: {e}", image_path.display());
                }
            }
        }
        return;
    }

    // Normal Tauri app
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // Load detection models
            for info in MODELS {
                let model_path = app.path()
                    .resolve(info.filename, BaseDirectory::Resource)
                    .ok()
                    .filter(|p| p.exists())
                    .or_else(|| dev_model_path(info.filename));

                match model_path {
                    Some(p) => load_and_log(&p, info.model_type),
                    None => eprintln!("[spoor] {:?} model not found", info.model_type),
                }
            }

            // Load species model + labels
            let species_model = app.path()
                .resolve(SPECIES_MODEL_FILENAME, BaseDirectory::Resource)
                .ok()
                .filter(|p| p.exists())
                .or_else(|| dev_model_path(SPECIES_MODEL_FILENAME));
            let species_labels = app.path()
                .resolve(SPECIES_LABELS_FILENAME, BaseDirectory::Resource)
                .ok()
                .filter(|p| p.exists())
                .or_else(|| dev_model_path(SPECIES_LABELS_FILENAME));

            match (species_model, species_labels) {
                (Some(mp), Some(lp)) => load_species(&mp, &lp),
                _ => eprintln!("[spoor] Species model or labels not found (optional)"),
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            detect_image,
            detect_image_bytes,
            process_folder,
            process_paths,
            cancel_processing,
            classify_detections,
            export_images
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::DragDrop(tauri::DragDropEvent::Drop { paths, .. }) = event {
                let path_strings: Vec<String> = paths.iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect();
                let _ = window.emit("native-drop", NativeDropEvent { paths: path_strings });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

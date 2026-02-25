use ab_glyph::{FontRef, PxScale, ScaleFont};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tauri::Emitter;

// ── Types ──────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportDetection {
    pub bbox: [f32; 4], // [x, y, w, h] normalized 0-1
    pub category: String,
    pub confidence: f32,
    pub species: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportItem {
    pub file_path: String,
    pub detections: Vec<ExportDetection>,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportProgressEvent {
    current: usize,
    total: usize,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportErrorEvent {
    file_path: String,
    error: String,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportCompleteEvent {
    total: usize,
    errors: usize,
}

// ── Category colors ────────────────────────────────────────────

fn category_color(category: &str) -> [u8; 3] {
    match category {
        "animal" => [90, 122, 82],   // #5a7a52
        "person" => [74, 106, 154],  // #4a6a9a
        "vehicle" => [176, 106, 40], // #b06a28
        _ => [136, 136, 136],        // #888
    }
}

// ── EXIF APP1 extraction ───────────────────────────────────────

fn extract_app1(jpeg_bytes: &[u8]) -> Option<Vec<u8>> {
    // JPEG starts with FF D8, then segments: marker (FF xx) + length (2 bytes big-endian)
    if jpeg_bytes.len() < 4 || jpeg_bytes[0] != 0xFF || jpeg_bytes[1] != 0xD8 {
        return None;
    }
    let mut pos = 2;
    while pos + 4 <= jpeg_bytes.len() {
        if jpeg_bytes[pos] != 0xFF {
            break;
        }
        let marker = jpeg_bytes[pos + 1];
        // APP1 = 0xE1
        if marker == 0xE1 {
            let seg_len = u16::from_be_bytes([jpeg_bytes[pos + 2], jpeg_bytes[pos + 3]]) as usize;
            if pos + 2 + seg_len <= jpeg_bytes.len() {
                // Return the full segment: marker + length + data
                return Some(jpeg_bytes[pos..pos + 2 + seg_len].to_vec());
            }
        }
        // Skip to next segment
        let seg_len = u16::from_be_bytes([jpeg_bytes[pos + 2], jpeg_bytes[pos + 3]]) as usize;
        pos += 2 + seg_len;
    }
    None
}

fn inject_app1(jpeg_bytes: &[u8], app1: &[u8]) -> Vec<u8> {
    // Insert APP1 segment right after SOI (FF D8)
    let mut out = Vec::with_capacity(jpeg_bytes.len() + app1.len());
    out.extend_from_slice(&jpeg_bytes[..2]); // FF D8
    out.extend_from_slice(app1);
    out.extend_from_slice(&jpeg_bytes[2..]);
    out
}

// ── Drawing ────────────────────────────────────────────────────

fn draw_rect(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    thickness: usize,
    color: [u8; 3],
) {
    let pitch = width * 3;
    // Top and bottom edges
    for t in 0..thickness {
        let ty = y1.saturating_sub(t / 2) + t;
        let by = y2.saturating_sub(thickness / 2) + t;
        if ty < height {
            for x in x1..=x2.min(width - 1) {
                let i = ty * pitch + x * 3;
                pixels[i] = color[0];
                pixels[i + 1] = color[1];
                pixels[i + 2] = color[2];
            }
        }
        if by < height {
            for x in x1..=x2.min(width - 1) {
                let i = by * pitch + x * 3;
                pixels[i] = color[0];
                pixels[i + 1] = color[1];
                pixels[i + 2] = color[2];
            }
        }
    }
    // Left and right edges
    for y in y1..=y2.min(height - 1) {
        for t in 0..thickness {
            let lx = x1.saturating_sub(t / 2) + t;
            let rx = x2.saturating_sub(thickness / 2) + t;
            if lx < width {
                let i = y * pitch + lx * 3;
                pixels[i] = color[0];
                pixels[i + 1] = color[1];
                pixels[i + 2] = color[2];
            }
            if rx < width {
                let i = y * pitch + rx * 3;
                pixels[i] = color[0];
                pixels[i + 1] = color[1];
                pixels[i + 2] = color[2];
            }
        }
    }
}

fn fill_rect_alpha(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    color: [u8; 3],
    alpha: f32,
) {
    let pitch = width * 3;
    let inv = 1.0 - alpha;
    for y in y1..=y2.min(height - 1) {
        for x in x1..=x2.min(width - 1) {
            let i = y * pitch + x * 3;
            pixels[i] = (color[0] as f32 * alpha + pixels[i] as f32 * inv) as u8;
            pixels[i + 1] = (color[1] as f32 * alpha + pixels[i + 1] as f32 * inv) as u8;
            pixels[i + 2] = (color[2] as f32 * alpha + pixels[i + 2] as f32 * inv) as u8;
        }
    }
}

fn draw_text(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    font: &FontRef,
    scale: PxScale,
    color: [u8; 3],
) {
    use ab_glyph::Font;
    let scaled = font.as_scaled(scale);
    let mut cursor_x = x as f32;
    let pitch = width * 3;

    for ch in text.chars() {
        let glyph_id = font.glyph_id(ch);
        let glyph = glyph_id.with_scale_and_position(scale, ab_glyph::point(cursor_x, y as f32));

        if let Some(outlined) = font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();
            outlined.draw(|gx, gy, coverage| {
                let px = bounds.min.x as usize + gx as usize;
                let py = bounds.min.y as usize + gy as usize;
                if px < width && py < height && coverage > 0.1 {
                    let i = py * pitch + px * 3;
                    let a = coverage.min(1.0);
                    let inv = 1.0 - a;
                    pixels[i] = (color[0] as f32 * a + pixels[i] as f32 * inv) as u8;
                    pixels[i + 1] = (color[1] as f32 * a + pixels[i + 1] as f32 * inv) as u8;
                    pixels[i + 2] = (color[2] as f32 * a + pixels[i + 2] as f32 * inv) as u8;
                }
            });
        }

        cursor_x += scaled.h_advance(glyph_id);
    }
}

fn text_width(text: &str, font: &FontRef, scale: PxScale) -> f32 {
    use ab_glyph::Font;
    let scaled = font.as_scaled(scale);
    text.chars()
        .map(|ch| scaled.h_advance(font.glyph_id(ch)))
        .sum()
}

// ── Core export ────────────────────────────────────────────────

fn export_single(
    item: &ExportItem,
    output_dir: &Path,
    base_dir: Option<&Path>,
    font: &FontRef,
) -> Result<(), String> {
    let src_path = Path::new(&item.file_path);
    let raw_bytes = fs::read(src_path).map_err(|e| format!("Read {}: {}", item.file_path, e))?;

    // Extract EXIF before decode
    let app1 = extract_app1(&raw_bytes);

    // Decode JPEG
    let mut decompressor = turbojpeg::Decompressor::new().map_err(|e| e.to_string())?;
    let header = decompressor
        .read_header(&raw_bytes)
        .map_err(|e| format!("Header {}: {}", item.file_path, e))?;
    let width = header.width;
    let height = header.height;

    let mut image = turbojpeg::Image {
        pixels: vec![0u8; width * height * 3],
        width,
        height,
        pitch: width * 3,
        format: turbojpeg::PixelFormat::RGB,
    };
    decompressor
        .decompress(&raw_bytes, image.as_deref_mut())
        .map_err(|e| format!("Decode {}: {}", item.file_path, e))?;

    // Draw annotations
    let stroke = (width.max(height) as f32 * 0.002).max(2.0) as usize;
    let font_size = (width.max(height) as f32 * 0.015).max(14.0);
    let scale = PxScale::from(font_size);
    let label_pad = (font_size * 0.3) as usize;

    for det in &item.detections {
        let color = category_color(&det.category);
        // bbox is [x, y, w, h] normalized
        let bx = (det.bbox[0] * width as f32) as usize;
        let by = (det.bbox[1] * height as f32) as usize;
        let bw = (det.bbox[2] * width as f32) as usize;
        let bh = (det.bbox[3] * height as f32) as usize;
        let x2 = (bx + bw).min(width - 1);
        let y2 = (by + bh).min(height - 1);

        // Bounding box
        draw_rect(&mut image.pixels, width, height, bx, by, x2, y2, stroke, color);

        // Label text
        let label = if let Some(ref species) = det.species {
            format!("{} {:.0}%", species, det.confidence * 100.0)
        } else {
            format!("{} {:.0}%", det.category, det.confidence * 100.0)
        };

        let tw = text_width(&label, font, scale) as usize;
        let th = font_size as usize;
        let lx = bx;
        let ly = if by > th + label_pad * 2 + stroke {
            by - th - label_pad * 2
        } else {
            by + stroke
        };

        // Label background
        fill_rect_alpha(
            &mut image.pixels,
            width,
            height,
            lx,
            ly,
            (lx + tw + label_pad * 2).min(width - 1),
            (ly + th + label_pad * 2).min(height - 1),
            color,
            0.75,
        );

        // Label text (white)
        draw_text(
            &mut image.pixels,
            width,
            height,
            lx + label_pad,
            ly + label_pad + th,
            &label,
            font,
            scale,
            [255, 255, 255],
        );
    }

    // Encode JPEG
    let mut compressor = turbojpeg::Compressor::new().map_err(|e| e.to_string())?;
    compressor.set_quality(92).map_err(|e| e.to_string())?;
    let mut encoded = compressor
        .compress_to_vec(image.as_deref())
        .map_err(|e| format!("Encode {}: {}", item.file_path, e))?;

    // Re-inject EXIF
    if let Some(ref app1_data) = app1 {
        encoded = inject_app1(&encoded, app1_data);
    }

    // Determine output path
    let out_path = if let Some(base) = base_dir {
        let relative = src_path
            .strip_prefix(base)
            .unwrap_or(src_path.file_name().map(Path::new).unwrap_or(src_path));
        output_dir.join(relative)
    } else {
        output_dir.join(src_path.file_name().unwrap_or(src_path.as_os_str()))
    };

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Create dir {}: {}", parent.display(), e))?;
    }

    fs::write(&out_path, &encoded)
        .map_err(|e| format!("Write {}: {}", out_path.display(), e))?;

    Ok(())
}

// ── Public API ─────────────────────────────────────────────────

pub fn export_images(
    app: tauri::AppHandle,
    output_dir: PathBuf,
    items: Vec<ExportItem>,
    base_dir: Option<PathBuf>,
    font_bytes: &'static [u8],
    cancel_flag: &'static std::sync::atomic::AtomicBool,
) {
    let total = items.len();
    let errors = std::sync::atomic::AtomicUsize::new(0);

    let font = match FontRef::try_from_slice(font_bytes) {
        Ok(f) => f,
        Err(e) => {
            let _ = app.emit(
                "export-error",
                ExportErrorEvent {
                    file_path: String::new(),
                    error: format!("Failed to load font: {}", e),
                },
            );
            return;
        }
    };

    // Safety check: refuse to write to any source directory
    let canon_out = output_dir.canonicalize().unwrap_or(output_dir.clone());
    for item in &items {
        let src = Path::new(&item.file_path);
        if let Some(parent) = src.parent() {
            if let Ok(canon_src) = parent.canonicalize() {
                if canon_out.starts_with(&canon_src) || canon_src.starts_with(&canon_out) {
                    let _ = app.emit(
                        "export-error",
                        ExportErrorEvent {
                            file_path: item.file_path.clone(),
                            error: "Output directory overlaps with source directory".into(),
                        },
                    );
                    return;
                }
            }
        }
    }

    let base_ref = base_dir.as_deref();
    let progress = std::sync::atomic::AtomicUsize::new(0);

    items.par_iter().for_each(|item| {
        if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        if let Err(e) = export_single(item, &output_dir, base_ref, &font) {
            errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = app.emit(
                "export-error",
                ExportErrorEvent {
                    file_path: item.file_path.clone(),
                    error: e,
                },
            );
        }

        let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let _ = app.emit(
            "export-progress",
            ExportProgressEvent {
                current: done,
                total,
            },
        );
    });

    if !cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = app.emit(
            "export-complete",
            ExportCompleteEvent {
                total,
                errors: errors.load(std::sync::atomic::Ordering::Relaxed),
            },
        );
    } else {
        let _ = app.emit("export-cancelled", serde_json::json!({}));
    }
}

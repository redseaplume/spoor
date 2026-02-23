# Image Export Design

Annotated image export for both web and Tauri. Same UX pattern as existing JSON/CSV export — respects current threshold slider and category filters.

## Shared Principles

- **Third button** alongside existing "Export JSON" and "Export CSV"
- **No export-specific filter UI** — exports whatever is currently visible (`visibleDetections(result)`)
- **Same visual language** as on-screen bounding boxes: category colors, label backgrounds, species names when available, confidence percentages
- **Full original resolution**, not display resolution
- **Only detections above threshold** — matches what's visible on screen
- **JPEG quality**: 0.92 hardcoded. Visually lossless on photographic content, meaningfully smaller than 1.0. Generational loss from re-encode is negligible at this level. Not worth exposing to the user — researchers doing pixel-level work should use originals + JSON, not annotated JPEGs.

## Web

### UX
- Button: "Export Images" in the export button row
- Click → browser downloads a zip. No folder selection, no dialog — same instant feel as JSON/CSV
- **Zip filename**: includes source folder name if available (`spoor_field_site_A.zip`), falls back to `spoor_images.zip` for individual file drops. Uses `state.folderName` when set.
- Button shows progress during export: "Exporting... (47/200)"

### Implementation
- **OffscreenCanvas in a Web Worker** — renders off main thread, UI stays responsive on weak machines
- Process one image at a time to control memory:
  1. Retain `File` reference on each result (`result.file = item.file` during processing)
  2. `createImageBitmap(result.file)` in the worker — reliable decode, doesn't depend on browser cache
  3. Draw on OffscreenCanvas at `origWidth × origHeight`
  4. Draw annotations: colored rectangles, semi-transparent label backgrounds, text (species name or category + confidence %)
  5. `canvas.convertToBlob('image/jpeg', 0.92)`
  6. Push blob into zip
  7. Release — one image in memory at a time
- **Zip via `fflate`** (~12KB) — fast, works in workers, supports streaming construction
- Worker posts progress messages back to main thread for button updates

### EXIF
- No. Canvas encode strips it. Re-injection adds complexity for the platform where it matters least — web users are triaging, not archiving. Desktop app preserves EXIF for researchers who need it.

### New dependencies
- `fflate` (~12KB, JS)

## Tauri

### UX
- Button: same "Export Images" button
- Click → native directory picker → user chooses output directory → backend processes → progress events update button → confirmation message ("Exported 147 images to /path")
- **Directory structure preserved**: `source/site_A/cam_03/IMG_0042.JPG` → `output/site_A/cam_03/IMG_0042.JPG`
- **Drag-and-drop individual files** (no common parent): writes flat to the chosen output directory, no subdirectories

### Implementation
- **Rendering in Rust** — no image bytes cross the IPC bridge
- Frontend sends one IPC call with:
  - Output directory path (from native picker)
  - Filtered result set: file paths, detections (bbox, category, confidence), species data
- Backend processes each image sequentially:
  1. Read original JPEG from disk
  2. Extract EXIF/APP1 segment from raw bytes (before decoding)
  3. Decode with `image` crate (already a dependency)
  4. Draw annotations: colored rectangles, semi-transparent label backgrounds, text via `ab_glyph`
  5. Encode to JPEG at 0.92 quality
  6. Re-inject original EXIF segment into output bytes
  7. Write to output directory, creating parent dirs as needed
  8. Emit progress event to frontend
  9. Drop decoded image — one in memory at a time
- **Safety check**: refuse to write to source directory

### EXIF
- Yes. Raw segment transplant — read APP1 from source bytes, re-inject into output bytes after JPEG encode. ~30-40 lines. Preserves GPS, timestamps, camera settings for downstream research workflows.

### Font
- **JetBrains Mono Regular** (OFL licensed, free, no restrictions on bundling)
- Designed for small-size readability, increased x-height, unambiguous characters (1/l/I, 0/O)
- Subsetted to Latin + digits + basic punctuation + common accented characters (for species names): ~20-25KB
- Single weight (Regular only), bundled as a resource
- Rasterized via `ab_glyph` — glyph cache means repeated characters are fast

### New dependencies
- `ab_glyph` (font rasterization, small, well-maintained)
- JetBrains Mono Regular subset (~20-25KB, bundled .ttf)

### Progress
- Tauri events per image, same pattern as batch detection progress

## Comparison

|                | Web                            | Tauri                              |
|----------------|--------------------------------|------------------------------------|
| Output         | Zip download                   | Directory on disk                  |
| Folder select  | No (browser download)          | Yes (native directory picker)      |
| Rendering      | OffscreenCanvas (JS)           | `image` + `ab_glyph` (Rust)       |
| Image source   | `File` ref → createImageBitmap | Original file path → fs read      |
| EXIF           | No                             | Yes                                |
| Font           | System font (Canvas API)       | JetBrains Mono Regular (bundled)   |
| New deps       | fflate (~12KB JS)              | ab_glyph + font (~25KB)           |
| Threading      | Web Worker                     | Async Tauri command                |

## Annotation Style

- **Bounding box**: 2-3px colored stroke (same category colors as on-screen)
- **Label**: species name (if available) or category name + confidence % (e.g. "coyote 94%", "person 87%")
- **Label background**: semi-transparent fill matching category color, white text
- **Font**: JetBrains Mono (Tauri) / system monospace (web)
- **Scaling**: line width and font size scale relative to image dimensions so annotations look proportional on any resolution

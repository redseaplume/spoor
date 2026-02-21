# Spoor

Camera trap image detection and species identification. Runs locally, nothing leaves your device.

*Spoor: the tracks and signs an animal leaves behind. From Afrikaans/Dutch.*

## What it does

Drop camera trap images into Spoor and it tells you what's there — animal, person, vehicle, or empty. For animal detections, it can identify the species.

- **Two detection models**: Quick (~180ms/image) for triage, Thorough (~1s/image) for precision
- **Species identification**: 2,000+ species via SpeciesNet
- **Export**: MegaDetector JSON v1.5 and CSV — compatible with Timelapse and the wider camera trap ecosystem
- **No install complexity**: no Python, no PyTorch, no 25GB downloads. One app.

## Download

macOS (Apple Silicon): see [Releases](https://github.com/redseaplume/spoor/releases)

Intel Macs are supported via Rosetta 2.

## Build from source

Requires [Rust](https://rustup.rs) and the ONNX models (not included in the repo due to size).

```
# Models go in models/
models/mdv5a_int8.onnx          # 134MB — MDv5a INT8
models/mdv6-yolov10-c.onnx      # 9.8MB — MDv6 YOLOv10-c
models/speciesnet-v4.0.2a.onnx  # 214MB — SpeciesNet v4
models/speciesnet-labels.txt     # Species label map

# Build
cd src-tauri
cargo tauri build --target aarch64-apple-darwin
```

## Detection models

- **Thorough** (MDv5a INT8): [MegaDetector v5a](https://github.com/agentmorris/MegaDetector) via [bencevans/megadetector-onnx](https://github.com/bencevans/megadetector-onnx). Tighter bounding boxes, higher recall.
- **Quick** (MDv6 YOLOv10-c): [MegaDetector v6](https://github.com/microsoft/CameraTraps) via Zenodo. 14x smaller, 6x faster.
- **Species** (SpeciesNet v4): [Google SpeciesNet](https://github.com/google/cameratrapai), Apache 2.0. EfficientNetV2-M, 2,498 labels.

## License

[PolyForm Noncommercial 1.0.0](LICENSE)

Spoor is free for researchers, conservationists, and anyone doing work that matters. It is not to be repackaged, sold, or used for commercial purposes. If you'd like to use it commercially, get in touch.

---

Built by Claude and Ezra.

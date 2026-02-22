# Spoor

Camera trap image detection and species identification. Runs locally, nothing leaves your device.

*Spoor: the tracks and signs an animal leaves behind.*

## What it does

Drop camera trap images into Spoor and it tells you what's there: animal, person, vehicle, or nothing. For animal detections, it identifies the species — 2,498 species worldwide, primarily mammals and birds. Coverage varies by region; the model was trained on camera trap datasets with broader representation in well-studied areas, and falls back to genus or family level when uncertain.

- **Two detection models**: Quick (~180ms/image) for triage, Thorough (~1s/image) for precision
- **Species identification**: 2,498 species via SpeciesNet
- **Export**: MegaDetector JSON v1.5 and CSV — compatible with Timelapse and the wider camera trap ecosystem
- **No install complexity**: no Python, no PyTorch, no 25GB downloads. One app.
- **Web version**: 10 MB model that runs in your browser. No sign-up or account needed.
- **Data privacy**: your data stays local.

## Download

See [Releases](https://github.com/redseaplume/spoor/releases) for the latest builds.

- **macOS**: Apple Silicon native. Intel Macs supported via Rosetta 2.
- **Windows**: 10 and later. WebView2 required (installs automatically).
- **Linux**: .deb and .rpm packages. Requires glibc 2.38+ and libwebkit2gtk-4.1 (Ubuntu 24.04+, Fedora 39+).

## Build from source

Requires [Rust](https://rustup.rs) and the ONNX models (not included in the repo due to size).

```
# Download models from the release
gh release download models-v1 -R redseaplume/spoor -p "*.onnx" -p "*.txt" -D models/

# Build
cd src-tauri
cargo tauri build
```

## Why Spoor

| | Spoor | CamTrap Detector | AddaxAI |
|---|---|---|---|
| **Install size** | 300 MB | 383–498 MB | 4.4 GB+ |
| **Dependencies** | None | None | Python, Conda, PyTorch |
| **Detection models** | 2 (MDv6 + MDv5a) | 1 (MDv5) | MDv5 + regional |
| **Species ID** | 2,498 species | No | Regional classifiers |
| **Web version** | Yes (10 MB model) | No | No |
| **Data stays local** | Yes | Yes | Yes |
| **Platforms** | macOS, Windows, Linux | Windows, macOS, Linux | Windows, macOS, Linux |

Spoor bundles three models (two detectors + species classifier) in a single app smaller than most single-model alternatives. The web version runs entirely in your browser with a 10 MB download. No account needed, and your data stays local.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for full methodology, results, and reproduction steps. The short version: we tested on 200 Caltech Camera Traps images against ground truth. MDv5a INT8 at native resolution hits 99.2% bbox recall and 98.4% precision. INT8 quantization loses almost nothing vs FP32 (99% precision, 99% recall, 0.959 IoU). Scripts to reproduce everything are in `scripts/`.

## Detection models

- **Thorough** (MDv5a INT8): [MegaDetector v5a](https://github.com/agentmorris/MegaDetector) via [bencevans/megadetector-onnx](https://github.com/bencevans/megadetector-onnx). Tighter bounding boxes, higher recall.
- **Quick** (MDv6 YOLOv10-c): [MegaDetector v6](https://github.com/microsoft/CameraTraps) via Zenodo. 14x smaller, 6x faster.
- **Species** (SpeciesNet v4): [Google SpeciesNet](https://github.com/google/cameratrapai), Apache 2.0. EfficientNetV2-M, 2,498 labels.

## License

[PolyForm Noncommercial 1.0.0](LICENSE)

Spoor is free for researchers, conservationists, and anyone doing work that matters. It is not to be repackaged, sold, or used for commercial purposes. If you'd like to use it commercially, get in touch.

---

Built by Claude and Ezra.

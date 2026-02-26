# Spoor

Camera trap image detection and species identification. Runs locally, nothing leaves your device.

## What it does

Drop camera trap images into Spoor and it tells you what's there: animal, person, vehicle, or nothing. For animal detections, it identifies the species using SpeciesNet v4: 2,498 species worldwide, primarily mammals and birds. Coverage varies by region.

- **Two detection models**: Quick (~180ms/image) for triage, Thorough (~1s/image) for precision. Thorough offers 1280px (native, highest recall) or 640px (~3.5x faster, ~9% fewer detections).
- **Species identification**: 2,498 species via SpeciesNet
- **Export**: [MegaDetector JSON v1.5](https://lila.science/megadetector-output-format) and CSV — compatible with Timelapse and the wider camera trap ecosystem. Species identifications are included in the JSON but use Spoor's own format, not the spec's `classifications` field.
- **No install complexity**: no Python, no PyTorch, no 25GB downloads. One app.
- **Web version**: MDv6 (10 MB) runs in your browser. No sign-up or account needed.
- **Data privacy**: your data stays local.

## Download

Grab the latest build for your platform from [Releases](https://github.com/redseaplume/spoor/releases).

### macOS

Download **Spoor_x.x.x_aarch64.dmg** and open it. A Finder window should appear — drag Spoor to the Applications folder. If no window appears, look for the **Spoor** volume in the Finder sidebar under Locations, or press `Cmd+Shift+G` and go to `/Volumes/Spoor`.

Spoor is not signed with an Apple Developer certificate. macOS will block the first launch. Go to **System Settings → Privacy & Security**, scroll down, and click **Open Anyway** next to the Spoor message. You only need to do this once.

Apple Silicon native. Intel Macs run via Rosetta 2.

### Windows

Download **Spoor_x.x.x_x64-setup.exe** and run the installer. Windows may show a SmartScreen warning ("Windows protected your PC") — click **More info**, then **Run anyway**. WebView2 is required and installs automatically if missing.

### Linux

Download the **.deb** or **.rpm** package.

```
# Debian/Ubuntu 24.04+
sudo dpkg -i spoor_x.x.x_amd64.deb

# Fedora 39+
sudo rpm -i spoor-x.x.x-1.x86_64.rpm
```

Requires glibc 2.38+ and libwebkit2gtk-4.1.

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
| **Species ID** | SpeciesNet v4 | No | Regional classifiers |
| **Web version** | Yes (MDv6, 10 MB) | No | No |
| **Data stays local** | Yes | Yes | Yes |
| **Platforms** | macOS, Windows, Linux | Windows, macOS, Linux | Windows, macOS, Linux |

Spoor bundles three models (two detectors + species classifier) in a single app smaller than most single-model alternatives. The web version runs entirely in your browser with a 10 MB download. No account needed, and your data stays local.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for full methodology, results, and reproduction steps. We quantized MDv5a from FP32 (534 MB) to INT8 (134 MB) using ONNX Runtime's dynamic quantization (`scripts/convert.py`). We tested on 200 labelled images from Caltech Camera Traps. MDv5a INT8 at native resolution hits 99.2% bbox recall and 98.4% precision. INT8 quantization loses almost nothing vs FP32 (99% precision, 99% recall, 0.959 IoU). Scripts to reproduce everything are in `scripts/`.

## Detection models

- **Quick** (MDv6 YOLOv10-c): [MegaDetector v6](https://github.com/microsoft/CameraTraps) via Zenodo. 14x smaller, 6x faster. Runs at 1280px.
- **Thorough** (MDv5a INT8): [MegaDetector v5a](https://github.com/agentmorris/MegaDetector) via [bencevans/megadetector-onnx](https://github.com/bencevans/megadetector-onnx). Tighter bounding boxes, higher recall. 1280px or 640px.
- **Species** (SpeciesNet v4): [Google SpeciesNet](https://github.com/google/cameratrapai), Apache 2.0. EfficientNetV2-M, 2,498 labels.

## License

[PolyForm Noncommercial 1.0.0](LICENSE)

Spoor is free for researchers, conservationists, and anyone doing work that matters. It is not to be repackaged, sold, or used for commercial purposes. If you'd like to use it commercially, get in touch.

---

Built by Claude and redseaplume.

# Benchmarks

The methodology/reasoning in testing Spoor's models, what was found, and what wasn't tested.

## Test data

200 images from [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/) (CCT), hosted on LILA. Selected for diversity: ~60% animal, ~30% empty, ~10% person, sampled with a fixed seed for reproducibility.

The selection and download script is in `scripts/download_test_data.py`. Ground truth comes from LILA's bounding box annotations in COCO format.

## Quantization

MegaDetector v5a ships as a 534 MB FP32 ONNX model. We quantized it to INT8 using ONNX Runtime's dynamic quantization (`quantize_dynamic` with QUInt8 weights), reducing it to 134 MB, a 4x size reduction.

INT8 quantization converts 32-bit floating point weights to 8-bit integers. For most architectures this has minimal impact on accuracy. We verified this by running both FP32 and INT8 models on the same 200 images and comparing outputs directly.

**We also tried INT8 quantization on SpeciesNet (EfficientNetV2-M). It was catastrophically bad:** predictions became essentially random. EfficientNet's architecture doesn't tolerate naive dynamic quantization. SpeciesNet ships as FP32 (214 MB).

The quantization script is in `scripts/convert.py`.

## INT8 vs FP32 accuracy

These numbers compare INT8 output directly against FP32 output on the same images, not against ground truth. This isolates the quantization impact from any model limitations.

### MDv5a at 1280x1280 (native resolution)

| | FP32 | INT8 |
|---|---|---|
| **Mean time** | 2,592 ms/image | 940 ms/image |
| **File size** | 534 MB | 134 MB |
| **Detections** | 216 | 217 |

INT8 vs FP32: **99% precision, 99% recall, 0.959 mean IoU, 2.76x speedup**.

Quantization lost almost nothing at native resolution.

### MDv5a at 640x640 (half resolution)

| | FP32 | INT8 |
|---|---|---|
| **Mean time** | 659 ms/image | 254 ms/image |
| **File size** | 534 MB | 134 MB |
| **Detections** | 196 | 198 |

INT8 vs FP32: **98.3% precision, 98.8% recall, 0.955 mean IoU, 2.59x speedup**.

Slightly more divergence at lower resolution, but still negligible.

## Detection accuracy vs ground truth

These numbers compare model predictions against Caltech Camera Traps ground truth annotations. This measures how well the models actually detect animals, not just how close INT8 is to FP32.

### MDv5a INT8

| Resolution | Image recall | Bbox recall | Bbox precision | Mean IoU |
|---|---|---|---|---|
| 1280x1280 | — | 99.2% | 98.4% | 0.865 |
| 640x640 | — | 90.3% | 98.2% | 0.835 |

1280 is the native resolution MDv5a was trained at. 640 trades recall for speed — it misses ~9% of bounding boxes but is nearly 4x faster.

### MDv6 YOLOv10-c

| Resolution | Image recall | Bbox recall | Bbox precision | Mean IoU |
|---|---|---|---|---|
| 1280x1280 | 97.5% (@0.1 conf) | 92.1% (@0.2 conf) | — | 0.787 |

MDv6 is a fundamentally different architecture (YOLOv10-c, NMS-free) and 14x smaller than MDv5a. It trades bounding box tightness for speed: 183 ms/image vs ~940 ms for MDv5a INT8. The lower IoU reflects looser boxes, not missed detections.

## Speed

All timings on Apple M-series CPU (single-threaded ONNX Runtime, no GPU). Python benchmarks — Tauri/Rust adds ~15% overhead from the `image` crate's decode/resize vs OpenCV.

| Model | Resolution | Mean time | Images/sec |
|---|---|---|---|
| MDv5a FP32 | 1280 | 2,592 ms | 0.4 |
| MDv5a INT8 | 1280 | 940 ms | 1.1 |
| MDv5a INT8 | 640 | 254 ms | 3.9 |
| MDv6 YOLOv10-c | 1280 | 183 ms | 5.5 |
| SpeciesNet v4 | 224 | ~240 ms | ~4.2 |

SpeciesNet timing is per crop in the Tauri release build, not from the Python benchmark.

## What we didn't test

Being honest about the gaps:

- **Species classification**: SpeciesNet comes from Google and was validated in their paper. We tested it for speed and quantization tolerance but did not independently evaluate species-level accuracy. Coverage likely varies by region.
- **Geolocation features**: The full Google SpeciesNet pipeline can use latitude/longitude to narrow species predictions. Spoor uses the vision model only — no geolocation input.
- **Edge cases**: Very small animals, partially occluded subjects, unusual lighting, non-standard camera trap setups. Our 200-image test set is too small to characterize tail behavior.
- **Cross-platform performance**: All benchmarks are from a single Mac. Windows and Linux timings may differ.

## Reproducing

```
# Download test data (200 images from Caltech Camera Traps)
uv run scripts/download_test_data.py

# Run quantization (downloads FP32 model, produces INT8)
uv run scripts/convert.py --model mdv5a --output ./models/

# Benchmark INT8 vs FP32
uv run scripts/benchmark.py --images ./test_data/images/ --models ./models/ --output benchmark_results.json

# Evaluate against ground truth
uv run scripts/evaluate.py --images ./test_data/images/ --ground-truth ./test_data/ground_truth.json --model ./models/mdv5a_int8.onnx
```

## Scripts

- `scripts/download_test_data.py` — Downloads test images and ground truth from LILA
- `scripts/convert.py` — Downloads FP32 ONNX model, quantizes to INT8, validates output
- `scripts/benchmark.py` — Benchmarks FP32 vs INT8 (speed and output comparison)
- `scripts/evaluate.py` — Evaluates model accuracy against ground truth annotations

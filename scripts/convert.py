"""
sieve/convert.py — Prepare optimized ONNX models for MegaDetector inference.

This script handles:
  1. Downloading a pre-exported ONNX model (from bencevans/megadetector-onnx)
  2. Quantizing to INT8 (dynamic quantization via ONNX Runtime)
  3. Validating that FP32 and INT8 outputs are close

The pre-exported ONNX approach avoids the fragile PyTorch/YOLOv5 dependency chain
entirely — you only need onnxruntime, no torch or ultralytics.

Usage:
  python convert.py --model mdv5a --output ./models/
  python convert.py --onnx-input /path/to/existing.onnx --output ./models/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# Pre-exported ONNX models (from bencevans/megadetector-onnx)
# These are clean YOLOv5 exports without baked-in NMS, dynamic input sizes.
KNOWN_MODELS = {
    "mdv5a": {
        "onnx_url": "https://github.com/bencevans/megadetector-onnx/releases/download/v0.2.0/md_v5a.0.0-dynamic.onnx",
        "onnx_filename": "mdv5a_fp32.onnx",
        "architecture": "yolov5x6",
        "input_size": (1280, 1280),
    },
    "mdv5b": {
        "onnx_url": "https://github.com/bencevans/megadetector-onnx/releases/download/v0.2.0/md_v5b.0.0-dynamic.onnx",
        "onnx_filename": "mdv5b_fp32.onnx",
        "architecture": "yolov5x6",
        "input_size": (1280, 1280),
    },
}


def download_onnx_model(model_id: str, output_dir: Path) -> Path:
    """Download a pre-exported ONNX model if not already present."""
    import urllib.request

    info = KNOWN_MODELS[model_id]
    dest = output_dir / info["onnx_filename"]

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"ONNX model already exists: {dest} ({size_mb:.0f} MB)")
        return dest

    print(f"Downloading {model_id} ONNX model...")
    print(f"  Source: bencevans/megadetector-onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(info["onnx_url"], dest, reporthook=_progress)
    print()  # newline after progress
    return dest


def quantize_to_int8(onnx_path: Path, quantized_path: Path) -> Path:
    """Quantize an ONNX model to INT8 using dynamic quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"Quantizing to INT8: {quantized_path}")
    quantized_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QUInt8,
    )

    print(f"INT8 quantization complete: {quantized_path}")
    return quantized_path


def measure_file_sizes(paths: dict[str, Path]) -> dict[str, float]:
    """Measure file sizes in MB."""
    sizes = {}
    for name, path in paths.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            sizes[name] = size_mb
    return sizes


def validate_outputs(onnx_path: Path, int8_path: Path, input_size: tuple[int, int] = (1280, 1280)):
    """Run a dummy input through FP32 and INT8 ONNX models and compare outputs."""
    import onnxruntime as ort

    h, w = input_size
    np.random.seed(42)
    test_input = np.random.randn(1, 3, h, w).astype(np.float32)

    # Get input name from the model
    sess_fp32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    print(f"Running ONNX FP32 inference (input: {input_name})...")
    fp32_output = sess_fp32.run(None, {input_name: test_input})[0]

    print("Running ONNX INT8 inference...")
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    int8_output = sess_int8.run(None, {input_name: test_input})[0]

    # Compare outputs
    diff = np.abs(fp32_output - int8_output)
    print(f"\nINT8 vs FP32 comparison:")
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Output shape: {fp32_output.shape}")

    return {
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "output_shape": list(fp32_output.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare optimized ONNX models for MegaDetector")
    parser.add_argument(
        "--model",
        default="mdv5a",
        choices=list(KNOWN_MODELS.keys()),
        help="Model identifier (downloads pre-exported ONNX from bencevans/megadetector-onnx)",
    )
    parser.add_argument(
        "--onnx-input",
        default=None,
        help="Path to an existing ONNX file to quantize (skips download)",
    )
    parser.add_argument(
        "--output",
        default="./models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip output validation",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=None,
        help="Input size for validation (height width). Defaults to model's native size.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Resolve ONNX FP32 path
    if args.onnx_input:
        onnx_path = Path(args.onnx_input)
        if not onnx_path.exists():
            print(f"Error: ONNX file not found: {onnx_path}", file=sys.stderr)
            sys.exit(1)
        model_name = onnx_path.stem.replace("_fp32", "").replace("_dynamic", "")
        input_size = tuple(args.input_size) if args.input_size else (1280, 1280)
    else:
        model_info = KNOWN_MODELS[args.model]
        model_name = args.model
        input_size = tuple(args.input_size) if args.input_size else model_info["input_size"]

        # Step 1: Download ONNX FP32
        print("\n=== Step 1: Download ONNX FP32 ===")
        t0 = time.time()
        onnx_path = download_onnx_model(args.model, output_dir)
        print(f"  Done in {time.time() - t0:.1f}s")

    int8_path = output_dir / f"{model_name}_int8.onnx"

    # Step 2: Quantize to INT8
    print("\n=== Step 2: ONNX FP32 → ONNX INT8 ===")
    t0 = time.time()
    quantize_to_int8(onnx_path, int8_path)
    print(f"  Quantization took {time.time() - t0:.1f}s")

    # Step 3: Compare file sizes
    print("\n=== Step 3: File sizes ===")
    sizes = measure_file_sizes({
        "ONNX FP32": onnx_path,
        "ONNX INT8": int8_path,
    })
    for name, size_mb in sizes.items():
        print(f"  {name}: {size_mb:.1f} MB")

    fp32_size = sizes.get("ONNX FP32", 0)
    int8_size = sizes.get("ONNX INT8", 0)
    if fp32_size > 0 and int8_size > 0:
        ratio = int8_size / fp32_size
        print(f"  INT8 is {ratio:.1%} of FP32 size ({(1-ratio)*100:.0f}% reduction)")

    # Step 4: Validate outputs
    if not args.skip_validation:
        print("\n=== Step 4: Output validation ===")
        diffs = validate_outputs(onnx_path, int8_path, input_size)
    else:
        print("\n=== Step 4: Skipped (--skip-validation) ===")

    print("\n=== Done ===")
    print(f"Models saved to: {output_dir}")
    print(f"  FP32: {onnx_path}")
    print(f"  INT8: {int8_path}")


if __name__ == "__main__":
    main()

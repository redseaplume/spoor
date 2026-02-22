"""
sieve/benchmark.py — Benchmark MegaDetector across PyTorch, ONNX FP32, and ONNX INT8.

Compares accuracy and speed on a set of test images.

Usage:
  python benchmark.py --images ./test_images/ --models ./models/ --model-name mdv5a
  python benchmark.py --images ./test_images/ --models ./models/ --model-name mdv5a --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def load_and_preprocess(image_path: Path, input_size: tuple[int, int] = (1280, 1280)) -> np.ndarray:
    """Load an image and preprocess for YOLO inference."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    target_h, target_w = input_size

    # Letterbox resize (maintain aspect ratio, pad with gray)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    # HWC BGR → CHW RGB, normalized to [0, 1]
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, axis=0)


def nms(boxes, scores, iou_threshold=0.45):
    """Simple NMS implementation."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess_yolo(output: np.ndarray, conf_threshold: float = 0.1, iou_threshold: float = 0.45):
    """Post-process YOLOv5 output into detections.

    YOLOv5 output shape: (batch, num_predictions, 5 + num_classes)
    For MegaDetector: 5 + 3 classes (animal, person, vehicle)
    Columns: [cx, cy, w, h, obj_conf, cls0_conf, cls1_conf, cls2_conf]
    """
    if output.ndim == 3:
        output = output[0]  # Remove batch dimension

    # Filter by objectness confidence
    obj_conf = output[:, 4]
    mask = obj_conf > conf_threshold
    filtered = output[mask]

    if len(filtered) == 0:
        return []

    # Get class scores = obj_conf * class_conf
    class_confs = filtered[:, 5:] * filtered[:, 4:5]
    class_ids = class_confs.argmax(axis=1)
    scores = class_confs.max(axis=1)

    # Convert cx, cy, w, h → x1, y1, x2, y2
    cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS per class
    detections = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        keep = nms(cls_boxes, cls_scores, iou_threshold)
        for idx in keep:
            detections.append({
                "bbox": cls_boxes[idx].tolist(),
                "confidence": float(cls_scores[idx]),
                "category": int(cls_id),
            })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# MegaDetector category mapping
MD_CATEGORIES = {0: "animal", 1: "person", 2: "vehicle"}


def run_onnx_inference(session, image_path: Path, input_size=(1280, 1280), conf_threshold=0.1):
    """Run inference with an ONNX Runtime session."""
    blob = load_and_preprocess(image_path, input_size)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]
    detections = postprocess_yolo(output, conf_threshold=conf_threshold)
    return detections


def run_pytorch_inference(model, image_path: Path, conf_threshold=0.1):
    """Run inference with the PyTorch MegaDetector model."""
    import torch

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # MegaDetector's own inference pipeline
    model.conf = conf_threshold
    results = model(img[:, :, ::-1])  # BGR → RGB

    detections = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        detections.append({
            "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
            "confidence": float(conf),
            "category": int(cls),
        })

    return detections


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def compare_detections(ref_dets, test_dets, iou_threshold=0.5):
    """Compare test detections against reference detections.

    Returns precision, recall, and average IoU for matched detections.
    """
    if len(ref_dets) == 0 and len(test_dets) == 0:
        return {"precision": 1.0, "recall": 1.0, "avg_iou": 1.0, "matched": 0, "missed": 0, "extra": 0}
    if len(ref_dets) == 0:
        return {"precision": 0.0, "recall": 1.0, "avg_iou": 0.0, "matched": 0, "missed": 0, "extra": len(test_dets)}
    if len(test_dets) == 0:
        return {"precision": 1.0, "recall": 0.0, "avg_iou": 0.0, "matched": 0, "missed": len(ref_dets), "extra": 0}

    matched_ref = set()
    matched_test = set()
    ious = []

    for i, test_det in enumerate(test_dets):
        best_iou = 0
        best_j = -1
        for j, ref_det in enumerate(ref_dets):
            if j in matched_ref:
                continue
            if test_det["category"] != ref_det["category"]:
                continue
            iou = compute_iou(test_det["bbox"], ref_det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            matched_ref.add(best_j)
            matched_test.add(i)
            ious.append(best_iou)

    n_matched = len(matched_ref)
    n_missed = len(ref_dets) - n_matched
    n_extra = len(test_dets) - n_matched

    precision = n_matched / len(test_dets) if test_dets else 1.0
    recall = n_matched / len(ref_dets) if ref_dets else 1.0
    avg_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "avg_iou": avg_iou,
        "matched": n_matched,
        "missed": n_missed,
        "extra": n_extra,
    }


def find_images(directory: Path, limit: int = 0) -> list[Path]:
    """Find image files in a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in extensions
    )
    if limit > 0:
        images = images[:limit]
    return images


def main():
    parser = argparse.ArgumentParser(description="Benchmark MegaDetector: PyTorch vs ONNX vs INT8")
    parser.add_argument("--images", required=True, help="Directory containing test images")
    parser.add_argument("--models", default="./models", help="Directory containing model files")
    parser.add_argument("--model-name", default="mdv5a", help="Model name prefix (e.g., mdv5a)")
    parser.add_argument("--conf-threshold", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--quick", action="store_true", help="Quick mode: limit to 50 images")
    parser.add_argument("--limit", type=int, default=0, help="Max number of images to process")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--input-size", type=int, nargs=2, default=[1280, 1280],
                        help="Input size (height width), default: 1280 1280")
    args = parser.parse_args()

    images_dir = Path(args.images)
    models_dir = Path(args.models)

    if not images_dir.exists():
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    limit = args.limit or (50 if args.quick else 0)
    images = find_images(images_dir, limit=limit)
    print(f"Found {len(images)} images in {images_dir}")

    if not images:
        print("No images found. Exiting.")
        sys.exit(1)

    # Resolve model paths
    onnx_fp32_path = models_dir / f"{args.model_name}_fp32.onnx"
    onnx_int8_path = models_dir / f"{args.model_name}_int8.onnx"

    input_size = tuple(args.input_size)
    results = {
        "num_images": len(images),
        "conf_threshold": args.conf_threshold,
        "model_name": args.model_name,
        "input_size": list(input_size),
        "benchmarks": {},
    }
    print(f"Input size: {input_size[0]}x{input_size[1]}")

    # --- ONNX Benchmarks ---
    # FP32 serves as the reference baseline (instead of PyTorch)
    fp32_detections = {}
    import onnxruntime as ort

    for label, model_path in [("onnx_fp32", onnx_fp32_path), ("onnx_int8", onnx_int8_path)]:
        if not model_path.exists():
            print(f"\n{label} model not found at {model_path}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"ONNX {label.upper()} Benchmark ({model_path.name})")
        print(f"  File size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"{'='*60}")

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        # Warmup (3 images)
        for img_path in images[:min(3, len(images))]:
            run_onnx_inference(sess, img_path, input_size=input_size, conf_threshold=args.conf_threshold)

        times = []
        onnx_detections = {}
        for img_path in images:
            t0 = time.perf_counter()
            dets = run_onnx_inference(sess, img_path, input_size=input_size, conf_threshold=args.conf_threshold)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            onnx_detections[str(img_path)] = dets

        times_arr = np.array(times)
        bench = {
            "total_time_s": float(times_arr.sum()),
            "mean_time_s": float(times_arr.mean()),
            "median_time_s": float(np.median(times_arr)),
            "std_time_s": float(times_arr.std()),
            "images_per_second": float(len(images) / times_arr.sum()),
            "file_size_mb": model_path.stat().st_size / (1024 * 1024),
            "num_detections": sum(len(d) for d in onnx_detections.values()),
        }

        # Store FP32 detections as reference baseline
        if label == "onnx_fp32":
            fp32_detections = onnx_detections

        # Compare INT8 against FP32 baseline
        if label == "onnx_int8" and fp32_detections:
            precisions, recalls, avg_ious = [], [], []
            for img_path in images:
                key = str(img_path)
                if key in fp32_detections:
                    comp = compare_detections(
                        fp32_detections[key],
                        onnx_detections.get(key, []),
                    )
                    precisions.append(comp["precision"])
                    recalls.append(comp["recall"])
                    if comp["avg_iou"] > 0:
                        avg_ious.append(comp["avg_iou"])

            bench["vs_fp32"] = {
                "mean_precision": float(np.mean(precisions)) if precisions else None,
                "mean_recall": float(np.mean(recalls)) if recalls else None,
                "mean_iou": float(np.mean(avg_ious)) if avg_ious else None,
                "speedup": (
                    results["benchmarks"]["onnx_fp32"]["mean_time_s"] / bench["mean_time_s"]
                    if "onnx_fp32" in results["benchmarks"] else None
                ),
            }

        results["benchmarks"][label] = bench

        print(f"  Total: {times_arr.sum():.1f}s for {len(images)} images")
        print(f"  Mean:  {times_arr.mean()*1000:.0f}ms/image")
        print(f"  Speed: {len(images)/times_arr.sum():.1f} images/sec")
        print(f"  Detections: {bench['num_detections']}")

        if "vs_fp32" in bench and bench["vs_fp32"]["speedup"]:
            vs = bench["vs_fp32"]
            print(f"\n  vs FP32:")
            print(f"    Speedup:   {vs['speedup']:.1f}x")
            print(f"    Precision: {vs['mean_precision']:.3f}")
            print(f"    Recall:    {vs['mean_recall']:.3f}")
            if vs["mean_iou"]:
                print(f"    Mean IoU:  {vs['mean_iou']:.3f}")

        del sess

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, bench in results["benchmarks"].items():
        size_str = f", {bench['file_size_mb']:.0f}MB" if "file_size_mb" in bench else ""
        print(f"  {name:12s}: {bench['mean_time_s']*1000:6.0f}ms/img, {bench['images_per_second']:.1f} img/s{size_str}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

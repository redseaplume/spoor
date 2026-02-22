"""
sieve/evaluate.py — Evaluate ONNX MegaDetector models against ground truth.

Scores models on:
  1. Binary classification: animal present vs empty (precision, recall, F1)
  2. Detection: bounding box overlap with ground truth annotations

Usage:
  python evaluate.py --images ./test_data/images/ --ground-truth ./test_data/ground_truth.json --model ./models/mdv5a_int8.onnx
  python evaluate.py --images ./test_data/images/ --ground-truth ./test_data/ground_truth.json --model ./models/mdv5a_fp32.onnx --input-size 640 640
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_and_preprocess(image_path: Path, input_size: tuple[int, int] = (1280, 1280)) -> np.ndarray:
    """Load an image and preprocess for YOLO inference."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    target_h, target_w = input_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, axis=0)


def nms(boxes, scores, iou_threshold=0.45):
    """Simple NMS."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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


def postprocess_yolo(output, conf_threshold=0.1):
    """Post-process YOLOv5 output into detections."""
    if output.ndim == 3:
        output = output[0]
    obj_conf = output[:, 4]
    mask = obj_conf > conf_threshold
    filtered = output[mask]
    if len(filtered) == 0:
        return []
    class_confs = filtered[:, 5:] * filtered[:, 4:5]
    class_ids = class_confs.argmax(axis=1)
    scores = class_confs.max(axis=1)
    cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    detections = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep = nms(cls_boxes, cls_scores)
        for idx in keep:
            detections.append({
                "bbox": cls_boxes[idx].tolist(),
                "confidence": float(cls_scores[idx]),
                "category": int(cls_id),
            })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


MD_CATEGORIES = {0: "animal", 1: "person", 2: "vehicle"}

# Ground truth categories that count as "animal" for MegaDetector
NON_ANIMAL_CATEGORIES = {"empty", "person", "human", "vehicle", "car", "truck"}


def rescale_detections(detections, input_size, orig_w, orig_h):
    """Convert detection boxes from model input coords back to original image coords."""
    target_h, target_w = input_size
    scale = min(target_w / orig_w, target_h / orig_h)
    pad_left = (target_w - int(orig_w * scale)) / 2
    pad_top = (target_h - int(orig_h * scale)) / 2

    rescaled = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale
        rescaled.append({
            **det,
            "bbox": [x1, y1, x2, y2],
        })
    return rescaled


def compute_iou(box1, box2):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def coco_to_xyxy(bbox):
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def main():
    parser = argparse.ArgumentParser(description="Evaluate MegaDetector ONNX models against ground truth")
    parser.add_argument("--images", required=True, help="Directory containing test images")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth.json")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--input-size", type=int, nargs=2, default=[1280, 1280],
                        help="Input size (height width)")
    parser.add_argument("--conf-threshold", type=float, default=0.2,
                        help="Confidence threshold for detections")
    args = parser.parse_args()

    images_dir = Path(args.images)
    model_path = Path(args.model)
    input_size = tuple(args.input_size)

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    print(f"Model: {model_path.name}")
    print(f"Input size: {input_size[0]}x{input_size[1]}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Ground truth images: {len(ground_truth)}")

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # --- Run inference on all images ---
    # Binary classification stats
    tp = fp = fn = tn = 0
    # Detection matching stats
    det_matched = det_missed = det_extra = 0
    ious = []
    conf_threshold = args.conf_threshold

    evaluated = 0
    for filename, gt_info in ground_truth.items():
        img_path = images_dir / filename
        if not img_path.exists():
            continue

        orig_w = gt_info.get("width", 2048)
        orig_h = gt_info.get("height", 1494)

        # Run inference
        blob = load_and_preprocess(img_path, input_size)
        output = sess.run(None, {input_name: blob})[0]
        detections = postprocess_yolo(output, conf_threshold=conf_threshold)
        detections = rescale_detections(detections, input_size, orig_w, orig_h)

        # Filter to animal detections only (class 0)
        animal_dets = [d for d in detections if d["category"] == 0]
        gt_has_animal = gt_info["has_animal"]
        pred_has_animal = len(animal_dets) > 0

        # Binary classification
        if gt_has_animal and pred_has_animal:
            tp += 1
        elif gt_has_animal and not pred_has_animal:
            fn += 1
        elif not gt_has_animal and pred_has_animal:
            fp += 1
        else:
            tn += 1

        # Bounding box matching (for images with GT boxes)
        gt_boxes = []
        for ann in gt_info["annotations"]:
            if ann["category"].lower() not in NON_ANIMAL_CATEGORIES and "bbox" in ann:
                gt_boxes.append(coco_to_xyxy(ann["bbox"]))

        if gt_boxes:
            matched_gt = set()
            for det in animal_dets:
                best_iou = 0
                best_j = -1
                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = compute_iou(det["bbox"], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= 0.3 and best_j >= 0:  # Lower threshold — camera trap boxes are approximate
                    matched_gt.add(best_j)
                    ious.append(best_iou)
                    det_matched += 1
                else:
                    det_extra += 1
            det_missed += len(gt_boxes) - len(matched_gt)

        evaluated += 1
        if evaluated % 50 == 0:
            print(f"  Processed {evaluated}/{len(ground_truth)} images...")

    # --- Results ---
    print(f"\n{'='*60}")
    print(f"RESULTS — {model_path.name} @ {input_size[0]}x{input_size[1]}")
    print(f"{'='*60}")

    total = tp + fp + fn + tn
    print(f"\n  Binary classification (animal present?)")
    print(f"  {'—'*40}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    print(f"    True positives:  {tp:4d}  (animal correctly found)")
    print(f"    True negatives:  {tn:4d}  (empty correctly identified)")
    print(f"    False positives: {fp:4d}  (empty image, model said animal)")
    print(f"    False negatives: {fn:4d}  (animal present, model missed)")
    print(f"    —")
    print(f"    Accuracy:  {accuracy:.1%}")
    print(f"    Precision: {precision:.1%}")
    print(f"    Recall:    {recall:.1%}")
    print(f"    F1 score:  {f1:.1%}")

    print(f"\n  Bounding box detection")
    print(f"  {'—'*40}")
    total_gt = det_matched + det_missed
    total_pred = det_matched + det_extra
    det_precision = det_matched / total_pred if total_pred > 0 else 0
    det_recall = det_matched / total_gt if total_gt > 0 else 0
    mean_iou = float(np.mean(ious)) if ious else 0

    print(f"    GT boxes:     {total_gt}")
    print(f"    Pred boxes:   {total_pred}")
    print(f"    Matched:      {det_matched}")
    print(f"    Missed:       {det_missed}")
    print(f"    Extra:        {det_extra}")
    print(f"    —")
    print(f"    Precision: {det_precision:.1%}")
    print(f"    Recall:    {det_recall:.1%}")
    print(f"    Mean IoU:  {mean_iou:.3f}")


if __name__ == "__main__":
    main()

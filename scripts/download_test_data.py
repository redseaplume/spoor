"""
sieve/download_test_data.py — Download a small test set of camera trap images for benchmarking.

Downloads images from LILA (Labeled Information Library of Alexandria) — public camera
trap datasets hosted on Google Cloud Storage.

Usage:
  python download_test_data.py                    # Download default ~200 image test set
  python download_test_data.py --count 50         # Quick smoke test (50 images)
  python download_test_data.py --count 500        # Larger validation set
  python download_test_data.py --dataset cct      # Caltech Camera Traps (default)
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


# LILA dataset URLs — verified from lila_camera_trap_datasets.csv
# Images are individually accessible via HTTP (no auth needed)
DATASETS = {
    "cct": {
        "name": "Caltech Camera Traps",
        # Unzipped images on GCP — individual files accessible via HTTP
        "image_base_url": "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/",
        # Annotations zip (contains COCO Camera Traps JSON)
        "annotations_url": "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip",
        # Bounding box annotations (direct JSON, no zip)
        "bbox_url": "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_bboxes_20200316.json",
        # Pre-computed MegaDetector v5a results (for baseline comparison)
        "md_results_url": "https://lila.science/public/lila-md-results/caltech-camera-traps_mdv5a.0.0_results.filtered_rde_0.150_0.850_10_0.300.json.zip",
        "description": "244K images from 140 camera locations in Southwestern US, 22 species",
    },
}


def download_file(url: str, dest: Path, quiet: bool = False) -> bool:
    """Download a file from URL to dest. Returns True on success."""
    try:
        if not quiet:
            print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        if not quiet:
            print(f"  Failed: {e}")
        return False


def download_and_extract_json(url: str, dest: Path, label: str = "file") -> Path | None:
    """Download a JSON file (or zip containing JSON) and save it."""
    if dest.exists():
        print(f"  {label} already exists: {dest}")
        return dest

    print(f"  Downloading {label}...")

    if url.endswith(".zip"):
        import zipfile
        import tempfile
        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        if not download_file(url, zip_path):
            return None
        with zipfile.ZipFile(zip_path) as zf:
            json_files = [f for f in zf.namelist() if f.endswith(".json")]
            if json_files:
                with zf.open(json_files[0]) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
        zip_path.unlink()
    else:
        if not download_file(url, dest):
            return None

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Saved: {dest} ({size_mb:.1f} MB)")
    return dest


def download_annotations(dataset_id: str, output_dir: Path) -> Path | None:
    """Download annotations JSON for a dataset."""
    dataset = DATASETS[dataset_id]

    # Prefer bounding box annotations (direct JSON, has bbox + category)
    bbox_url = dataset.get("bbox_url")
    if bbox_url:
        bbox_path = output_dir / f"{dataset_id}_bboxes.json"
        result = download_and_extract_json(bbox_url, bbox_path, "bounding box annotations")
        if result:
            return result

    # Fall back to main annotations (may be in a zip)
    annotations_url = dataset.get("annotations_url")
    if annotations_url:
        ann_path = output_dir / f"{dataset_id}_annotations.json"
        return download_and_extract_json(annotations_url, ann_path, "annotations")

    print(f"No annotation URLs configured for {dataset_id}")
    return None


def select_diverse_images(annotations_path: Path, count: int) -> list[dict]:
    """Select a diverse set of images from annotations.

    Tries to get a mix of:
    - Images with animals
    - Images with people
    - Empty images
    - Different locations
    """
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path) as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}

    # Build image_id → annotations lookup
    image_annotations = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Categorize images
    animal_images = []
    person_images = []
    empty_images = []
    other_images = []

    for img in images:
        img_id = img["id"]
        anns = image_annotations.get(img_id, [])
        if not anns:
            empty_images.append(img)
        else:
            cat_names = {categories.get(a.get("category_id"), "unknown").lower() for a in anns}
            if "person" in cat_names or "human" in cat_names:
                person_images.append(img)
            elif "empty" in cat_names:
                empty_images.append(img)
            else:
                animal_images.append(img)

    print(f"  Dataset: {len(images)} images total")
    print(f"  Animals: {len(animal_images)}, People: {len(person_images)}, Empty: {len(empty_images)}")

    # Select a balanced subset
    import random
    random.seed(42)  # Reproducible selection

    # Aim for ~60% animal, ~10% person, ~30% empty (roughly mirrors real camera trap data)
    n_animal = max(1, int(count * 0.6))
    n_person = max(1, min(int(count * 0.1), len(person_images)))
    n_empty = count - n_animal - n_person

    selected = []
    if animal_images:
        selected.extend(random.sample(animal_images, min(n_animal, len(animal_images))))
    if person_images:
        selected.extend(random.sample(person_images, min(n_person, len(person_images))))
    if empty_images:
        selected.extend(random.sample(empty_images, min(n_empty, len(empty_images))))

    # If we don't have enough, fill from whatever's available
    remaining = count - len(selected)
    if remaining > 0:
        all_available = [img for img in images if img not in selected]
        selected.extend(random.sample(all_available, min(remaining, len(all_available))))

    random.shuffle(selected)
    print(f"  Selected {len(selected)} images for download")
    return selected


def download_images(
    dataset_id: str,
    selected_images: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Download selected images from LILA."""
    dataset = DATASETS[dataset_id]
    base_url = dataset["image_base_url"]
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    failed = 0

    for i, img in enumerate(selected_images):
        filename = img["file_name"]
        # Flatten directory structure for simplicity
        safe_name = filename.replace("/", "_").replace("\\", "_")
        dest = images_dir / safe_name

        if dest.exists():
            downloaded.append(dest)
            continue

        url = base_url + filename
        success = download_file(url, dest, quiet=True)

        if success:
            downloaded.append(dest)
        else:
            failed += 1

        # Progress
        if (i + 1) % 20 == 0 or i == len(selected_images) - 1:
            print(f"  Downloaded {len(downloaded)}/{i+1} images ({failed} failed)")

    return downloaded


def save_ground_truth(
    selected_images: list[dict],
    annotations_path: Path,
    output_dir: Path,
) -> Path:
    """Save ground truth annotations for selected images in a simple format."""
    with open(annotations_path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}

    # Build annotation lookup
    image_annotations = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Build ground truth
    ground_truth = {}
    for img in selected_images:
        filename = img["file_name"].replace("/", "_").replace("\\", "_")
        anns = image_annotations.get(img["id"], [])

        gt_entries = []
        for ann in anns:
            entry = {
                "category": categories.get(ann.get("category_id"), "unknown"),
            }
            if "bbox" in ann:
                entry["bbox"] = ann["bbox"]  # COCO format: [x, y, width, height]
            gt_entries.append(entry)

        ground_truth[filename] = {
            "annotations": gt_entries,
            "has_animal": any(
                e["category"].lower() not in ("empty", "person", "human", "vehicle")
                for e in gt_entries
            ),
            "width": img.get("width"),
            "height": img.get("height"),
        }

    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Ground truth saved: {gt_path} ({len(ground_truth)} images)")
    return gt_path


def main():
    parser = argparse.ArgumentParser(description="Download camera trap test images from LILA")
    parser.add_argument("--dataset", default="cct", choices=list(DATASETS.keys()),
                        help="Dataset to download from")
    parser.add_argument("--count", type=int, default=200,
                        help="Number of images to download")
    parser.add_argument("--output", default="./test_data",
                        help="Output directory")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List available datasets and exit")
    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets:")
        for ds_id, ds in DATASETS.items():
            print(f"  {ds_id:12s}: {ds['name']}")
            print(f"               {ds['description']}")
        sys.exit(0)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DATASETS[args.dataset]
    print(f"Dataset: {dataset['name']}")
    print(f"Target:  {args.count} images → {output_dir}/")

    # Step 1: Download annotations
    annotations_path = download_annotations(args.dataset, output_dir)
    if not annotations_path:
        print("Failed to download annotations. Exiting.")
        sys.exit(1)

    # Step 2: Select diverse images
    selected = select_diverse_images(annotations_path, args.count)

    # Step 3: Download images
    print(f"\nDownloading {len(selected)} images...")
    downloaded = download_images(args.dataset, selected, output_dir)

    # Step 4: Save ground truth for selected images
    save_ground_truth(selected, annotations_path, output_dir)

    print(f"\nDone! {len(downloaded)} images downloaded to {output_dir}/images/")
    print(f"Ground truth: {output_dir}/ground_truth.json")
    print(f"\nNext steps:")
    print(f"  python convert.py --model mdv5a --output ./models/")
    print(f"  python benchmark.py --images {output_dir}/images/ --models ./models/")


if __name__ == "__main__":
    main()

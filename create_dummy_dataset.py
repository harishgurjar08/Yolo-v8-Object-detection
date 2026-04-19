"""
create_dummy_dataset.py
=======================
Generates a dummy YOLO-format dataset for testing and demonstration.
Run this once before training to create the folder structure and sample
annotation files. Real images are NOT included (they must be added by you),
but placeholder .txt annotation files are created so the structure is clear.

Usage:
    python create_dummy_dataset.py
"""

import os
import random

# ── Folder structure ──────────────────────────────────────────────────────────
SPLITS = ["train", "val", "test"]
BASE   = "dataset"

def create_folders():
    """Create the full YOLO dataset directory tree."""
    for split in SPLITS:
        os.makedirs(os.path.join(BASE, "images", split), exist_ok=True)
        os.makedirs(os.path.join(BASE, "labels", split), exist_ok=True)
    print("✅ Folder structure created:")
    print(f"""
    {BASE}/
    ├── images/
    │   ├── train/   ← put your training images here (.jpg / .png)
    │   ├── val/     ← put your validation images here
    │   └── test/    ← put your test images here (optional)
    └── labels/
        ├── train/   ← YOLO .txt annotations for train images
        ├── val/     ← YOLO .txt annotations for val images
        └── test/    ← YOLO .txt annotations for test images
    """)


def random_bbox():
    """Generate a random valid normalized bounding box (x, y, w, h)."""
    x = round(random.uniform(0.1, 0.9), 4)
    y = round(random.uniform(0.1, 0.9), 4)
    w = round(random.uniform(0.05, min(0.4, 1 - x)), 4)
    h = round(random.uniform(0.05, min(0.4, 1 - y)), 4)
    return x, y, w, h


def create_dummy_labels(split: str, count: int, num_classes: int = 3):
    """
    Create 'count' dummy .txt label files for a given split.
    Each file contains 1–3 random annotations.
    """
    label_dir = os.path.join(BASE, "labels", split)
    for i in range(count):
        filename = os.path.join(label_dir, f"image_{i:04d}.txt")
        num_objects = random.randint(1, 3)
        lines = []
        for _ in range(num_objects):
            cls = random.randint(0, num_classes - 1)
            x, y, w, h = random_bbox()
            lines.append(f"{cls} {x} {y} {w} {h}")
        with open(filename, "w") as f:
            f.write("\n".join(lines))
    print(f"  ✅ {count} dummy label files created in labels/{split}/")


def create_placeholder_readme():
    """Drop a README in the images folders so users know what to add."""
    msg = (
        "Place your images here.\n"
        "Supported formats: .jpg, .jpeg, .png, .bmp, .tiff\n"
        "Each image must have a matching .txt annotation in the labels/ folder.\n"
        "The .txt file must have the SAME base name as the image.\n\n"
        "Example:\n"
        "  images/train/dog_001.jpg  ←→  labels/train/dog_001.txt\n"
    )
    for split in SPLITS:
        path = os.path.join(BASE, "images", split, "PUT_IMAGES_HERE.txt")
        with open(path, "w") as f:
            f.write(msg)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔧 Creating dummy YOLO dataset structure...\n")
    create_folders()
    print("📝 Creating dummy annotation files...")
    create_dummy_labels("train", count=80)
    create_dummy_labels("val",   count=20)
    create_dummy_labels("test",  count=10)
    create_placeholder_readme()

    print("\n🎉 Done! Next steps:")
    print("  1. Copy your images into dataset/images/train| val | test/")
    print("  2. Copy your YOLO .txt labels into dataset/labels/train | val | test/")
    print("  3. Edit dataset.yaml to match your class names")
    print("  4. Run: python train.py\n")

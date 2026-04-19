"""
train.py
========
Fine-tune a YOLOv8 model on a custom YOLO-format dataset.

This script:
  - Loads the pretrained YOLOv8n (nano) model as a starting point.
  - Reads dataset paths and class names from dataset.yaml.
  - Trains for a configurable number of epochs.
  - Saves the best and last checkpoint to runs/train/expN/weights/.
  - Exports the best model to ONNX for deployment (optional).

Usage:
    python train.py                          # uses defaults below
    python train.py --epochs 50 --batch 8   # custom settings

Requirements:
    pip install -r requirements.txt
"""

import argparse
import os
import sys
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics not found. Run:  pip install ultralytics")
    sys.exit(1)

# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on a custom dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset.yaml",
        help="Path to dataset YAML config (default: dataset.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained weights to start from (default: yolov8n.pt). "
             "Options: yolov8n.pt | yolov8s.pt | yolov8m.pt | yolov8l.pt | yolov8x.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30). "
             "More epochs → better accuracy but longer training.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640). "
             "Use 416 or 320 for faster training on low-end machines.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8). "
             "Reduce to 4 or 2 if you run out of memory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker threads (default: 2). "
             "Set to 0 on Windows if you get multiprocessing errors.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on: '' = auto, 'cpu', '0' = GPU 0, '0,1' = multi-GPU.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Root directory for saving results (default: runs/train).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name — results go to <project>/<name> (default: exp).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export best.pt to ONNX format after training.",
    )
    return parser.parse_args()


# ── Validation helpers ────────────────────────────────────────────────────────
def validate_dataset(yaml_path: str):
    """Ensure the dataset YAML exists and the referenced folders are present."""
    if not os.path.isfile(yaml_path):
        print(f"❌ Dataset YAML not found: {yaml_path}")
        print("   Run 'python create_dummy_dataset.py' to create a sample structure,")
        print("   then edit dataset.yaml to point to your actual images.")
        sys.exit(1)

    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    root = cfg.get("path", ".")
    for split_key in ("train", "val"):
        rel_path = cfg.get(split_key, "")
        full_path = Path(root) / rel_path
        if not full_path.exists():
            print(f"⚠️  Warning: {split_key} folder not found at '{full_path}'.")
            print("   Make sure to add images before training or the run will fail.")

    print(f"✅ Dataset config loaded: {yaml_path}")
    print(f"   Classes ({cfg.get('nc', '?')}): {cfg.get('names', {})}")


def print_training_summary(args):
    """Print a human-readable summary of training settings."""
    print("\n" + "═" * 55)
    print("  YOLOv8 Training Configuration")
    print("═" * 55)
    print(f"  Model       : {args.model}")
    print(f"  Dataset     : {args.data}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Image size  : {args.imgsz}×{args.imgsz}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Workers     : {args.workers}")
    print(f"  Device      : {'auto' if not args.device else args.device}")
    print(f"  Output dir  : {args.project}/{args.name}")
    print("═" * 55 + "\n")


# ── Main training routine ─────────────────────────────────────────────────────
def train(args):
    # 1. Validate dataset config
    validate_dataset(args.data)

    # 2. Load the pretrained YOLOv8 model
    #    YOLOv8 will auto-download weights the first time if not cached.
    print(f"📦 Loading model: {args.model}")
    model = YOLO(args.model)

    # 3. Print settings overview
    print_training_summary(args)

    # 4. Start training
    print("🚀 Starting training...\n")
    results = model.train(
        data=args.data,           # path to dataset.yaml
        epochs=args.epochs,       # total training epochs
        imgsz=args.imgsz,         # resize all images to this size
        batch=args.batch,         # images per gradient-update step
        workers=args.workers,     # parallel data-loading threads
        device=args.device,       # CPU / GPU selection
        project=args.project,     # where to save results
        name=args.name,           # sub-folder name
        exist_ok=True,            # overwrite instead of creating exp2, exp3 …
        pretrained=True,          # use ImageNet-pretrained backbone weights
        optimizer="AdamW",        # AdamW is stable for fine-tuning
        lr0=0.001,                # initial learning rate
        lrf=0.01,                 # final LR = lr0 * lrf
        momentum=0.937,           # SGD momentum (unused for AdamW)
        weight_decay=0.0005,      # L2 regularization
        warmup_epochs=3,          # gradually ramp LR for first N epochs
        warmup_momentum=0.8,
        box=7.5,                  # bounding-box loss weight
        cls=0.5,                  # classification loss weight
        dfl=1.5,                  # distribution focal loss weight
        val=True,                 # run validation after every epoch
        save=True,                # save checkpoints
        save_period=-1,           # -1 = only save best and last
        plots=True,               # generate loss/metric plots
        verbose=True,             # print per-epoch metrics
        # ── Data augmentation (reduce if overfitting on small datasets) ──
        hsv_h=0.015,              # hue jitter
        hsv_s=0.7,                # saturation jitter
        hsv_v=0.4,                # value/brightness jitter
        degrees=0.0,              # rotation (degrees)
        translate=0.1,            # translation fraction
        scale=0.5,                # scale jitter
        fliplr=0.5,               # horizontal flip probability
        mosaic=1.0,               # mosaic augmentation probability
        mixup=0.0,                # mixup probability (disable for small datasets)
    )

    # 5. Report the location of the best weights
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "═" * 55)
    print("  ✅ Training complete!")
    print(f"  Best weights : {best_weights}")
    print("═" * 55)

    # 6. Validate on the test/val set using the best checkpoint
    print("\n🔍 Running final validation with best.pt …")
    try:
        model_best = YOLO(str(best_weights))
        val_results = model_best.val(data=args.data, imgsz=args.imgsz, verbose=True)
        print(f"\n  mAP50      : {val_results.box.map50:.4f}")
        print(f"  mAP50-95   : {val_results.box.map:.4f}")
    except Exception as e:
        print(f"⚠️  Validation skipped ({e}). Re-run manually if needed.")

    # 7. Optional ONNX export for production deployment
    if args.export:
        print("\n📤 Exporting best.pt → ONNX …")
        try:
            model_best = YOLO(str(best_weights))
            model_best.export(format="onnx", imgsz=args.imgsz, simplify=True)
            print("  ✅ ONNX model saved alongside best.pt")
        except Exception as e:
            print(f"⚠️  Export failed: {e}")

    print("\n👉 To run inference, use:")
    print(f"   python detect.py --source path/to/image.jpg --weights {best_weights}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)

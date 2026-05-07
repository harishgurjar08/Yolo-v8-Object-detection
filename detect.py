"""
detect.py
=========
Run YOLOv8 object detection on images, video files, or a webcam feed.

Modes
-----
  image   – Detect objects in a single image or folder of images.
  video   – Process a video file frame-by-frame and save the result.
  webcam  – Real-time detection from a connected camera.

Usage examples
--------------
  python detect.py --source image.jpg
  python detect.py --source images/          # all images in a folder
  python detect.py --source video.mp4 --mode video
  python detect.py --mode webcam
  python detect.py --source image.jpg --weights runs/train/exp/weights/best.pt
  python detect.py --source image.jpg --conf 0.4 --iou 0.5 --no-save
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics not found. Run:  pip install ultralytics")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Object Detection — image | video | webcam"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to image file, image folder, or video file. "
             "Not needed for webcam mode.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Model weights file (default: yolov8n.pt — downloads automatically).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        choices=["image", "video", "webcam"],
        help="Detection mode (default: image).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold — only show detections above this score "
             "(default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for Non-Maximum Suppression (default: 0.45). "
             "Lower value → fewer duplicate boxes.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device: '' = auto, 'cpu', '0' = GPU 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save annotated results (default: outputs/).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output files — only display.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a display window (useful for headless servers).",
    )
    parser.add_argument(
        "--webcam-id",
        type=int,
        default=0,
        help="Camera index for webcam mode (default: 0).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum number of detections per image (default: 300).",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

# A fixed palette of visually distinct BGR colors for up to 20 classes.
_PALETTE = [
    (0, 114, 189),   (217, 83, 25),   (237, 177, 32),  (126, 47, 142),
    (119, 172, 48),  (77, 190, 238),  (162, 20, 47),   (76, 76, 76),
    (153, 153, 153), (255, 0, 0),     (0, 255, 0),     (0, 0, 255),
    (255, 255, 0),   (0, 255, 255),   (255, 0, 255),   (255, 128, 0),
    (128, 255, 0),   (0, 128, 255),   (128, 0, 255),   (255, 0, 128),
]

def get_color(class_id: int) -> tuple:
    """Return a consistent BGR color for a given class id."""
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(frame: np.ndarray, results, class_names: list) -> np.ndarray:
    """
    Draw bounding boxes, class labels, and confidence scores on a frame.

    Parameters
    ----------
    frame       : BGR image as a NumPy array.
    results     : Ultralytics Results object from model.predict().
    class_names : List of class name strings.

    Returns
    -------
    Annotated frame as a NumPy array.
    """
    annotated = frame.copy()

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            # ── Extract values ────────────────────────────────────────────────
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf            = float(box.conf[0])
            cls_id          = int(box.cls[0])
            label           = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
            color           = get_color(cls_id)

            # ── Bounding box ──────────────────────────────────────────────────
            thickness = 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # ── Label background ──────────────────────────────────────────────
            text      = f"{label} {conf:.2f}"
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
            label_y   = max(y1 - 4, th + 4)          # keep label inside frame
            cv2.rectangle(
                annotated,
                (x1, label_y - th - baseline),
                (x1 + tw, label_y + baseline),
                color,
                cv2.FILLED,
            )

            # ── Label text ────────────────────────────────────────────────────
            # Choose black or white text for readability based on background brightness
            brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            cv2.putText(
                annotated, text, (x1, label_y),
                font, font_scale, text_color, 1, cv2.LINE_AA,
            )

    return annotated


def overlay_stats(frame: np.ndarray, num_detections: int, fps: float = 0.0) -> np.ndarray:
    """
    Overlay a small info bar at the top-left corner of the frame.
    Shows the number of detected objects and (for video/webcam) the FPS.
    """
    info = f"Objects: {num_detections}"
    if fps > 0:
        info += f"  |  FPS: {fps:.1f}"
    cv2.putText(
        frame, info, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA,   # shadow
    )
    cv2.putText(
        frame, info, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(weights: str, device: str) -> YOLO:
    """Load YOLOv8 model from weights path. Downloads from HuggingFace if needed."""
    print(f"📦 Loading weights: {weights}")
    try:
        model = YOLO(weights)
        # Warm-up: run one dummy inference so the first real call isn't slow
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, device=device, verbose=False)
        print("  ✅ Model ready.\n")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)


# ── Image detection ────────────────────────────────────────────────────────────

def detect_image(args) -> None:
    """
    Detect objects in a single image or all images inside a directory.
    Saves annotated copies to args.output_dir.
    """
    source = Path(args.source)
    if not source.exists():
        print(f"❌ Source not found: {source}")
        sys.exit(1)

    # Gather image file paths
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if source.is_dir():
        paths = [p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if not paths:
            print(f"❌ No images found in '{source}'")
            sys.exit(1)
    else:
        paths = [source]

    model       = load_model(args.weights, args.device)
    class_names = model.names  # dict {0: 'cat', 1: 'dog', …}
    output_dir  = Path(args.output_dir) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_dets = 0
    for img_path in paths:
        # ── Read image ────────────────────────────────────────────────────────
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ⚠️  Could not read '{img_path}', skipping.")
            continue

        # ── Inference ─────────────────────────────────────────────────────────
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )

        num_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
        total_dets += num_dets

        # ── Draw & print ──────────────────────────────────────────────────────
        annotated = draw_detections(frame, results, list(class_names.values()))
        annotated = overlay_stats(annotated, num_dets)

        print(f"  🖼  {img_path.name}  →  {num_dets} detection(s)")
        if num_dets > 0:
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    name   = class_names.get(cls_id, f"cls{cls_id}")
                    print(f"       • {name}: {conf:.2%}")

        # ── Save output ───────────────────────────────────────────────────────
        if not args.no_save:
            out_path = output_dir / f"detected_{img_path.name}"
            cv2.imwrite(str(out_path), annotated)
            print(f"       💾 Saved → {out_path}")

        # ── Display ───────────────────────────────────────────────────────────
        if not args.no_display:
            cv2.imshow(f"YOLOv8 — {img_path.name}", annotated)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("q"):          # press Q to quit early
                break

    print(f"\n✅ Done. Total detections across all images: {total_dets}")


# ── Video detection ────────────────────────────────────────────────────────────

def detect_video(args) -> None:
    """
    Process a video file frame-by-frame and save an annotated copy.
    Displays a live preview window unless --no-display is set.
    """
    source = Path(args.source)
    if not source.exists():
        print(f"❌ Video file not found: {source}")
        sys.exit(1)

    model       = load_model(args.weights, args.device)
    class_names = model.names
    output_dir  = Path(args.output_dir) / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(source).isdigit():
       cap = cv2.VideoCapture(int(source))  # webcam
    else:
            cap = cv2.VideoCapture(str(source))  # video file
    if not cap.isOpened():
        print(f"❌ Cannot open video: {source}")
        sys.exit(1)



    # ── Video writer setup ────────────────────────────────────────────────────
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = output_dir / f"detected_{source.name}"

    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))

    print(f"🎬 Processing '{source.name}'  ({total_frames} frames, {fps_in:.1f} fps)")
    print("   Press 'Q' in the preview window to stop early.\n")

    frame_idx  = 0
    fps_timer  = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── Inference ─────────────────────────────────────────────────────────
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )
        num_dets = sum(len(r.boxes) for r in results if r.boxes is not None)

        # ── FPS calculation (rolling per-frame) ───────────────────────────────
        now         = time.time()
        fps_display = 1.0 / max(now - fps_timer, 1e-6)
        fps_timer   = now

        # ── Annotate ──────────────────────────────────────────────────────────
        annotated = draw_detections(frame, results, list(class_names.values()))
        annotated = overlay_stats(annotated, num_dets, fps_display)

        # ── Progress bar (terminal) ───────────────────────────────────────────
        pct = frame_idx / max(total_frames, 1) * 100
        print(
            f"\r  Frame {frame_idx}/{total_frames}  [{pct:5.1f}%]  "
            f"FPS: {fps_display:5.1f}  Dets: {num_dets}   ",
            end="",
            flush=True,
        )

        # ── Write frame ───────────────────────────────────────────────────────
        if writer:
            writer.write(annotated)

        # ── Display ───────────────────────────────────────────────────────────
        if not args.no_display:
            cv2.imshow("YOLOv8 — Video Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  ⏹  Stopped by user.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n\n✅ Video processing complete.")
    if not args.no_save:
        print(f"   💾 Saved → {out_path}")


# ── Webcam detection ───────────────────────────────────────────────────────────

def detect_webcam(args) -> None:
    """
    Run real-time object detection from a webcam.
    Press 'Q' to quit, 'S' to save the current frame.
    """
    model       = load_model(args.weights, args.device)
    class_names = model.names
    output_dir  = Path(args.output_dir) / "webcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try different backends for cross-platform compatibility
    cap = None
    backends = [
        (args.webcam_id, cv2.CAP_DSHOW),    # Windows DirectShow (fastest)
        (args.webcam_id, cv2.CAP_V4L2),     # Linux
        (args.webcam_id, cv2.CAP_AVFOUNDATION),  # macOS
        (args.webcam_id, cv2.CAP_ANY),      # Auto-detect fallback
    ]

    for cam_id, backend in backends:
        cap = cv2.VideoCapture(cam_id, backend)
        if cap.isOpened():
            print(f"✅ Camera opened with backend: {backend}")
            break
        cap.release()

    if cap is None or not cap.isOpened():
        print(f"❌ Cannot open camera (id={args.webcam_id}).")
        print("   Check that a webcam is connected and not in use by another app.")
        sys.exit(1)

    # Set smaller buffer for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Try to set resolution, but fall back if not supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Camera resolution: {actual_width}×{actual_height}")

    print("📷 Webcam detection started.")
    print("   Q = quit   |   S = save current frame\n")

    fps_timer   = time.time()
    fps_display = 0.0
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to grab frame. Check your camera.")
            break

        frame_count += 1

        # ── Inference ─────────────────────────────────────────────────────────
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )
        num_dets = sum(len(r.boxes) for r in results if r.boxes is not None)

        # ── FPS ───────────────────────────────────────────────────────────────
        now         = time.time()
        fps_display = 1.0 / max(now - fps_timer, 1e-6)
        fps_timer   = now

        # ── Annotate ──────────────────────────────────────────────────────────
        annotated = draw_detections(frame, results, list(class_names.values()))
        annotated = overlay_stats(annotated, num_dets, fps_display)

        cv2.imshow("YOLOv8 — Webcam  [Q=quit | S=save]", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("⏹  Webcam detection stopped.")
            break
        elif key == ord("s"):
            # Save current frame on 'S' key press
            ts       = time.strftime("%Y%m%d_%H%M%S")
            out_path = output_dir / f"webcam_{ts}.jpg"
            cv2.imwrite(str(out_path), annotated)
            saved_count += 1
            print(f"  💾 Frame saved → {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Session ended. Frames processed: {frame_count}. Snapshots saved: {saved_count}.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Route to the appropriate detection function
    if args.mode == "image":
        if not args.source:
            print("❌ Please provide --source for image mode.")
            sys.exit(1)
        detect_image(args)

    elif args.mode == "video":
        if not args.source:
            print("❌ Please provide --source for video mode.")
            sys.exit(1)
        detect_video(args)

    elif args.mode == "webcam":
        detect_webcam(args)

    else:
        print(f"❌ Unknown mode '{args.mode}'. Choose: image | video | webcam")
        sys.exit(1)


if __name__ == "__main__":
    main()

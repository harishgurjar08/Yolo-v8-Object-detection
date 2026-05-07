"""
app.py
======
Streamlit web dashboard for YOLOv8 Object Detection.

Features
--------
  • Upload an image or video and run detection instantly.
  • Live webcam feed with real-time bounding boxes.
  • Adjustable confidence / IoU sliders.
  • Download annotated outputs.
  • Detection results table (class, confidence, bounding box).
  • Model selector and training launcher.

Run
---
    streamlit run app.py
"""
# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import time
import tempfile
import threading
from pathlib import Path
from io import BytesIO
import queue

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
import streamlit as st
try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ `ultralytics` not installed. Run:  `pip install ultralytics`")
    st.stop()

# ── Local utility (re-use drawing code from detect.py) ───────────────────────
# We import selectively so app.py works even if detect.py is missing.
try:
    from detect import draw_detections, overlay_stats, get_color
except ImportError:
    # Inline minimal fallback if detect.py is absent
    def get_color(cls_id):
        palette = [(0,114,189),(217,83,25),(237,177,32),(126,47,142),(119,172,48)]
        return palette[cls_id % len(palette)]

    def draw_detections(frame, results, class_names):
        ann = frame.copy()
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0]); conf = float(box.conf[0])
                label = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
                color = get_color(cls_id)
                cv2.rectangle(ann,(x1,y1),(x2,y2),color,2)
                cv2.putText(ann,f"{label} {conf:.2f}",(x1,max(y1-6,12)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        return ann

    def overlay_stats(frame, num_dets, fps=0.0):
        txt = f"Objects: {num_dets}" + (f"  |  FPS: {fps:.1f}" if fps > 0 else "")
        cv2.putText(frame, txt, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame, txt, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1,cv2.LINE_AA)
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="YOLOv8 Object Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a Clean, Professional SaaS "Soft Dark Mode"
st.markdown("""
<style>
/* ── Hide Default Streamlit Elements ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/*header {visibility: hidden;}

/* ── Base Theme ── */
.stApp {
    background-color: #0f111a;
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #161925;
    border-right: 1px solid #232738;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0.2rem !important;
    padding-bottom: 1.5rem !important;
}

.stSelectbox label, .stSlider label, .stTextInput label, .stNumberInput label {
    font-weight: 500;
    color: #9ba1b0;
}

/* ── Metric Cards ── */
div[data-testid="stMetric"] {
    background-color: #1c1f30;
    border: 1px solid #2b3046;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
}

/* ── Header Area ── */
.hero-header {
    background-color: #1c1f30;
    border: 1px solid #2b3046;
    padding: 32px 40px;
    border-radius: 16px;
    margin-bottom: 32px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #4f46e5, #ec4899, #f59e0b);
}
.hero-header h1 { 
    margin: 0; 
    font-size: 2.2rem; 
    font-weight: 800; 
    letter-spacing: -0.025em;
    color: #f8fafc;
}
.hero-header p { 
    margin: 8px 0 0; 
    color: #94a3b8; 
    font-size: 1.05rem; 
    font-weight: 400;
}

/* ── Premium Primary Buttons ── */
.stButton>button, div.stDownloadButton>button {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
    border: 1px solid #4f46e5 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
    transition: all 0.2s ease !important;
}
.stButton>button:hover, div.stDownloadButton>button:hover {
    background-color: #4338ca !important;
    border-color: #4338ca !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 8px -1px rgba(79, 70, 229, 0.3) !important;
}

/* ── Secondary Secondary Button styling override ── */
button[kind="secondary"] {
    background-color: #1e293b !important;
    border-color: #334155 !important;
    color: #cbd5e1 !important;
    box-shadow: none !important;
}
button[kind="secondary"]:hover {
    background-color: #334155 !important;
    border-color: #475569 !important;
    color: #ffffff !important;
}

/* ── Section Dividers ── */
hr { 
    border-color: #2b3046 !important; 
    margin-top: 1rem !important;
    margin-bottom: 1rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #161925;
    padding: 8px;
    border-radius: 12px;
    border-bottom: none;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    color: #9ba1b0;
    font-weight: 600;
    opacity: 0.8;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #2b3046 !important;
    color: #f8fafc !important;
    opacity: 1;
}

/* ── Tables & DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid #2b3046;
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(weights: str) -> YOLO:
    """
    Load and cache the YOLO model so it is not reloaded on every interaction.
    st.cache_resource keeps the object alive across reruns.
    """
    try:
        model = YOLO(weights)
        # Warm-up pass
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        model.predict(dummy, verbose=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model '{weights}': {e}")
        st.stop()


def frame_to_bytes(frame: np.ndarray, ext: str = ".jpg") -> bytes:
    """Convert an OpenCV BGR frame to bytes for Streamlit download."""
    success, buf = cv2.imencode(ext, frame)
    return buf.tobytes() if success else b""


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR to RGB for Streamlit/matplotlib display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_results_df(results, class_names) -> pd.DataFrame:
    """Build a Pandas DataFrame from detection results for the UI table."""
    rows = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name   = class_names.get(cls_id, f"cls{cls_id}")
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = x2 - x1; h = y2 - y1
            rows.append({
                "Class": name,
                "Confidence": f"{conf:.2%}",
                "x1": x1, "y1": y1,
                "Width": w, "Height": h,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def sidebar() -> dict:
    """Render the sidebar and return a config dict."""
    with st.sidebar:
        st.markdown("###  Quick Start Guide")
        st.markdown(
            "<div style='font-size:0.85rem; color:#94a3b8; line-height:1.5; margin-bottom:-10px;'>"
            "1. <b>Pick a source:</b> Image, Video, or Webcam<br>"
            "2. <b>Select Model:</b> Nano (fast) or Large (accurate)<br>"
            "3. <b>Adjust settings</b> below."
            "</div>", unsafe_allow_html=True
        )
        st.divider()

        # ── Model selection ───────────────────────────────────────────────────
        st.subheader("Sidebar Control panel")
        model_options = {
            "YOLOv8n (Nano — fastest)":   "yolov8n.pt",
            "YOLOv8s (Small)":            "yolov8s.pt",
            "YOLOv8m (Medium)":           "yolov8m.pt",
            "YOLOv8l (Large)":            "yolov8l.pt",
            "YOLOv8x (Extra Large)":      "yolov8x.pt",
            "Custom (best.pt)":           "runs/train/exp/weights/best.pt",
        }
        model_label = st.selectbox("Select model", list(model_options.keys()))
        weights     = model_options[model_label]

        # Allow custom path override
        custom_path = st.text_input("Or enter custom weights path", placeholder="path/to/best.pt")
        if custom_path.strip():
            weights = custom_path.strip()

        st.divider()

        # ── Inference settings ────────────────────────────────────────────────
        st.subheader("Detection Settings")
        
        with st.expander("What do these settings mean?"):
            st.markdown(
                "**Confidence Threshold:** Minimum probability required for a detected object to be considered valid (filters out low-confidence predictions).\n\n"
                "**IoU (Intersection over Union):** Measures overlap between predicted box and actual box to decide if detection is correct and to remove duplicates.\n\n"
                "**Inference Image Size:** The resolution (e.g., 640×640) at which the model processes the image during prediction, affecting speed and accuracy."
            )
        conf_thresh = st.slider(
            "Confidence threshold",
            min_value=0.05, max_value=1.0, value=0.25, step=0.05,
            help="Only show detections above this confidence score.",
        )
        iou_thresh = st.slider(
            "IoU threshold (NMS)",
            min_value=0.1, max_value=1.0, value=0.45, step=0.05,
            help="Controls how aggressively overlapping boxes are suppressed.",
        )
        img_size = st.select_slider(
            "Inference image size",
            options=[320, 416, 512, 640, 768, 1024],
            value=640,
            help="Smaller = faster; larger = more accurate.",
        )
        max_det = st.number_input("Max detections per frame", min_value=1, max_value=1000, value=100)

        st.divider()


    return {
        "weights":    weights,
        "conf":       conf_thresh,
        "iou":        iou_thresh,
        "imgsz":      img_size,
        "max_det":    int(max_det),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: IMAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def tab_image(cfg: dict):
    st.subheader("🖼️ Image Detection")
    st.write("Upload an image and the model will detect all objects in it.")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported: JPG, PNG, BMP, WEBP",
    )

    if uploaded is None:
        st.info("⬆️ Upload an image to get started.")
        return

    # ── Read uploaded image ───────────────────────────────────────────────────
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("❌ Could not read the image. Please try another file.")
        return

    col_orig, col_det = st.columns(2, gap="medium")

    with col_orig:
        st.markdown("**Original Image**")
        st.image(frame, channels="BGR", use_container_width=True)

    # ── Run detection ─────────────────────────────────────────────────────────
    model = load_model_cached(cfg["weights"])

    with st.spinner("🔍 Running detection…"):
        t0 = time.perf_counter()
        results = model.predict(
            source=frame,
            conf=cfg["conf"],
            iou=cfg["iou"],
            imgsz=cfg["imgsz"],
            max_det=cfg["max_det"],
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

    class_names = model.names
    num_dets    = sum(len(r.boxes) for r in results if r.boxes is not None)
    annotated   = draw_detections(frame, results, list(class_names.values()))
    annotated   = overlay_stats(annotated, num_dets)

    with col_det:
        st.markdown("**Detected Objects**")
        st.image(annotated, channels="BGR", use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Objects found",     num_dets)
    m2.metric("Inference time",    f"{inference_ms:.1f} ms")
    m3.metric("Image size",        f"{frame.shape[1]}×{frame.shape[0]}")
    m4.metric("Conf threshold",    f"{cfg['conf']:.0%}")

    # ── Results table ─────────────────────────────────────────────────────────
    if num_dets > 0:
        st.divider()
        st.markdown("####  Detection Details")
        df = build_results_df(results, class_names)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Class distribution bar ────────────────────────────────────────────
        if len(df) > 1:
            st.markdown("#### 📊 Class Distribution")
            counts = df["Class"].value_counts().reset_index()
            counts.columns = ["Class", "Count"]
            st.bar_chart(counts.set_index("Class"))
    else:
        st.warning("⚠️ No objects detected above the confidence threshold. Try lowering it in the sidebar.")

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    img_bytes = frame_to_bytes(annotated)
    out_name  = f"detected_{uploaded.name}"
    st.download_button(
        label="⬇️ Download Annotated Image",
        data=img_bytes,
        file_name=out_name,
        mime="image/jpeg",
    )

    # Also save to disk
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "images" / out_name), annotated)
    st.caption(f"file {out_name} not stored on our server")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: VIDEO DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def tab_video(cfg: dict):
    st.subheader("🎬 Video Detection")
    st.write("Upload a video file and get an annotated version back.")

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        help="Supported: MP4, AVI, MOV, MKV, WMV",
    )

    if uploaded is None:
        st.info("⬆️ Upload a video to get started.")
        return

    # ── Save to temp file (OpenCV needs a path) ───────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("❌ Could not open video. Please try another file.")
        os.unlink(tmp_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s   = total_frames / fps_in

    st.info(f"📹 **{uploaded.name}** — {width}×{height}px | {fps_in:.1f} fps | "
            f"{total_frames} frames | {duration_s:.1f}s")

    # ── Sample the first frame for preview ───────────────────────────────────
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)    # rewind

    if ret and first_frame is not None:
        st.image(first_frame, channels="BGR", caption="First frame preview",
                 use_container_width=True)

    if not st.button("▶️ Process Video", type="primary"):
        cap.release()
        return

    # ── Processing ────────────────────────────────────────────────────────────
    model       = load_model_cached(cfg["weights"])
    class_names = model.names
    out_dir     = OUTPUT_DIR / "videos"
    out_dir.mkdir(exist_ok=True)
    out_path    = out_dir / f"detected_{uploaded.name.rsplit('.', 1)[0]}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))

    progress_bar   = st.progress(0, text="Processing frames…")
    preview_slot   = st.empty()
    stats_slot     = st.empty()

    frame_idx    = 0
    total_dets   = 0
    t_start      = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model.predict(
            source=frame,
            conf=cfg["conf"],
            iou=cfg["iou"],
            imgsz=cfg["imgsz"],
            max_det=cfg["max_det"],
            verbose=False,
        )
        num_dets  = sum(len(r.boxes) for r in results if r.boxes is not None)
        total_dets += num_dets

        annotated = draw_detections(frame, results, list(class_names.values()))

        elapsed  = time.perf_counter() - t_start
        fps_live = frame_idx / max(elapsed, 0.001)
        annotated = overlay_stats(annotated, num_dets, fps_live)

        writer.write(annotated)

        # Update UI every 10 frames to avoid slowdown
        if frame_idx % 10 == 0 or frame_idx == total_frames:
            pct = frame_idx / max(total_frames, 1)
            progress_bar.progress(pct, text=f"Processing…  {frame_idx}/{total_frames} frames  ({pct*100:.0f}%)")
            preview_slot.image(annotated, channels="BGR", caption=f"Frame {frame_idx}", use_container_width=True)
            eta = (total_frames - frame_idx) / max(fps_live, 0.01)
            stats_slot.caption(f"⚡ {fps_live:.1f} fps  |  ETA: {eta:.0f}s  |  Detections so far: {total_dets}")

    cap.release()
    writer.release()
    os.unlink(tmp_path)

    progress_bar.progress(1.0, text="✅ Done!")
    elapsed_total = time.perf_counter() - t_start

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total detections",  total_dets)
    m2.metric("Frames processed",  frame_idx)
    m3.metric("Total time",        f"{elapsed_total:.1f}s")

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    if out_path.exists():
        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Annotated Video",
                data=f.read(),
                file_name=out_path.name,
                mime="video/mp4",
            )
        st.caption(f"💾 Also saved to `{out_path}`")
    else:
        st.warning("⚠️ Output file not found after processing.")


@st.cache_resource
def get_camera_state():
    return {
        "recording": False,
        "snapshot_req": False,
        "latest_snapshot": None,
        "video_writer": None,
        "out_path": None
    }

def tab_webcam(cfg: dict):
    st.subheader("📷 Real-Time Webcam Detection")
    st.write(
        "Capture frames from your **browser's webcam** for real-time object detection. "
        "Detection runs continuously on the live video stream."
    )

    # Streamlit Cloud compatibility: if PyAV / streamlit-webrtc isn't importable,
    # fall back to Streamlit's built-in camera input (snapshot-based).
    if not HAS_WEBRTC:
        st.warning(
            "WebRTC webcam mode isn't available in this deployment (missing `streamlit-webrtc`/`av`). "
            "Using snapshot mode instead."
        )
        cam_img = st.camera_input("Take a photo")
        if cam_img is None:
            return

        file_bytes = np.frombuffer(cam_img.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("❌ Could not read the camera image.")
            return

        model = load_model_cached(cfg["weights"])
        class_names = model.names
        with st.spinner("🔍 Running detection…"):
            results = model.predict(
                source=frame,
                conf=cfg["conf"],
                iou=cfg["iou"],
                imgsz=cfg["imgsz"],
                max_det=cfg["max_det"],
                verbose=False,
            )

        num_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
        annotated = draw_detections(frame, results, list(class_names.values()))
        annotated = overlay_stats(annotated, num_dets)
        st.image(annotated, channels="BGR", use_container_width=True)

        st.download_button(
            label="⬇️ Download Annotated Photo",
            data=frame_to_bytes(annotated),
            file_name=f"webcam_photo_{int(time.time())}.jpg",
            mime="image/jpeg",
        )
        return

    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False

    if not st.session_state.webcam_active:
        st.info("💡 Click the button below to turn on your webcam and start the live feed.")
        if st.button("🚀 Open Webcam", use_container_width=False, type="primary"):
            st.session_state.webcam_active = True
            st.rerun()
        return

    # If active, show the stop button
    if st.button("🛑 Close Webcam", use_container_width=False, type="secondary"):
        st.session_state.webcam_active = False
        st.rerun()

    cam_state = get_camera_state()
    model = load_model_cached(cfg["weights"])
    class_names = model.names

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run inference
        results = model.predict(
            source=img,
            conf=cfg["conf"],
            iou=cfg["iou"],
            imgsz=cfg["imgsz"],
            max_det=cfg["max_det"],
            verbose=False,
        )
        
        num_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
        annotated = draw_detections(img, results, list(class_names.values()))
        annotated = overlay_stats(annotated, num_dets)

        # Handle Snapshot Request
        if cam_state["snapshot_req"]:
            cam_state["latest_snapshot"] = annotated.copy()
            cam_state["snapshot_req"] = False
            
        # Handle Video Recording
        if cam_state["recording"]:
            if cam_state["video_writer"] is None:
                h, w = annotated.shape[:2]
                out_name = f"webcam_rec_{int(time.time())}.mp4"
                out_path = str(OUTPUT_DIR / out_name)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # Using 20.0 fps as a reasonable default for WebRTC streams
                cam_state["video_writer"] = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                cam_state["out_path"] = out_path
            cam_state["video_writer"].write(annotated)
        else:
            if cam_state["video_writer"] is not None:
                cam_state["video_writer"].release()
                cam_state["video_writer"] = None

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    # Setup the two-column layout for the player and the controls
    c1, c2 = st.columns([3, 1], gap="large")
    
    with c1:
        webrtc_ctx = webrtc_streamer(
            key="webcam-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            desired_playing_state=st.session_state.webcam_active,
        )
        
    with c2:
        st.markdown("### 🎛️ Controls")
        if webrtc_ctx.state.playing:
            st.success("🟢 Live Stream Active")
            
            # --- Snapshot Button ---
            if st.button("📸 Capture Snapshot", use_container_width=True):
                cam_state["snapshot_req"] = True
                
            st.divider()
            
            # --- Recording Toggle ---
            if not cam_state["recording"]:
                if st.button("🎥 Start Recording", use_container_width=True):
                    cam_state["recording"] = True
                    st.rerun()
            else:
                if st.button("🛑 Stop Recording", type="primary", use_container_width=True):
                    cam_state["recording"] = False
                    if cam_state["video_writer"] is not None:
                        cam_state["video_writer"].release()
                        cam_state["video_writer"] = None
                    st.rerun()

            if cam_state["recording"]:
                st.error("🔴 Recording in progress... Please stand by.")
        
        else:
            st.warning("⏳ Waiting for stream to start...")

    st.divider()
    
    # ── Display the latest snapshot ──
    if cam_state["latest_snapshot"] is not None:
        st.subheader("🖼️ Latest Snapshot")
        col_snap, col_snap_btn = st.columns([3, 1])
        with col_snap:
            st.image(cam_state["latest_snapshot"], channels="BGR", use_container_width=True)
            
        with col_snap_btn:
            snap_bytes = frame_to_bytes(cam_state["latest_snapshot"])
            st.download_button(
                label="⬇️ Download Snapshot",
                data=snap_bytes,
                file_name=f"webcam_snap_{int(time.time())}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
            if st.button("🗑️ Clear Snapshot", use_container_width=True):
                cam_state["latest_snapshot"] = None
                st.rerun()

    # ── Provide link to download last recording ──
    if not cam_state["recording"] and cam_state["out_path"] is not None:
        if os.path.exists(cam_state["out_path"]):
            with open(cam_state["out_path"], "rb") as f:
                rec_bytes = f.read()
            st.success(f"✅ Video successfully saved!")
            st.download_button(
                label="⬇️ Download Last Recording",
                data=rec_bytes,
                file_name=os.path.basename(cam_state["out_path"]),
                mime="video/mp4"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: TRAINING LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════

def tab_train():
    st.subheader("🏋️ Train a Custom Model")
    st.write(
        "Configure and launch a training run directly from the UI. "
        "Logs will appear below. The model is saved to `runs/train/`."
    )

    with st.expander("📚 Dataset Setup Instructions", expanded=False):
        st.markdown("""
        **1. Run the dataset creator:**
        ```bash
        python create_dummy_dataset.py
        ```
        **2. Copy your images into:**
        ```
        dataset/images/train/   ← training images
        dataset/images/val/     ← validation images
        ```
        **3. Copy matching YOLO .txt labels into:**
        ```
        dataset/labels/train/
        dataset/labels/val/
        ```
        **4. Edit `dataset.yaml`** to set class names.

        **YOLO label format (one line per object):**
        ```
        <class_id> <x_center> <y_center> <width> <height>
        ```
        All values normalized 0–1 relative to image dimensions.
        """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        base_model = st.selectbox(
            "Base model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            help="Nano is fastest; Medium is most accurate."
        )
        epochs = st.slider("Epochs", 5, 300, 30,
                           help="More epochs → better accuracy but longer training.")
        batch  = st.select_slider("Batch size", [2, 4, 8, 16, 32], value=8,
                                   help="Reduce if you run out of memory.")
    with col2:
        data_yaml = st.text_input("Dataset YAML path", value="dataset.yaml")
        img_size  = st.select_slider("Image size", [320, 416, 512, 640], value=640)
        workers   = st.number_input("DataLoader workers", 0, 8, 2,
                                     help="Set to 0 on Windows if errors occur.")

    st.divider()

    if st.button("🚀 Launch Training", type="primary"):
        if not Path(data_yaml).exists():
            st.error(f"❌ Dataset YAML not found: `{data_yaml}`. "
                     "Run `python create_dummy_dataset.py` first.")
            return

        cmd = (
            f"python train.py "
            f"--data {data_yaml} "
            f"--model {base_model} "
            f"--epochs {epochs} "
            f"--batch {batch} "
            f"--imgsz {img_size} "
            f"--workers {workers}"
        )

        st.code(cmd, language="bash")
        st.info("📋 Copy the command above and run it in your terminal. "
                "Training output (metrics, plots, weights) will be saved to `runs/train/exp/`.")

        # Show training tips
        with st.expander("💡 Training Tips"):
            st.markdown("""
            - **Overfitting?** Reduce epochs, increase data augmentation, or add more images.
            - **Underfitting?** Increase epochs, use a larger model (yolov8m.pt).
            - **Out of memory?** Reduce batch size or image size.
            - **Speed up training:** Use GPU (set `--device 0`).
            - **Best weights:** Saved at `runs/train/exp/weights/best.pt` after training.
            """)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>     AI Surveillance System Real-time object detection and monitoring</h1>
        <p>        All uploads are processed in real time and are not stored on our servers</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar config ────────────────────────────────────────────────────────
    cfg = sidebar()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️  Image", "🎬  Video", "📷  Webcam", "🏋️  Train"
    ])

    with tab1:
        tab_image(cfg)

    with tab2:
        tab_video(cfg)

    with tab3:
        tab_webcam(cfg)

    with tab4:
        tab_train()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; color: #FFFFFF; padding-top: 10px; padding-bottom: 40px;">
        <p style="margin-bottom: 12px; font-size: 1.05rem;">Developed by <strong>Harish Singh</strong> — AI & Data Science Engineer</p>
        <div style="display: flex; justify-content: center; gap: 24px; font-weight: 500;">
            <a href="https://www.linkedin.com/in/harishgurjar11/" target="_blank" style="color: #FFFFFF; text-decoration: none; display: flex; align-items: center; gap: 6px; transition: color 0.2s;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="18" style="filter: grayscale(20%);"> LinkedIn
            </a>
            <a href="https://github.com/harishgurjar08/Yolo-v8-Object-detection" target="_blank" style="color: #FFFFFF; text-decoration: none; display: flex; align-items: center; gap: 6px; transition: color 0.2s;">
                <img src="https://cdn.iconscout.com/icon/free/png-256/free-github-logo-icon-svg-download-png-8630395.png?f=webp" width="18" style="filter: invert(100%);"> GitHub
            </a>
            <a href="https://harish-portfolio-eta.vercel.app/" class="portfolio-link" target="_blank" title="Replace '#' with your actual Portfolio URL" style="color: #FFFFFF; text-decoration: none; display: flex; align-items: center; gap: 6px; transition: color 0.2s;">
                <span style="font-size: 18px;">🌐</span> Portfolio
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

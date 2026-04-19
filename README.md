# 🎯 YOLOv8 Object Detection — Complete Project

A production-ready object detection pipeline built with **Ultralytics YOLOv8**, **OpenCV**, and **Streamlit**.  
Supports custom dataset training, image/video/webcam detection, and a clean interactive dashboard.

---

## 📁 Project Structure

```
yolov8_project/
│
├── app.py                    # Streamlit web dashboard (main UI)
├── train.py                  # Model training script
├── detect.py                 # CLI detection (image / video / webcam)
├── create_dummy_dataset.py   # Generates example dataset structure
├── dataset.yaml              # Dataset configuration (classes, paths)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── dataset/                  # Auto-created by create_dummy_dataset.py
│   ├── images/
│   │   ├── train/            # ← Put training images here
│   │   ├── val/              # ← Put validation images here
│   │   └── test/             # ← Put test images here (optional)
│   └── labels/
│       ├── train/            # YOLO .txt annotations for train images
│       ├── val/              # YOLO .txt annotations for val images
│       └── test/
│
├── runs/                     # Auto-created during training
│   └── train/exp/
│       ├── weights/
│       │   ├── best.pt       # ← Use this for inference
│       │   └── last.pt
│       └── results.png       # Training metrics plot
│
└── outputs/                  # Auto-created during detection
    ├── images/               # Annotated image outputs
    ├── videos/               # Annotated video outputs
    └── webcam/               # Saved webcam snapshots
```

---

## ⚡ Quick Start

### 1 — Clone & enter the project

```bash
git clone https://github.com/yourname/yolov8-project.git
cd yolov8-project
```

### 2 — Create a Python virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

### 4 — Create the dataset folder structure

```bash
python create_dummy_dataset.py
```

This creates `dataset/` with the correct folders and example label files.

### 5 — Add your images and labels

Copy your images into:
- `dataset/images/train/` → training images
- `dataset/images/val/` → validation images

Copy matching YOLO `.txt` annotation files into:
- `dataset/labels/train/`
- `dataset/labels/val/`

Each annotation file must have the **same base name** as its image:
```
dataset/images/train/dog_001.jpg  ←→  dataset/labels/train/dog_001.txt
```

### 6 — Edit `dataset.yaml`

Open `dataset.yaml` and update the class names to match your dataset:

```yaml
nc: 3
names:
  0: cat
  1: dog
  2: person
```

---

## 🏋️ Training

```bash
# Default settings (30 epochs, batch 8, image size 640)
python train.py

# Custom settings
python train.py --epochs 100 --batch 16 --imgsz 640 --model yolov8s.pt

# All options
python train.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--data` | `dataset.yaml` | Dataset config file |
| `--model` | `yolov8n.pt` | Base model (n/s/m/l/x) |
| `--epochs` | `30` | Training epochs |
| `--batch` | `8` | Batch size |
| `--imgsz` | `640` | Input image size |
| `--device` | auto | `cpu` or `0` for GPU |
| `--export` | off | Export to ONNX after training |

> **Tip:** Start with `yolov8n.pt` (nano) for fast experimentation, then scale up to `yolov8s.pt` or `yolov8m.pt` for better accuracy.

After training, best weights are saved at:
```
runs/train/exp/weights/best.pt
```

---

## 🔍 Detection (CLI)

### Image

```bash
# Single image (uses yolov8n.pt pretrained — detects 80 COCO classes)
python detect.py --source photo.jpg

# With custom weights
python detect.py --source photo.jpg --weights runs/train/exp/weights/best.pt

# Folder of images
python detect.py --source images/

# Adjust thresholds
python detect.py --source photo.jpg --conf 0.4 --iou 0.5
```

### Video

```bash
python detect.py --source video.mp4 --mode video
python detect.py --source video.mp4 --mode video --weights best.pt --no-display
```

### Webcam

```bash
python detect.py --mode webcam

# Press Q to quit | S to save a snapshot
```

### All CLI options

```bash
python detect.py --help
```

---

## 🖥️ Streamlit Dashboard

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Dashboard features:

| Tab | Feature |
|---|---|
| 🖼️ Image | Upload image → detect → download annotated result |
| 🎬 Video | Upload video → process all frames → download annotated video |
| 📷 Webcam | Live real-time detection from connected camera |
| 🏋️ Train | Configure and launch training from the UI |

---

## 📊 YOLO Annotation Format

Each image needs a `.txt` annotation file with one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** (0.0–1.0) relative to image dimensions.

**Example** (`dog_001.txt`):
```
1 0.512 0.437 0.321 0.489
0 0.230 0.310 0.180 0.260
```

Line 1: class 1 (dog) centered at 51.2% x, 43.7% y, 32.1% wide, 48.9% tall.  
Line 2: class 0 (cat) centered at 23.0% x, 31.0% y, 18.0% wide, 26.0% tall.

> **Recommended labeling tools:** [Roboflow](https://roboflow.com), [LabelImg](https://github.com/HumanSignal/labelImg), [CVAT](https://cvat.org)

---

## 🤖 YOLOv8 Model Variants

| Model | Size | mAP50-95 | Speed (CPU) | Use case |
|---|---|---|---|---|
| YOLOv8n | 3.2M | 37.3 | ~80ms | Edge / real-time |
| YOLOv8s | 11.2M | 44.9 | ~128ms | Balanced |
| YOLOv8m | 25.9M | 50.2 | ~234ms | High accuracy |
| YOLOv8l | 43.7M | 52.9 | ~375ms | Very high accuracy |
| YOLOv8x | 68.2M | 53.9 | ~479ms | Maximum accuracy |

---

## ⚙️ Performance Tips for Low-End Systems

- Use `yolov8n.pt` (nano) — fastest and smallest.
- Reduce `--imgsz` to `320` or `416`.
- Reduce `--batch` to `4` or `2`.
- Set `--workers 0` on Windows.
- Disable display with `--no-display` for faster video processing.
- Use `--device cpu` if GPU causes issues.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce `--batch` and `--imgsz` |
| `No module named ultralytics` | Run `pip install ultralytics` |
| `Cannot open camera` | Check another app isn't using it; try `--webcam-id 1` |
| `No images found` | Check images are in `dataset/images/train/` |
| `Label files not found` | Ensure `.txt` files match image base names |
| Webcam not working in Streamlit | Camera must be local to the Streamlit server |
| Streamlit multiprocessing error | Add `--server.runOnSave false` to the Streamlit command |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection framework
- [OpenCV](https://opencv.org) — image and video processing
- [Streamlit](https://streamlit.io) — web dashboard framework

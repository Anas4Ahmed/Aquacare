# 🌊 River Garbage Detector
Real-time webcam detection of garbage, rocks, grass, and water using YOLOv8.

---

## 📁 File Overview

```
river-garbage-detector/
├── quickstart.py           ← Run immediately, no training needed
├── 1_download_dataset.py   ← Pull labeled data from Roboflow
├── 2_train.py              ← Train custom YOLOv8 model
├── 3_detect_webcam.py      ← Live webcam detection (after training)
├── requirements.txt        ← All dependencies
└── output/                 ← Screenshots & recordings saved here
```

---

## ⚡ Option A — Run Immediately (No Training)

Detects garbage objects (bottles, cups, bags…) out of the box using a pretrained model.

```bash
pip install ultralytics opencv-python
python quickstart.py
```

That's it. No Roboflow account. No training. Opens your webcam instantly.

---

## 🎯 Option B — Train Custom River Model (Better Accuracy)

### Step 0 — Install everything
```bash
pip install -r requirements.txt
```

### Step 1 — Get a Roboflow API key
1. Go to https://roboflow.com and create a free account
2. Go to Settings → API Keys → copy your key

### Step 2 — Download dataset
```bash
python 1_download_dataset.py --api-key YOUR_KEY_HERE
```

This downloads a labeled river garbage dataset with 4 classes:
- **garbage** — plastic waste, bottles, litter
- **rock**    — obstacles in water
- **grass**   — riverbank reference
- **water**   — safe navigation zone

### Step 3 — Train the model (~20–40 min on CPU, ~5 min on GPU)
```bash
python 2_train.py
```

For faster training with a GPU:
```bash
python 2_train.py --device cuda --epochs 100
```

### Step 4 — Run live detection
```bash
python 3_detect_webcam.py
```

---

## 🎮 Controls (while running)

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `S` | Save screenshot to `output/` |
| `+` | Raise confidence threshold |
| `-` | Lower confidence threshold |
| `R` | Start/stop video recording |

---

## 🎨 Detection Colors

| Class   | Color  | Meaning |
|---------|--------|---------|
| garbage | 🔴 Red    | Collect it! |
| rock    | 🟠 Orange | Obstacle — avoid |
| grass   | 🟢 Green  | Riverbank edge |
| water   | 🔵 Blue   | Safe zone |

---

## ⚙️ Command Line Options

### quickstart.py
```
--cam 0        Camera index (0 = default webcam, 1 = second camera)
--model nano   Model size: nano / small / medium / large
--conf 0.35    Confidence threshold (0.0–1.0)
```

### 3_detect_webcam.py
```
--weights path/to/best.pt   Custom model weights
--cam 0                      Camera index
--conf 0.35                  Confidence threshold
--save-video                 Auto-record from start
--no-labels                  Hide text labels (boxes only)
```

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python    | 3.9     | 3.11+       |
| RAM       | 4 GB    | 8 GB+       |
| GPU       | Optional| NVIDIA (CUDA) |
| Webcam    | Any USB | 1080p preferred |

---

## 🔧 Troubleshooting

**Camera won't open:**
```bash
python quickstart.py --cam 1   # try camera index 1
```

**Out of memory during training:**
```bash
python 2_train.py --batch 8    # reduce batch size
```

**Low FPS:**
```bash
python 3_detect_webcam.py --imgsz 320   # smaller inference size
```

**Roboflow download fails:**
The script will suggest alternative public datasets you can use instead.

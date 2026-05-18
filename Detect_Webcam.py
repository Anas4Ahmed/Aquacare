"""
Step 3: Real-time webcam garbage detection
==========================================
Opens your webcam, runs YOLOv8 inference every frame, and draws
color-coded boxes with confidence scores.

Class colors:
  garbage → RED    (collect it!)
  rock    → ORANGE (obstacle!)
  grass   → GREEN  (riverbank)
  water   → BLUE   (safe zone)

Controls:
  Q or ESC  → Quit
  S         → Save screenshot
  +/-       → Adjust confidence threshold
  R         → Toggle recording

Usage:
  python 3_detect_webcam.py
  python 3_detect_webcam.py --weights yolov8n.pt        (pretrained, no training needed)
  python 3_detect_webcam.py --weights runs/river_garbage/weights/best.pt
  python 3_detect_webcam.py --cam 1                     (second webcam)
  python 3_detect_webcam.py --conf 0.4 --save-video
"""

import argparse
import os
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ── Class styling ──────────────────────────────────────────────────────────────
CLASS_STYLES = {
    "biodegradable": {"color": (30, 180, 30), "emoji": "🍃"},
    "cardboard":    {"color": (0, 140, 255), "emoji": "📦"},
    "glass":        {"color": (230, 230, 230), "emoji": "🍾"},
    "metal":        {"color": (150, 150, 150), "emoji": "🥫"},
    "paper":        {"color": (200, 200, 200), "emoji": "📄"},
    "plastic":      {"color": (0, 0, 220), "emoji": "🥤"},
}
# Fallback colors for classes not in CLASS_STYLES
FALLBACK_COLORS = [
    (180,  50, 255),
    (255, 100,  50),
    (50,  200, 200),
    (200, 200,  50),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time river garbage detector")
    parser.add_argument("--weights",    default="",    help="Model weights path (.pt)")
    parser.add_argument("--cam",        default=0, type=int, help="Camera index (default 0)")
    parser.add_argument("--conf",       default=0.35, type=float, help="Confidence threshold")
    parser.add_argument("--iou",        default=0.45, type=float, help="NMS IoU threshold")
    parser.add_argument("--imgsz",      default=640, type=int, help="Inference image size")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--no-labels",  action="store_true", help="Hide class labels")
    return parser.parse_args()


def resolve_weights(weights_arg):
    """Figure out which model weights to use."""
    # 1. Explicitly provided
    if weights_arg and os.path.exists(weights_arg):
        return weights_arg

    # 2. From config.yaml (after training)
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        if "weights" in cfg and os.path.exists(cfg["weights"]):
            print(f"📦 Using trained weights: {cfg['weights']}")
            return cfg["weights"]

    # 3. Default pretrained YOLOv8n (works immediately, no training needed)
    print("ℹ️  No custom weights found — using pretrained YOLOv8n (COCO classes)")
    print("   For custom garbage detection, train first:  python 2_train.py\n")
    return "yolov8n.pt"


def get_class_style(class_name, class_idx):
    """Return color + label for any class name."""
    name_lower = class_name.lower()
    for key, style in CLASS_STYLES.items():
        if key in name_lower:
            return style["color"], style.get("alert", class_name.upper())
    color = FALLBACK_COLORS[class_idx % len(FALLBACK_COLORS)]
    return color, class_name.upper()


def draw_box(frame, x1, y1, x2, y2, label, conf, color, show_label=True):
    """Draw a detection box with label."""
    # Box
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if not show_label:
        return

    # Label background
    text = f"{label}  {conf:.0%}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thick = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

    label_y = max(y1 - 8, th + 8)
    cv2.rectangle(frame,
                  (x1, label_y - th - baseline - 4),
                  (x1 + tw + 8, label_y + baseline - 4),
                  color, -1)

    # White or dark text depending on brightness
    brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    cv2.putText(frame, text, (x1 + 4, label_y - 4),
                font, font_scale, text_color, font_thick, cv2.LINE_AA)


def draw_hud(frame, fps, conf_thresh, detections, recording):
    """Draw the heads-up display overlay."""
    h, w = frame.shape[:2]

    # ── Top bar ───────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "RIVER GARBAGE DETECTOR",
                (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}   CONF: {conf_thresh:.0%}",
                (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 20, 20), 7, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 50, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # ── Bottom status bar ─────────────────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 36), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    # Count per class
    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    status_parts = []
    for key, style in CLASS_STYLES.items():
        n = counts.get(key, 0)
        status_parts.append(f"{style['emoji']} {key}: {n}")

    status_text = "   ".join(status_parts)
    cv2.putText(frame, status_text,
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Controls hint
    cv2.putText(frame, "Q:quit  S:save  +/-:conf  R:record",
                (w - 290, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    # ── Garbage alert flash ───────────────────────────────────────────────────
    if "garbage" in counts and counts["garbage"] > 0:
        t = time.time()
        if int(t * 2) % 2 == 0:  # blink every 0.5s
            cv2.rectangle(frame, (0, 48), (w, 90), (0, 0, 180), -1)
            cv2.putText(frame, f"⚠  {counts['garbage']} GARBAGE DETECTED!",
                        (12, 78), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def run_detection(weights, cam_idx, conf_thresh, iou_thresh, img_size, save_video, show_labels):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run:  pip install ultralytics")
        sys.exit(1)

    print(f"\n🔄 Loading model: {weights}")
    model = YOLO(weights)
    class_names = model.names  # {0: 'garbage', 1: 'rock', ...}
    print(f"✅ Model loaded — {len(class_names)} classes: {list(class_names.values())}")

    # ── Open webcam ───────────────────────────────────────────────────────────
    print(f"\n📷 Opening camera {cam_idx}...")
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {cam_idx}")
        print("   Try --cam 0 or --cam 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐 Resolution: {actual_w}×{actual_h}")

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    recording = save_video
    os.makedirs("output", exist_ok=True)

    if save_video:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"output/detection_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 20, (actual_w, actual_h))
        print(f"🎬 Recording to: {out_path}")

    print("\n" + "═" * 60)
    print("   🎥 Detection LIVE — Controls:")
    print("   Q / ESC  → Quit")
    print("   S        → Save screenshot")
    print("   +        → Increase confidence threshold")
    print("   -        → Decrease confidence threshold")
    print("   R        → Toggle video recording")
    print("═" * 60 + "\n")

    # ── FPS tracking ──────────────────────────────────────────────────────────
    fps         = 0.0
    frame_count = 0
    t_start     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame capture failed — retrying...")
            time.sleep(0.1)
            continue

        # ── Inference ─────────────────────────────────────────────────────────
        results = model(frame,
                        imgsz=img_size,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        verbose=False)[0]

        # ── Parse detections ──────────────────────────────────────────────────
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_idx  = int(box.cls[0])
                conf_val = float(box.conf[0])
                label    = class_names.get(cls_idx, f"class_{cls_idx}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color, alert = get_class_style(label, cls_idx)
                detections.append({
                    "label": label,
                    "conf":  conf_val,
                    "box":   (x1, y1, x2, y2),
                    "color": color,
                    "alert": alert,
                })

                draw_box(frame, x1, y1, x2, y2, alert, conf_val, color, show_labels)

        # ── FPS calc ──────────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t_start
        if elapsed >= 0.5:
            fps     = frame_count / elapsed
            frame_count = 0
            t_start = time.time()

        # ── HUD overlay ───────────────────────────────────────────────────────
        draw_hud(frame, fps, conf_thresh, detections, recording)

        # ── Show ──────────────────────────────────────────────────────────────
        cv2.imshow("River Garbage Detector", frame)

        if writer and recording:
            writer.write(frame)

        # ── Terminal log (garbage only) ───────────────────────────────────────
        for d in detections:
            if "garbage" in d["label"].lower():
                print(f"[{datetime.now().strftime('%H:%M:%S')}]  "
                      f"🗑️  GARBAGE  {d['conf']:.0%}  at box {d['box']}")

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # Q or ESC
            break

        elif key == ord("s"):      # Screenshot
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"output/screenshot_{ts}.jpg"
            cv2.imwrite(path, frame)
            print(f"📸 Screenshot saved: {path}")

        elif key == ord("+") or key == ord("="):
            conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
            print(f"🔧 Confidence threshold: {conf_thresh:.0%}")

        elif key == ord("-"):
            conf_thresh = max(0.05, round(conf_thresh - 0.05, 2))
            print(f"🔧 Confidence threshold: {conf_thresh:.0%}")

        elif key == ord("r"):
            if writer is None:
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = f"output/detection_{ts}.mp4"
                fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                writer   = cv2.VideoWriter(out_path, fourcc, 20, (actual_w, actual_h))
                recording = True
                print(f"🔴 Recording started: {out_path}")
            else:
                writer.release()
                writer    = None
                recording = False
                print("⏹️  Recording stopped")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\n✅ Detection stopped. Goodbye!\n")


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("   🌊  RIVER GARBAGE DETECTOR — Live Webcam")
    print("═" * 60)

    args    = parse_args()
    weights = resolve_weights(args.weights)

    run_detection(
        weights    = weights,
        cam_idx    = args.cam,
        conf_thresh= args.conf,
        iou_thresh = args.iou,
        img_size   = args.imgsz,
        save_video = args.save_video,
        show_labels= not args.no_labels,
    )

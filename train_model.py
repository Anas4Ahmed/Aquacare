"""
Step 2: Train YOLOv8 on river garbage dataset
=============================================
Trains YOLOv8-nano (smallest + fastest) to detect:
  - garbage   → red boxes
  - rock      → orange boxes
  - grass     → green boxes
  - water     → blue boxes

Usage:
  python 2_train.py
  python 2_train.py --epochs 50 --model yolov8s.pt  (larger model)
"""

import argparse
import os
import sys
import yaml
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 river garbage detector")
    parser.add_argument("--config",  default="config.yaml",  help="Config from step 1")
    parser.add_argument("--epochs",  default=50,  type=int,  help="Training epochs (default 50)")
    parser.add_argument("--model",   default="",             help="Override base model (e.g. yolov8s.pt)")
    parser.add_argument("--batch",   default=16,  type=int,  help="Batch size (default 16)")
    parser.add_argument("--device",  default="",             help="cuda / cpu / mps (auto if blank)")
    return parser.parse_args()


def print_banner():
    print("\n" + "═" * 60)
    print("   🌊  RIVER GARBAGE DETECTOR — Training")
    print("═" * 60)


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        print("   Run step 1 first:  python 1_download_dataset.py --api-key YOUR_KEY")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def patch_data_yaml(cfg):
    """Make sure data.yaml has absolute paths (required by YOLO)."""
    data_yaml = cfg["data_yaml"]
    if not os.path.exists(data_yaml):
        print(f"❌ data.yaml not found at {data_yaml}")
        sys.exit(1)

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    dataset_root = cfg["dataset_path"]

    # Fix relative paths → absolute
    for split in ("train", "val", "test"):
        if split in data and not os.path.isabs(str(data[split])):
            data[split] = os.path.join(dataset_root, data[split])

    with open(data_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ data.yaml patched with absolute paths")
    return data_yaml


def train(cfg, epochs, model_override, batch, device):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run:  pip install ultralytics")
        sys.exit(1)

    base_model = model_override or cfg.get("model", "yolov8n.pt")
    img_size   = cfg.get("img_size", 640)
    data_yaml  = patch_data_yaml(cfg)

    print(f"\n📦 Base model  : {base_model}")
    print(f"📋 Classes     : {cfg['classes']}")
    print(f"🖼️  Image size  : {img_size}")
    print(f"🔁 Epochs      : {epochs}")
    print(f"📦 Batch size  : {batch}")
    print(f"⚙️  Device      : {device or 'auto'}\n")

    model = YOLO(base_model)

    train_kwargs = dict(
        data      = data_yaml,
        epochs    = epochs,
        imgsz     = img_size,
        batch     = batch,
        name      = "river_garbage",
        project   = "runs",
        patience  = 15,          # early stopping
        plots     = True,
        save      = True,
        exist_ok  = True,
    )
    if device:
        train_kwargs["device"] = device

    print("🚀 Training started...\n")
    results = model.train(**train_kwargs)

    # ── Find best weights ─────────────────────────────────────────────────────
    best_weights = Path("runs/river_garbage/weights/best.pt")
    if not best_weights.exists():
        best_weights = Path("runs/river_garbage/weights/last.pt")

    print(f"\n✅ Training complete!")
    print(f"🏆 Best weights: {best_weights}")

    # ── Save path to config for detection script ──────────────────────────────
    cfg["weights"] = str(best_weights)
    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("\n📊 Training results saved to: runs/river_garbage/")
    print("═" * 60)
    print("   Next step:")
    print("   python 3_detect_webcam.py")
    print("═" * 60 + "\n")

    return str(best_weights)


if __name__ == "__main__":
    print_banner()
    args  = parse_args()
    cfg   = load_config(args.config)
    train(cfg, args.epochs, args.model, args.batch, args.device)

"""
Step 1: Download dataset from Roboflow API
==========================================
Gets a river garbage detection dataset with 4 classes:
  - garbage
  - rock
  - grass
  - water

Usage:
  python 1_download_dataset.py --api-key YOUR_KEY
  python 1_download_dataset.py --api-key YOUR_KEY --workspace myworkspace --project myproject
"""

import argparse
import os
import sys
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Download river garbage dataset from Roboflow")
    parser.add_argument("--api-key",   required=True,  help="CH31pKeEcmuUkxBPPT2L")
    parser.add_argument("--workspace", default="",     help="Roboflow workspace slug (optional)")
    parser.add_argument("--project",   default="",     help="Roboflow project slug (optional)")
    parser.add_argument("--version",   default=1, type=int, help="Dataset version (default: 1)")
    parser.add_argument("--output",    default="dataset", help="Output folder name")
    return parser.parse_args()


# ── Recommended public Roboflow datasets ─────────────────────────────────────
RECOMMENDED_DATASETS = [
    {
        "workspace": "material-identification",
        "project":   "garbage-classification-3",
        "version":   2,
        "note":      "General garbage classification (bottles, plastic, etc.)"
    },
    {
        "workspace": "roboflow-100",
        "project":   "aquarium-qlnqy",
        "version":   2,
        "note":      "Aquatic environment detection"
    },
    {
        "workspace": "recyclables",
        "project":   "river-waste-detection",
        "version":   1,
        "note":      "River-specific waste detection ✅ BEST MATCH"
    },
]


def print_banner():
    print("\n" + "═" * 60)
    print("   🌊  RIVER GARBAGE DETECTOR — Dataset Downloader")
    print("═" * 60)


def show_recommended():
    print("\n📦 Recommended public Roboflow datasets:")
    for i, ds in enumerate(RECOMMENDED_DATASETS, 1):
        print(f"  {i}. {ds['workspace']}/{ds['project']} v{ds['version']}")
        print(f"     └─ {ds['note']}")
    print()


def download(api_key, workspace, project, version, output_dir):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌  roboflow not installed. Run:  pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)

    if not workspace or not project:
        # Auto-use best matching public dataset
        best = RECOMMENDED_DATASETS[2]  # river-waste-detection
        workspace = best["workspace"]
        project   = best["project"]
        version   = best["version"]
        print(f"ℹ️  No project specified — using: {workspace}/{project} v{version}")

    print(f"\n⬇️  Downloading: {workspace}/{project}  version {version}")
    print("    Format: YOLOv8 (for training)\n")

    try:
        proj    = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=output_dir)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n💡 Try one of the recommended datasets above with:")
        print("   python 1_download_dataset.py --api-key YOUR_KEY "
              "--workspace material-identification "
              "--project garbage-classification-3 --version 2")
        sys.exit(1)

    dataset_path = dataset.location
    print(f"\n✅ Dataset saved to: {dataset_path}")

    # ── Parse class names from data.yaml ─────────────────────────────────────
    yaml_path = os.path.join(dataset_path, "data.yaml")
    classes = ["garbage", "rock", "grass", "water"]  # fallback defaults

    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data_yaml = yaml.safe_load(f)
        if "names" in data_yaml:
            classes = data_yaml["names"]
            print(f"📋 Detected classes: {classes}")

    # ── Save unified config for training script ───────────────────────────────
    config = {
        "dataset_path": dataset_path,
        "data_yaml":    yaml_path,
        "classes":      classes,
        "num_classes":  len(classes),
        "img_size":     640,
        "model":        "yolov8n.pt",   # nano = fastest
    }
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("📝 config.yaml saved — ready for training!\n")
    print("═" * 60)
    print("   Next step:")
    print("   python 2_train.py")
    print("═" * 60 + "\n")

    return config


if __name__ == "__main__":
    print_banner()
    show_recommended()
    args = parse_args()
    download(args.api_key, args.workspace, args.project, args.version, args.output)

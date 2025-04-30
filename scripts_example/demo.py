import yaml
import os
from utils import load_image, save_image
from pipeline_loader import TRANSFORM_REGISTRY
import glob
from pipeline_loader import load_pipeline_from_yaml


def apply_each_transform(yaml_path, mode, out_dir, img):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    steps = config["pipeline"]
    mode = config.get("mode", mode)

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        for name, params in step.items():
            if name in TRANSFORM_REGISTRY:
                transform = TRANSFORM_REGISTRY[name](mode=mode, **params)
                out_img = transform(img)
                out_name = f"{i:02d}_{mode}_{name.lower()}.png"
                save_image(out_img, os.path.join(out_dir, out_name))
                print(f"[✓] Saved: {out_name}")
            else:
                print(f"[!] Skipped unsupported transform: {name}")
def apply_and_save(yaml_path, mode, out_dir, img):
    pipeline = load_pipeline_from_yaml(yaml_path)
    pipeline.mode = mode

    out_img = pipeline(img)
    save_path = os.path.join(out_dir, f"{mode}_pipeline_output.png")
    save_image(out_img, save_path)
    print(f"[✓] Saved full pipeline output: {save_path}")

# === Paths
yaml_path = "../configs/all_basic.yaml"
output_random = "../testoutput/demo_output_random"
output_fixed = "../testoutput/demo_output_fixed"
os.makedirs(output_random, exist_ok=True)
os.makedirs(output_fixed, exist_ok=True)
folder_path = "../../../Data/monkbrill_cleaned/Alef"
image_list = sorted(glob.glob(os.path.join(folder_path, "*.png")))

if not image_list:
    print(f"[ERROR] No PNG images found in {folder_path}")
    exit(1)

img_path = image_list[0]  # 取第一张图像作为样例
print(f"[INFO] Using image: {img_path}")

# === Load image
img = load_image(img_path)
if img is None:
    print(f"[ERROR] Failed to load image: {img_path}")
    exit(1)

# === Per-transform augmentation
apply_each_transform(yaml_path, "random", output_random, img)
apply_each_transform(yaml_path, "fixed", output_fixed, img)

# === Full-pipeline augmentation
apply_and_save(yaml_path, "random", output_random, img)
apply_and_save(yaml_path, "fixed", output_fixed, img)

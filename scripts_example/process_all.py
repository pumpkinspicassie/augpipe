import os
from tqdm import tqdm
from utils import load_image, save_image
from pipeline_loader import load_pipeline_from_yaml

# === Path to the YAML config file ===
config_path = "../configs/all_basic.yaml"

# === Input and output root directories ===
input_root = "../../../Data/monkbrill_cleaned"
output_root = "../testoutput/output_all"
os.makedirs(output_root, exist_ok=True)

# === Load augmentation pipeline from YAML ===
pipeline = load_pipeline_from_yaml(config_path)

# === Set number of augmented versions per image ===
N = 5

# === Collect all valid image paths with relative subfolders preserved ===
image_paths = []
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, file)
            rel_dir = os.path.relpath(root, input_root)
            image_paths.append((img_path, rel_dir, file))

# === Now process with tqdm ===
for img_path, rel_dir, file in tqdm(image_paths, desc="Augmenting images"):
    img = load_image(img_path)
    if img is None:
        print(f"[WARN] Failed to load: {img_path}")
        continue

    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    basename = os.path.splitext(file)[0]
    for i in range(N):
        augmented = pipeline(img.copy())
        out_path = os.path.join(out_dir, f"{basename}_aug_{i+1}.png")
        save_image(augmented, out_path)

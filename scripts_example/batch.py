import os
from tqdm import tqdm
from utils import load_image, save_image
from pipeline_loader import load_pipeline_from_yaml  


config_path = "../configs/all_basic.yaml"  


input_dir = "../../../Data/monkbrill_cleaned/Alef"
output_dir = "../testoutput/batch_output_n"
os.makedirs(output_dir, exist_ok=True)


pipeline = load_pipeline_from_yaml(config_path)


N = 5  


for filename in tqdm(os.listdir(input_dir)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, filename)
    img = load_image(img_path)
    basename = os.path.splitext(filename)[0]

    for i in range(N):
        augmented = pipeline(img.copy())
        save_path = os.path.join(output_dir, f"{basename}_aug_{i+1}.png")
        save_image(augmented, save_path)

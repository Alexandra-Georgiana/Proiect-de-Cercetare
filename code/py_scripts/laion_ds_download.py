import os
from dagshub.streaming import DagsHubFilesystem
from PIL import Image
import requests

# ------------------------------------------------------------------
# Configure DagsHub LAION Aesthetics dataset
# ------------------------------------------------------------------
fs = DagsHubFilesystem(
    '.', 
    repo_url='https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus'
)
fs.install_hooks()

# Output directory
out_dir = "laion_subset"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------
# Read labels.tsv with UTF-8 decoding
# ------------------------------------------------------------------
num_to_download = 100
count = 0

with fs.open('data/labels.tsv', mode='r', encoding='utf-8') as tsv:
    for line in tsv:
        if count >= num_to_download:
            break

        line = line.strip()
        img_file, caption, score, url = line.split('\t')

        try:
            # stream and save image
            img_path_remote = os.path.join('data', img_file)
            with fs.open(img_path_remote, 'rb') as remote_img:
                img = Image.open(remote_img)
                img.save(os.path.join(out_dir, img_file))

            print(f"[{count+1}] Saved {img_file} | aesthetics score: {score}")
            count += 1

        except Exception as e:
            print(f"Could not load {img_file}: {e}")
            continue

print("Done! Downloaded", count, "images.")

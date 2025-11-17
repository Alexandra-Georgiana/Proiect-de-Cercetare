# ai_art_analysis.py
# Full pipeline for AI art style and originality analysis
# Requires: torch, torchvision, open_clip_torch, opendino, umap-learn, scikit-learn, matplotlib, pillow, pandas

import os
from PIL import Image
import torch
import open_clip
import numpy as np
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --------------------------
# SETTINGS
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
folder_unguided = "D:/Cercetare/AI-Unguided"  # replace with your path
folder_guided = "D:/Cercetare/AI-Guided"      # replace with your path
n_clusters = 5  # number of style clusters
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# --------------------------
# LOAD MODELS
# --------------------------
print("Loading models...")
clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(device).eval()

# Placeholder for DINOv2 model
# from dino import DINOModel
# dino_model = DINOModel(pretrained='dino_vits16')
# dino_model = dino_model.to(device).eval()
# For simplicity here we will use only CLIP embeddings; DINOv2 can be added similarly

# --------------------------
# LOAD IMAGES
# --------------------------
def load_images_from_folder(folder_path):
    images = []
    file_names = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            file_names.append(filename)
    return images, file_names

print("Loading images...")
images_unguided, names_unguided = load_images_from_folder(folder_unguided)
images_guided, names_guided = load_images_from_folder(folder_guided)

# --------------------------
# EXTRACT EMBEDDINGS
# --------------------------
def get_clip_embeddings(images):
    embeddings = []
    for img in images:
        x = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

print("Extracting embeddings...")
clip_emb_unguided = get_clip_embeddings(images_unguided)
clip_emb_guided = get_clip_embeddings(images_guided)

# --------------------------
# CLUSTERING
# --------------------------
print("Clustering embeddings...")
all_embeddings = np.vstack([clip_emb_unguided, clip_emb_guided])
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(all_embeddings)
labels = kmeans.labels_
labels_unguided = labels[:len(clip_emb_unguided)]
labels_guided = labels[len(clip_emb_unguided):]

# --------------------------
# NOVELTY / ORIGINALITY
# --------------------------
def compute_novelty(embeddings):
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    novelty_scores = 1 - sim_matrix.max(axis=1)
    return novelty_scores

novelty_unguided = compute_novelty(clip_emb_unguided)
novelty_guided = compute_novelty(clip_emb_guided)

# --------------------------
# SAVE RESULTS
# --------------------------
print("Saving results...")
results_unguided = pd.DataFrame({
    'Image': names_unguided,
    'Cluster': labels_unguided,
    'NoveltyScore': novelty_unguided
})
results_guided = pd.DataFrame({
    'Image': names_guided,
    'Cluster': labels_guided,
    'NoveltyScore': novelty_guided
})

results_unguided.to_csv(os.path.join(output_folder, "Unguided_Results.csv"), index=False)
results_guided.to_csv(os.path.join(output_folder, "Guided_Results.csv"), index=False)

# --------------------------
# VISUALIZATION
# --------------------------
print("Generating UMAP visualization...")
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embedding_2d = reducer.fit_transform(all_embeddings)

plt.figure(figsize=(10,8))
plt.scatter(embedding_2d[:len(clip_emb_unguided),0], embedding_2d[:len(clip_emb_unguided),1], c='red', label='Unguided AI')
plt.scatter(embedding_2d[len(clip_emb_unguided):,0], embedding_2d[len(clip_emb_unguided):,1], c='blue', label='Guided AI')
plt.legend()
plt.title("AI Art Style Clustering (UMAP)")
plt.savefig(os.path.join(output_folder, "UMAP_Clustering.png"))
plt.show()

print("Done. Results saved in folder:", output_folder)

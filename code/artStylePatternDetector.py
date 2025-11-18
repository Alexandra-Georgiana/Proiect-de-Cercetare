# ai_style_analysis.py
# Full pipeline for AI art emergent style analysis
# Requirements: torch, open_clip_torch, numpy, pandas, scikit-learn, umap-learn, matplotlib, pillow

import os
from PIL import Image
import torch
import open_clip
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt

# --------------------------
# SETTINGS
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
folder_unguided = "D:\\Cercetare\\AI-Unguided"  # replace with your path
folder_guided = "D:\\Cercetare\\AI-Guided"      # replace with your path
n_clusters = 5  # number of style clusters
output_folder = "resultsArtStyle"
os.makedirs(output_folder, exist_ok=True)

# --------------------------
# LOAD MODELS
# --------------------------
print("Loading CLIP model...")
clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(device).eval()

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def load_images(folder_path):
    images, names = [], []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            images.append(Image.open(path).convert("RGB"))
            names.append(filename)
    return images, names

def extract_clip_embeddings(images):
    embeddings = []
    for img in images:
        x = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

def color_histogram(img, bins=16):
    arr = np.array(img).astype(np.float32) / 255.0
    hist = []
    for i in range(3):  # RGB
        channel_hist, _ = np.histogram(arr[:,:,i], bins=bins, range=(0,1), density=True)
        hist.append(channel_hist)
    return np.concatenate(hist)

def texture_metric(img):
    arr = np.array(img.convert("L")).astype(np.float32)
    contrast = arr.std() / 255.0
    return np.array([contrast])

def compute_novelty(embeddings):
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    novelty_scores = 1 - sim_matrix.max(axis=1)
    return novelty_scores

# --------------------------
# LOAD IMAGES
# --------------------------
print("Loading images...")
images_unguided, names_unguided = load_images(folder_unguided)
images_guided, names_guided = load_images(folder_guided)

# --------------------------
# FEATURE EXTRACTION
# --------------------------
print("Extracting features...")
clip_emb_unguided = extract_clip_embeddings(images_unguided)
clip_emb_guided = extract_clip_embeddings(images_guided)

color_emb_unguided = np.array([color_histogram(img) for img in images_unguided])
color_emb_guided = np.array([color_histogram(img) for img in images_guided])

texture_emb_unguided = np.array([texture_metric(img) for img in images_unguided])
texture_emb_guided = np.array([texture_metric(img) for img in images_guided])

# Combine all features
features_unguided = np.hstack([clip_emb_unguided, color_emb_unguided, texture_emb_unguided])
features_guided = np.hstack([clip_emb_guided, color_emb_guided, texture_emb_guided])

# --------------------------
# CLUSTERING
# --------------------------
print("Clustering images...")
all_features = np.vstack([features_unguided, features_guided])
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(all_features)
labels = kmeans.labels_
labels_unguided = labels[:len(features_unguided)]
labels_guided = labels[len(features_unguided):]

# --------------------------
# NOVELTY / COHESION
# --------------------------
novelty_unguided = compute_novelty(features_unguided)
novelty_guided = compute_novelty(features_guided)

def cluster_cohesion(features, labels):
    cohesion = []
    for cluster in np.unique(labels):
        members = features[labels==cluster]
        if len(members) > 1:
            sims = cosine_similarity(members)
            np.fill_diagonal(sims, 0)
            cohesion.append(sims.mean())
        else:
            cohesion.append(1.0)
    return np.mean(cohesion)

cohesion_unguided = cluster_cohesion(features_unguided, labels_unguided)
cohesion_guided = cluster_cohesion(features_guided, labels_guided)

# --------------------------
# SAVE RESULTS
# --------------------------
print("Saving results...")
df_unguided = pd.DataFrame({
    "Image": names_unguided,
    "Cluster": labels_unguided,
    "NoveltyScore": novelty_unguided
})
df_guided = pd.DataFrame({
    "Image": names_guided,
    "Cluster": labels_guided,
    "NoveltyScore": novelty_guided
})
df_unguided.to_csv(os.path.join(output_folder, "Unguided_Results.csv"), index=False)
df_guided.to_csv(os.path.join(output_folder, "Guided_Results.csv"), index=False)

# --------------------------
# UMAP VISUALIZATION
# --------------------------
print("Generating UMAP visualization...")
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embedding_2d = reducer.fit_transform(all_features)

plt.figure(figsize=(10,8))
plt.scatter(embedding_2d[:len(features_unguided),0], embedding_2d[:len(features_unguided),1],
            c='red', label='Unguided AI')
plt.scatter(embedding_2d[len(features_unguided):,0], embedding_2d[len(features_unguided):,1],
            c='blue', label='Guided AI')
plt.legend()
plt.title("AI Art Emergent Style Clustering (UMAP)")
plt.savefig(os.path.join(output_folder, "UMAP_Clustering.png"))
plt.show()

print("Done. Cohesion (Unguided):", cohesion_unguided, "Cohesion (Guided):", cohesion_guided)
print("Results saved in folder:", output_folder)

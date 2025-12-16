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
import shutil

# --------------------------
# SETTINGS
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
folder_unguided = "D:/Cercetare/AI-Unguided"  
folder_guided = "D:/Cercetare/AI-Guided"     
folder_training = "D:/Cercetare/laion_subset" 
n_clusters = 5  
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
images_training, _ = load_images_from_folder(folder_training) # Load training images

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
clip_emb_training = get_clip_embeddings(images_training) # Get training embeddings

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
# ADVANCED ORIGINALITY (SYNTHESIS SCORE)
# --------------------------
def compute_synthesis_score(generated_embeddings, training_embeddings, k=5):
    """
    Computes a synthesis score by comparing a generated image to a blend of the
    'k' most similar images from the training set.
    A lower score indicates higher originality.
    """
    # Calculate cosine similarity between each generated embedding and all training embeddings
    sim_matrix = cosine_similarity(generated_embeddings, training_embeddings)
    
    synthesis_scores = []
    for i in range(len(generated_embeddings)):
        # Get the top k most similar training images for the current generated image
        top_k_indices = np.argsort(sim_matrix[i, :])[-k:]
        
        # Get the embeddings of these top k images
        top_k_embeddings = training_embeddings[top_k_indices]
        
        # Create the "synthesized" embedding by averaging the top k
        synthesized_embedding = np.mean(top_k_embeddings, axis=0, keepdims=True)
        
        # Calculate the similarity of the generated image to this new synthesized blend
        synthesis_similarity = cosine_similarity(generated_embeddings[i:i+1], synthesized_embedding)[0, 0]
        
        synthesis_scores.append(synthesis_similarity)
        
    # The final score is the similarity to the synthesized blend. Higher score = less original.
    return np.array(synthesis_scores)

print("Computing advanced synthesis scores...")
synthesis_unguided = compute_synthesis_score(clip_emb_unguided, clip_emb_training)
synthesis_guided = compute_synthesis_score(clip_emb_guided, clip_emb_training)


# --------------------------
# SAVE RESULTS
# --------------------------
print("Saving results...")
results_unguided = pd.DataFrame({
    'Image': names_unguided,
    'Cluster': labels_unguided,
    'SynthesisScore': synthesis_unguided
})
results_guided = pd.DataFrame({
    'Image': names_guided,
    'Cluster': labels_guided,
    'SynthesisScore': synthesis_guided
})

results_unguided.to_csv(os.path.join(output_folder, "Unguided_Results.csv"), index=False)
results_guided.to_csv(os.path.join(output_folder, "Guided_Results.csv"), index=False)

# --------------------------
# ORGANIZE IMAGES BY CLUSTER
# --------------------------
print("Organizing images into cluster folders...")
cluster_output_dir = os.path.join(output_folder, "style_clusters")
if os.path.exists(cluster_output_dir):
    shutil.rmtree(cluster_output_dir) # Clear old results
os.makedirs(cluster_output_dir)

for i in range(n_clusters):
    os.makedirs(os.path.join(cluster_output_dir, f"Cluster_{i}"))

# Process Guided images
for index, row in results_guided.iterrows():
    src_path = os.path.join(folder_guided, row['Image'])
    dst_path = os.path.join(cluster_output_dir, f"Cluster_{row['Cluster']}", row['Image'])
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

# Process Unguided images
for index, row in results_unguided.iterrows():
    src_path = os.path.join(folder_unguided, row['Image'])
    dst_path = os.path.join(cluster_output_dir, f"Cluster_{row['Cluster']}", row['Image'])
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

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
print("Image files have been sorted into subfolders inside:", cluster_output_dir)

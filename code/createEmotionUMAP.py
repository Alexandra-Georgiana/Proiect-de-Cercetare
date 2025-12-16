# createEmotionUMAP.py
# Generates a UMAP visualization of image embeddings, colored by dominant emotion.

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import open_clip
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# SETTINGS
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
results_folder = "D:/Cercetare/code/resultsEmotionalAnalysis"
guided_csv = os.path.join(results_folder, "Guided_Emotion_Results.csv")
unguided_csv = os.path.join(results_folder, "Unguided_Emotion_Results.csv")

folder_guided = "D:/Cercetare/AI-Guided"
folder_unguided = "D:/Cercetare/AI-Unguided"

output_plot_file = os.path.join(results_folder, "Emotion_UMAP.png")

# --------------------------
# LOAD RESULTS DATA
# --------------------------
print("Loading emotion results data...")
try:
    df_guided = pd.read_csv(guided_csv)
    df_unguided = pd.read_csv(unguided_csv)
    
    # Add a column to map image names to their original folders
    df_guided['SourceFolder'] = folder_guided
    df_unguided['SourceFolder'] = folder_unguided
    
    # Combine into a single dataframe
    combined_df = pd.concat([df_guided, df_unguided], ignore_index=True)
    print(f"Successfully loaded data for {len(combined_df)} images.")

except FileNotFoundError as e:
    print(f"Error: Could not find the results file. {e}")
    print("Please run emotionAnalyzer.py first.")
    exit()

# --------------------------
# LOAD MODEL & IMAGES
# --------------------------
print("Loading CLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(device).eval()

def load_images_from_df(dataframe):
    images = []
    for index, row in dataframe.iterrows():
        img_path = os.path.join(row['SourceFolder'], row['Image'])
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            images.append(None) # Add a placeholder
    return images

print("Loading images...")
images = load_images_from_df(combined_df)
# Filter out any images that failed to load
valid_indices = [i for i, img in enumerate(images) if img is not None]
images = [images[i] for i in valid_indices]
combined_df = combined_df.iloc[valid_indices].reset_index(drop=True)


# --------------------------
# EXTRACT EMBEDDINGS
# --------------------------
def get_clip_embeddings(image_list):
    embeddings = []
    with torch.no_grad():
        for img in image_list:
            x = clip_preprocess(img).unsqueeze(0).to(device)
            emb = clip_model.encode_image(x)
            emb /= emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

print("Extracting image embeddings...")
all_embeddings = get_clip_embeddings(images)

# --------------------------
# UMAP & VISUALIZATION
# --------------------------
print("Performing UMAP dimensionality reduction...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
embedding_2d = reducer.fit_transform(all_embeddings)

print("Generating UMAP plot...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(16, 12))

# Create a scatter plot
sns.scatterplot(
    x=embedding_2d[:, 0],
    y=embedding_2d[:, 1],
    hue=combined_df['DominantEmotion'],
    palette='viridis', 
    s=50, 
    alpha=0.8
)

plt.title('UMAP Projection of Image Embeddings by Dominant Emotion', fontsize=18, weight='bold')
plt.xlabel('UMAP Dimension 1', fontsize=12)
plt.ylabel('UMAP Dimension 2', fontsize=12)
plt.legend(title='Dominant Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig(output_plot_file, dpi=300)
print(f"UMAP plot saved successfully to: {output_plot_file}")

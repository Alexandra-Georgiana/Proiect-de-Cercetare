# emotionAnalyzer.py
# Analyzes the emotional content of AI-generated images.
# Requires: torch, torchvision, open_clip_torch, pillow, pandas

import os
from PIL import Image
import torch
import open_clip
import numpy as np
import pandas as pd

# --------------------------
# SETTINGS
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
folder_unguided = "D:/Cercetare/AI-Unguided"
folder_guided = "D:/Cercetare/AI-Guided"
output_folder = "resultsEmotionalAnalysis"
os.makedirs(output_folder, exist_ok=True)

# Define the set of emotions to test against
EMOTIONS = [
    "Joy", "Happiness", "Ecstasy",
    "Sadness", "Sorrow", "Grief",
    "Anger", "Rage", "Frustration",
    "Fear", "Anxiety", "Terror",
    "Surprise", "Awe", "Amazement",
    "Calm", "Peace", "Serenity",
    "Love", "Tenderness",
    "Disgust", "Contempt"
]

# --------------------------
# LOAD MODEL
# --------------------------
print("Loading CLIP model for emotion analysis...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(device).eval()

# --------------------------
# LOAD IMAGES
# --------------------------
def load_images_from_folder(folder_path):
    images = []
    file_names = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                file_names.append(filename)
            except Exception as e:
                print(f"Could not load image {filename}: {e}")
    return images, file_names

# --------------------------
# EMOTIONAL ANALYSIS
# --------------------------
def get_emotion_scores(images, image_names, emotions_list):
    """
    Calculates the dominant emotion and confidence score for a list of images.
    """
    # Prepare text prompts for the emotions
    text_prompts = [f"A painting expressing {emotion}" for emotion in emotions_list]
    text_tokens = open_clip.tokenize(text_prompts).to(device)

    results = []

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for img, name in zip(images, image_names):
            image_input = clip_preprocess(img).unsqueeze(0).to(device)
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity scores between the image and all emotion prompts
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Find the top emotion and its score
            top_score, top_index = similarity.topk(1)
            dominant_emotion = emotions_list[top_index.cpu().item()]
            accuracy_score = top_score.cpu().item()

            results.append({
                'Image': name,
                'DominantEmotion': dominant_emotion,
                'EmotionalAccuracy': accuracy_score
            })
            print(f"Analyzed {name}: Top emotion is {dominant_emotion} with score {accuracy_score:.2f}")

    return pd.DataFrame(results)

# --------------------------
# EXECUTE AND SAVE
# --------------------------
print("\n--- Analyzing Guided Images ---")
images_guided, names_guided = load_images_from_folder(folder_guided)
guided_emotion_results = get_emotion_scores(images_guided, names_guided, EMOTIONS)
guided_output_path = os.path.join(output_folder, "Guided_Emotion_Results.csv")
guided_emotion_results.to_csv(guided_output_path, index=False)
print(f"Guided emotion results saved to {guided_output_path}")

print("\n--- Analyzing Unguided Images ---")
images_unguided, names_unguided = load_images_from_folder(folder_unguided)
unguided_emotion_results = get_emotion_scores(images_unguided, names_unguided, EMOTIONS)
unguided_output_path = os.path.join(output_folder, "Unguided_Emotion_Results.csv")
unguided_emotion_results.to_csv(unguided_output_path, index=False)
print(f"Unguided emotion results saved to {unguided_output_path}")

print("\nEmotion analysis complete.")

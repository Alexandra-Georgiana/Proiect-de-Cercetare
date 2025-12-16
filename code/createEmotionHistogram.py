# createEmotionHistogram.py
# Generates a histogram to compare the distribution of Emotional Accuracy scores
# for guided and unguided AI-generated images.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------
# SETTINGS
# --------------------------
results_folder = "D:/Cercetare/code/resultsEmotionalAnalysis"
guided_csv = os.path.join(results_folder, "Guided_Emotion_Results.csv")
unguided_csv = os.path.join(results_folder, "Unguided_Emotion_Results.csv")
output_plot_file = os.path.join(results_folder, "Emotional_Accuracy_Histogram.png")

# --------------------------
# LOAD DATA
# --------------------------
print("Loading emotional analysis results data...")
try:
    df_guided = pd.read_csv(guided_csv)
    df_unguided = pd.read_csv(unguided_csv)
    print("Successfully loaded both guided and unguided emotion results.")

except FileNotFoundError as e:
    print(f"Error: Could not find the results file. Have you run the emotionAnalyzer.py script yet? {e}")
    exit()

# --------------------------
# GENERATE HISTOGRAM
# --------------------------
print("Generating histogram...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))

# Create the histograms for both datasets on the same plot
sns.histplot(df_guided['EmotionalAccuracy'], color='skyblue', label='Guided', kde=True, stat="density", linewidth=0)
sns.histplot(df_unguided['EmotionalAccuracy'], color='red', label='Unguided', kde=True, stat="density", linewidth=0, alpha=0.6)

# Add plot titles and labels
plt.title('Distribution of Emotional Accuracy Scores', fontsize=16, weight='bold')
plt.xlabel('Emotional Accuracy Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()

# Save the plot
plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
print(f"Histogram saved successfully to: {output_plot_file}")

# Optional: Show the plot
# plt.show()

print("\nAnalysis complete.")

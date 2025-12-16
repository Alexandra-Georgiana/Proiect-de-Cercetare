# createBoxplot.py
# Generates a boxplot to visualize the distribution of Synthesis Scores across stylistic clusters.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------
# SETTINGS
# --------------------------
results_folder = "D:/Cercetare/code/results"
guided_csv = os.path.join(results_folder, "Guided_Results.csv")
unguided_csv = os.path.join(results_folder, "Unguided_Results.csv")
output_plot_file = os.path.join(results_folder, "Synthesis_Score_Distribution_Boxplot.png")

# --------------------------
# LOAD AND COMBINE DATA
# --------------------------
print("Loading and combining results data...")
try:
    df_guided = pd.read_csv(guided_csv)
    df_unguided = pd.read_csv(unguided_csv)
    
    # Add a column to distinguish between the two groups
    df_guided['Group'] = 'Guided'
    df_unguided['Group'] = 'Unguided'
    
    # Combine into a single dataframe
    combined_df = pd.concat([df_guided, df_unguided], ignore_index=True)
    print(f"Successfully combined data with {len(combined_df)} total entries.")

except FileNotFoundError as e:
    print(f"Error: Could not find the results file. {e}")
    exit()

# --------------------------
# GENERATE BOXPLOT
# --------------------------
print("Generating boxplot...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))

# Create the boxplot
sns.boxplot(x='Cluster', y='SynthesisScore', data=combined_df, palette='viridis')

# Add plot titles and labels
plt.title('Distribution of Synthesis Scores Across Stylistic Clusters', fontsize=16, weight='bold')
plt.xlabel('Stylistic Cluster', fontsize=12)
plt.ylabel('Synthesis Score (Higher = Less Original)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Save the plot
plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
print(f"Boxplot saved successfully to: {output_plot_file}")

# Optional: Show the plot
# plt.show()

print("\nAnalysis complete.")

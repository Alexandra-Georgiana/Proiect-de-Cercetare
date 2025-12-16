@echo off
echo --- Ensuring required packages are installed ---
python -m pip install umap-learn seaborn

echo --- Running the emotion UMAP creation script ---
python createEmotionUMAP.py

echo --- Script finished ---
pause

@echo off
echo --- Ensuring seaborn is installed ---
python -m pip install seaborn

echo --- Running the boxplot creation script ---
python createBoxplot.py

echo --- Script finished ---
pause

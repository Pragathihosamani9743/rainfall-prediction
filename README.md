# Rainfall Prediction Project

This project predicts annual and daily rainfall using Random Forest regression on historical rainfall data from Gundlupet.

## Features
- Loads and preprocesses rainfall CSV data.
- Handles missing values and creates date-based features.
- Predicts annual rainfall from monsoon rainfall.
- Predicts daily rainfall for any given date.
- Evaluates model performance with RMSE and R².
- Visualizes actual vs predicted rainfall and daily rainfall trends.

## How to Use
1. Place your CSV data file (`Gundlupet.csv`) in the project folder.
2. Run the Python script.
3. Input a date (YYYY-MM-DD) to get daily rainfall prediction.
4. Input a year to get predicted daily rainfall plot for that year.

## Requirements
- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn

## Sample Prediction Output
 Data loaded. Shape: (12784, 22)

📊 Annual Rainfall Model Evaluation:
Training RMSE: 47.97281580117237
Training R² Score: 0.9264243547139512
Testing RMSE: 148.37259550145617
Testing R² Score: 0.3683932405584672

🌧 Predicted Annual Rainfall for 2025: 1207.62 mm

📅 Daily Rainfall Model Evaluation:
Training RMSE: 1.6260562930647602
Testing RMSE: 5.365483283735088
R² Score (Test): 0.05510636159303561
Enter a date (YYYY-MM-DD) to predict rainfall: 2025-05-21

📅 Predicted Rainfall on 2025-05-21: ~13.37 mm
Enter a year to predict daily rainfall for: 2024

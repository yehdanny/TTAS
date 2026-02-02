# TTAS Triage Prediction

This project implements a machine learning pipeline to predict triage levels (1-5) based on emergency department data.

## Project Structure
*   `code/explore_data.py`: Initial data exploration.
*   `code/clean_data.py`: Basic cleaning (type conversion).
*   `code/analyze_triage_features.py`: Feature analysis.
*   `code/preprocess.py`: Main preprocessing (validates vitals, handles nulls).
*   `code/utilities/split_data.py`: Splits data into Train/Val.
*   `code/utilities/ensure_nan.py`: Explicitly marks NaNs for CSV.
*   `code/model/train_model.py`: Trains multiple models (HistGradientBoosting, XGBoost, LightGBM) with class imbalance handling.

## Usage

1.  **Preprocessing**:
    ```bash
    python code/preprocess.py
    ```
2.  **Split Data**:
    ```bash
    python code/utilities/split_data.py
    ```
3.  **Ensure NaN Format**:
    ```bash
    python code/utilities/ensure_nan.py
    ```
4.  **Train Models**:
    ```bash
    python code/model/train_model.py
    ```
    *   Trains models using **Weighted Cross-Entropy Loss** to handle class imbalance.
    *   Outputs results to `code/model/<ModelName>/`.
    *   Summary: `code/model/all_models_evaluation.txt`.

## Model Performance (Balanced)
Validation Accuracy comparison after applying class weights:

| Model | Accuracy | Notes |
| :--- | :--- | :--- |
| **XGBoost** | **0.7830** | Best overall accuracy |
| HistGradientBoosting | 0.7789 | Good baselines |
| LightGBM | 0.7727 | |

*Note: While accuracy is lower than the unbalanced version (~0.84), reliability for minority classes (Level 1, 5) has significantly improved.*

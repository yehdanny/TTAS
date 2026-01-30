# Taiwan Triage and Acuity Scale (TTAS) Data Project

This project focuses on preprocessing and analyzing emergency department data to predict the **Triage Level (檢傷分級)**.

## Project Structure

```
TTAS/
├── code/
│   ├── explore_data.py          # Initial data exploration
│   ├── analyze_triage_features.py # Feature quality & correlation analysis
│   ├── preprocess.py            # Main preprocessing script (No Imputation)
│   ├── feature_analysis_report.txt # Output of analysis
│   └── README.md                # This file
└── data/
    └── 0128/
        ├── data.csv             # Raw Input
        └── processed_data.csv   # Cleaned Output (Generated)
```

## Feature Engineering & Preprocessing

The `preprocess.py` script transforms the raw CSV into a machine-learning ready format `processed_data.csv`.

### Key Principles
1.  **No Imputation**: As requested, missing values (NaN) are preserved to allow models (like XGBoost) to learn from the missingness pattern itself.
2.  **Leakage Prevention**: Post-triage administrative columns (e.g., `檢傷名稱中文名`, `預設檢傷級數`) are removed.

### Transformations
-   **Vitals (`體溫`, `血壓`, etc.)**: Converted to numeric. Errors coerced to NaN.
-   **Age**: Calculated from `急診日期` and `生日`.
-   **GCS Total**: Sum of E, V, M. If any component is missing, Total is NaN.
-   **Shock Index**: Calculated as `脈搏 / 收縮壓`.
-   **Complaint Length**: Character count of `病人主訴`.
-   **Missing Flags**: Boolean columns added for key vitals (e.g., `收縮壓_Missing`).

### Utilities

1.  **Split Data (`utilities/split_data.py`)**:
    *   Splits `processed_data.csv` into training (80%) and validation (20%) sets.
    *   Output: `train_data.csv`, `val_data.csv`.

2.  **Ensure NaN (`utilities/ensure_nan.py`)**:
    *   Reads `train_data.csv` and `val_data.csv`.
    *   Explicitly converts missing values to the string "nan".
    *   Output: `train_data_nan.csv`, `val_data_nan.csv`.

## Usage

1.  **Run Preprocessing**:
    ```bash
    python code/preprocess.py
    ```
    Output: `data/0128/processed_data.csv`.

2.  **Run Split**:
    ```bash
    python code/utilities/split_data.py
    ```
    Output: `data/0128/train_data.csv`, `data/0128/val_data.csv`.

3.  **Run NaN Conversion**:
    ```bash
    python code/utilities/ensure_nan.py
    ```
    Output: `data/0128/train_data_nan.csv`, `data/0128/val_data_nan.csv`.

4.  **Run Analysis** (Optional):
    ```bash
    python code/analyze_triage_features.py
    ```
    Generates `feature_analysis_report.txt`.

5.  **Run Training**:
    ```bash
    python code/model/train_model.py
    ```
    *   Trains **HistGradientBoosting**, **XGBoost**, and **LightGBM**.
    *   Output: Organized folders in `code/model/` and aggregated summary `all_models_evaluation.txt`.

## Model Performance (Validation Set)
| Model Name | Accuracy |
| :--- | :--- |
| **HistGradientBoosting** | **0.8405** |
| LightGBM | 0.8374 |
| XGBoost | 0.8358 |

*Note: All models natively support missing values (NaN).*

## Requirements
-   Python 3.x
-   pandas
-   numpy

import pandas as pd
import numpy as np
import os
import sys


def preprocess_data(file_path, output_path):
    print(f"Starting preprocessing for {file_path}...")

    # 1. Load Data with Encoding Handling
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(file_path, encoding="big5")
        except:
            df = pd.read_csv(file_path, encoding="gbk")

    print(f"Original shape: {df.shape}")

    # 2. Type Coercion (Vital Signs) - Coerce to NaN if invalid
    numeric_cols = ["體溫", "體重", "收縮壓", "舒張壓", "脈搏", "呼吸", "SAO2", "身高"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. GCS Handling
    gcs_cols = ["GCS_E", "GCS_V", "GCS_M"]
    for col in gcs_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Date Parsing
    df["急診日期"] = pd.to_datetime(
        df["急診日期"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["生日"] = pd.to_datetime(
        df["生日"].astype(str), format="%Y%m%d", errors="coerce"
    )

    # 5. Feature Engineering
    print("Generating features...")

    # Age
    # Calculate age properly using year differences to be safe or (date - date) / 365.25
    df["Age"] = (df["急診日期"] - df["生日"]).dt.days / 365.25
    # Fix negative ages or unlikely ages if any (simple sanity check, but strict no imputation means we just leave it unless it's clearly data error to be dropped?)
    # For now, we keep calculated age. One might want to clip < 0 to 0.
    df.loc[df["Age"] < 0, "Age"] = np.nan
    df.loc[df["體重"] <= 0, "體重"] = np.nan
    df.loc[df["身高"] <= 0, "身高"] = np.nan
    df.loc[df["瞳孔左"] == " ", "瞳孔左"] = np.nan
    df.loc[df["瞳孔右"] == " ", "瞳孔右"] = np.nan

    # GCS Total
    # Only if all 3 are present? Or sum available? Standard is sum. If any missing, usually GCS is invalid.
    # We will enforce GCS_Total as NaN if any component is NaN
    df["GCS_Total"] = df["GCS_E"] + df["GCS_V"] + df["GCS_M"]

    # Shock Index (HR / SBP)
    # Handle division by zero or NaN
    df["Shock_Index"] = df["脈搏"] / df["收縮壓"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Complaint Length
    df["Complaint_Length"] = df["病人主訴"].astype(str).str.len()

    # 6. Flag Missing Vitals (Optional but helpful for tree models)
    # The user said "No imputation", but flags are new features, not imputation.
    # Let's add simple flags for key vitals.
    for col in ["收縮壓", "脈搏", "SAO2", "體重"]:
        df[f"{col}_Missing"] = df[col].isnull().astype(int)

    # 7. Drop Leakage / Unused Columns
    # Leakage: 檢傷名稱, 檢傷分類 (These are what we might want to predict or are derived from the result)
    # Unused: Names, IDs (unless ID is needed for join, but for modeling it's useless)
    drop_cols = [
        "檢傷大分類英文名",
        "檢傷大分類中文名",
        "檢傷名稱英文名",
        "檢傷名稱中文名",
        "檢傷主訴",  # Often synonymous with chief complaint but triager entered
        "預設檢傷級數",  # This sounds like a system suggestion, might be leakage
        "急診日期",
        "生日",  # Used for Age
        "病人主訴",  # Processed into length/NLP (raw text usually dropped for tabular model unless vectorizing)
        "藥物過敏史",  # Complex text, maybe drop for now or just keep? User didn't specify. dropping for MVP clean dataset.
    ]

    # Also drop original string versions of Vitals if they are replaced?
    # We replaced them in-place in step 2.

    # Keep '檢傷分級' as Target

    final_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 8. Save
    print(f"processed shape: {final_df.shape}")
    final_df.to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )  # sig for Excel compatibility in Taiwan
    print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming code hierarchy
    data_dir = os.path.join(os.path.dirname(script_dir), "data", "0128")

    input_csv = os.path.join(data_dir, "data.csv")
    output_csv = os.path.join(data_dir, "processed_data.csv")

    if not os.path.exists(data_dir):
        # fallback if running with different structure
        data_dir = os.path.join("data", "0128")
        input_csv = "data/0128/data.csv"
        output_csv = "data/0128/processed_data.csv"

    preprocess_data(input_csv, output_csv)

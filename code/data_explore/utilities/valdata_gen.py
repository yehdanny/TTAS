"""
valdata_gen.py

用途:
- 用來生成驗證資料
- 每個檢傷分級取50筆

輸入:
- data.csv

輸出:
- val_data.csv

"""

import pandas as pd
import os


def generate_val_data():
    # Define paths
    base_dir = r"c:\Users\ygz08\Work\TTAS"
    input_csv = os.path.join(base_dir, "code", "data_explore", "test_file", "data.csv")
    output_val_csv = os.path.join(
        base_dir, "code", "data_explore", "test_file", "val_data.csv"
    )

    try:
        # Check if input file exists
        if not os.path.exists(input_csv):
            print(f"Error: Input file not found at {input_csv}")
            return

        # Read CSV
        print(f"Reading from {input_csv}...")
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    target_col = "檢傷分級"
    if target_col not in df.columns:
        print(f"Error: Column '{target_col}' not found in CSV.")
        print("Available columns:", df.columns.tolist())
        return

    # Clean the column: force numeric, handle errors
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Filter for valid triage levels (1-5)
    valid_levels = [1, 2, 3, 4, 5]
    val_data_frames = []

    for level in valid_levels:
        subset = df[df[target_col] == level]
        count = len(subset)
        print(f"Found {count} records for level {level}")

        if count >= 50:
            sample = subset.sample(n=50, random_state=42)
        else:
            print(f"Warning: Only {count} records found for level {level}, taking all.")
            sample = subset

        val_data_frames.append(sample)

    if val_data_frames:
        result_df = pd.concat(val_data_frames)
        # Shuffle the result
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

        try:
            result_df.to_csv(output_val_csv, index=False)
            print(f"Successfully saved validation data to {output_val_csv}")
            print("\nDistribution of Triage Levels in Validation Set:")
            print(result_df[target_col].value_counts().sort_index())
        except Exception as e:
            print(f"Error saving CSV: {e}")
    else:
        print("No data found for specified triage levels.")


if __name__ == "__main__":
    generate_val_data()

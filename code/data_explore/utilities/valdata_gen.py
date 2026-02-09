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
import logging
import os

logger = logging.getLogger(__name__)


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
            logger.error(f"Error: Input file not found at {input_csv}")
            return

        # Read CSV
        logger.info(f"Reading from {input_csv}...")
        df = pd.read_csv(input_csv)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return

    target_col = "檢傷分級"
    if target_col not in df.columns:
        logger.error(f"Error: Column '{target_col}' not found in CSV.")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return

    # Clean the column: force numeric, handle errors
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Filter for valid triage levels (1-5)
    valid_levels = [1, 2, 3, 4, 5]
    val_data_frames = []

    for level in valid_levels:
        subset = df[df[target_col] == level]
        count = len(subset)
        logger.info(f"Found {count} records for level {level}")

        if count >= 50:
            sample = subset.sample(n=50, random_state=42)
        else:
            logger.warning(
                f"Warning: Only {count} records found for level {level}, taking all."
            )
            sample = subset

        val_data_frames.append(sample)

    if val_data_frames:
        result_df = pd.concat(val_data_frames)
        # Shuffle the result
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

        try:
            result_df.to_csv(output_val_csv, index=False)
            logger.info(f"Successfully saved validation data to {output_val_csv}")
            logger.info("\nDistribution of Triage Levels in Validation Set:")
            logger.info(result_df[target_col].value_counts().sort_index())
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
    else:
        logger.warning("No data found for specified triage levels.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        encoding="utf-8",
        format="%(levelname)s: %(message)s",
    )
    generate_val_data()

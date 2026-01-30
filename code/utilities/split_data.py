import pandas as pd
import numpy as np
import os
import sys


def split_dataset(input_path, output_dir, train_ratio=0.8, seed=42):
    print(f"Reading {input_path}...")
    try:
        df = pd.read_csv(input_path, encoding="utf-8-sig")  # Reading the processed file
    except:
        df = pd.read_csv(input_path)

    print(f"Total records: {len(df)}")

    # Random split
    # Using numpy for reproducibility or pandas sample
    train_df = df.sample(frac=train_ratio, random_state=seed)
    val_df = df.drop(train_df.index)

    print(f"Training set: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"Validation set: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")

    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_path = os.path.join(output_dir, "train_data.csv")
    val_path = os.path.join(output_dir, "val_data.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")

    print(f"Saved to:\n  {train_path}\n  {val_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script is in code/utilities/
    # data is in ../../data/0128/processed_data.csv

    project_root = os.path.dirname(os.path.dirname(script_dir))  # Work/TTAS
    data_dir = os.path.join(project_root, "data", "0128")
    input_file = os.path.join(data_dir, "processed_data_nan.csv")

    if not os.path.exists(input_file):
        # Fallback for running from root
        input_file = "data/0128/processed_data_nan.csv"
        data_dir = "data/0128"

    split_dataset(input_file, data_dir)

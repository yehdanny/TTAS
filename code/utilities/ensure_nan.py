import pandas as pd
import os
import sys


def ensure_nan_string(input_path, output_path):
    print(f"Processing {input_path}...")
    try:
        # Read with default NaN handling first
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    except:
        df = pd.read_csv(input_path)

    # Fill NaN with string 'nan'
    # Check current dtypes first. If we fillna with string, numeric cols become object.
    # The user asked to "write empty values as nan", which usually implies string representation in CSV
    # or ensuring the CSV output literally has 'nan' where missing.
    # Pandas to_csv by default writes empty string for NaN if na_rep is not specified (or default).
    # Actually default to_csv writes empty string for NaN? No, default is empty string.
    # Wait, pandas to_csv default na_rep is empty string.
    # If user wants "nan", we should use na_rep='nan' in to_csv without changing DataFrame types if possible,
    # OR change DataFrame values to string 'nan'.
    # Using na_rep='nan' in to_csv is cleaner and keeps types in valid state until save.

    print(f"Saving to {output_path} with na_rep='nan'...")
    df.to_csv(output_path, index=False, encoding="utf-8-sig", na_rep="nan")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data", "0128")

    # Files to process
    files = ["train_data.csv", "val_data.csv"]

    for f in files:
        input_f = os.path.join(data_dir, f)
        output_f = os.path.join(data_dir, f.replace(".csv", "_nan.csv"))

        if os.path.exists(input_f):
            ensure_nan_string(input_f, output_f)
        else:
            print(f"Warning: {input_f} not found.")
            # Fallback for simple path
            input_simple = f"data/0128/{f}"
            if os.path.exists(input_simple):
                output_simple = f"data/0128/{f.replace('.csv', '_nan.csv')}"
                ensure_nan_string(input_simple, output_simple)

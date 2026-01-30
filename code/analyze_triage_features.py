import pandas as pd
import numpy as np
import os
import sys


def analyze_features(file_path, output_file):
    print(f"Analyzing {file_path} for Triage Prediction...")

    # 1. Load Data (Handle Encoding from previous exploration)
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(file_path, encoding="big5")
        except:
            df = pd.read_csv(file_path, encoding="gbk")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Triage Feature Analysis Report\n")
        f.write("==============================\n\n")

        # 2. Preprocessing Simulation (for analysis only)
        # Convert numeric columns
        numeric_cols = [
            "體溫",
            "體重",
            "收縮壓",
            "舒張壓",
            "脈搏",
            "呼吸",
            "SAO2",
            "身高",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Date conversion
        df["急診日期"] = pd.to_datetime(
            df["急診日期"].astype(str), format="%Y%m%d", errors="coerce"
        )
        df["生日"] = pd.to_datetime(
            df["生日"].astype(str), format="%Y%m%d", errors="coerce"
        )

        # Derive Age
        now = pd.Timestamp.now()
        # Use simple year difference or accurate calculation
        df["Age"] = (df["急診日期"] - df["生日"]).dt.days / 365.25

        # GCS Total
        # Extract numeric part from GCS string (assuming it might be mixed)
        # The exploration showed GCS_E values like '4', 'nan'.
        gcs_cols = ["GCS_E", "GCS_V", "GCS_M"]
        for col in gcs_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["GCS_Total"] = df["GCS_E"] + df["GCS_V"] + df["GCS_M"]

        # Target
        target = "檢傷分級"

        # 3. Missing Value Analysis (Crucial: No Imputation)
        f.write("1. Missing Value Analysis (Constraint: No Imputation)\n")
        f.write("-" * 50 + "\n")
        missing = df.isnull().mean() * 100
        f.write(missing[missing > 0].sort_values(ascending=False).to_string())
        f.write(
            "\n\nNOTE: High missingness in vitals implies specific triage context (e.g. minor injury vs critical).\n\n"
        )

        # 4. Feature Correlations with Target
        f.write("2. Numeric Feature Correlation with Target (Pearson)\n")
        f.write("-" * 50 + "\n")

        analysis_cols = numeric_cols + ["Age", "GCS_Total", "GCS_E", "GCS_V", "GCS_M"]
        corr = df[analysis_cols + [target]].corr()[target].sort_values(ascending=False)
        f.write(corr.to_string())
        f.write("\n\n")

        # 5. Categorical/Text Analysis
        f.write("3. Categorical/Text Feature Insights\n")
        f.write("-" * 50 + "\n")

        # Check '病人主訴' (Chief Complaint) length vs Triage
        df["Complaint_Length"] = df["病人主訴"].astype(str).apply(len)
        corr_len = df[["Complaint_Length", target]].corr().iloc[0, 1]
        f.write(
            f"Correlation between Complaint Length and Triage Level: {corr_len:.4f}\n"
        )

        # Check Leakage Candidates
        leakage_candidates = ["檢傷名稱中文名", "檢傷大分類中文名"]
        for col in leakage_candidates:
            if col in df.columns:
                f.write(f"\ncardinality of {col}: {df[col].nunique()}\n")
                # Check if 1-to-1 mapping with target
                crosstab = pd.crosstab(df[col], df[target])
                # If for a given category, there is only 1 target value, it's strong predictor/leakage
                max_confidence = (crosstab.max(axis=1) / crosstab.sum(axis=1)).mean()
                f.write(
                    f"Average predictability of Triage Level given {col}: {max_confidence:.4f} (1.0 = perfect leakage)\n"
                )

    print(f"Analysis complete. Report saved to {output_file}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming data path relative to script based on previous knowledge
    data_path = os.path.join(os.path.dirname(script_dir), "data", "0128", "data.csv")
    output_path = os.path.join(script_dir, "feature_analysis_report.txt")

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}, trying simple path...")
        data_path = "data/0128/data.csv"

    analyze_features(data_path, output_path)

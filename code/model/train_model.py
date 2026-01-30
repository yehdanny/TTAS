import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import xgboost as xgb
import lightgbm as lgb


def get_categorical_indices(df):
    return [
        i
        for i, col in enumerate(df.columns)
        if df[col].dtype == "object" or df[col].dtype.name == "category"
    ]


def train_hist_gradient_boosting(
    X_train, y_train, X_val, y_val, output_dir, cat_cols_idx=None
):
    # Sklearn HistGradientBoosting handles NaNs natively.
    # It needs encoded categories (done in main prep) or strict numerical input.
    # Assuming X_train is already Ordinal Encoded for categorical columns.

    print("Training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=100,
        categorical_features=cat_cols_idx if cat_cols_idx else None,
    )
    clf.fit(X_train, y_train)
    return clf


def train_xgboost(X_train, y_train, X_val, y_val, output_dir):
    print("Training XGBoost...")
    # XGBoost handles NaNs.
    # Needs classes to be 0-indexed for multi-class classification.
    # Our Triage Level is 1-5. We need to map to 0-4.

    y_train_mapped = y_train - 1
    y_val_mapped = y_val - 1

    # XGBoost supports categorical data natively with 'enable_categorical=True' in recent versions (since 1.5/1.6+).
    # But usually creating DMatrix with enable_categorical is safer or just use scikit-learn API with enable_categorical=True.
    # We will use the Sklearn API for consistency.

    clf = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=5,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        enable_categorical=True,
        missing=np.nan,  # Explicitly tell it NaN is missing
    )

    # For XGBoost with enable_categorical, we need pandas dataframe with category dtype
    # We will cast object columns to category in the main loop before passing here.

    clf.fit(X_train, y_train_mapped, eval_set=[(X_val, y_val_mapped)], verbose=False)
    return clf


def train_lightgbm(X_train, y_train, X_val, y_val, output_dir):
    print("Training LightGBM...")
    # LightGBM handles NaNs. Supports 'category' dtype.

    clf = lgb.LGBMClassifier(
        random_state=42, objective="multiclass", num_class=5, verbose=-1
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=-1)],
    )  # turn off verbose logging
    return clf


def evaluate_and_save(model, model_name, X_val, y_val, output_root):
    model_dir = os.path.join(output_root, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"Evaluating {model_name}...")

    # XGBoost prediction fix (if model predicts 0-4, we need to map back to 1-5)
    y_pred = model.predict(X_val)

    if model_name == "XGBoost":
        # Check if predictions are 0-4
        if min(y_pred) == 0 and max(y_pred) <= 4:
            y_pred = y_pred + 1

    # Metrics
    acc = accuracy_score(y_val, y_pred)

    # Detailed Metrics
    # Micro (Global)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_val, y_pred, average="micro"
    )
    # Macro (Unweighted mean of per-class)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro"
    )
    # Weighted (Weighted by support)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_val, y_pred, average="weighted"
    )

    report = classification_report(y_val, y_pred, output_dict=False)
    cm = confusion_matrix(y_val, y_pred)

    # Save Report
    report_path = os.path.join(model_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")

        f.write("Global Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'Metric':<15} | {'Micro':<10} | {'Macro':<10} | {'Weighted':<10}\n")
        f.write(
            f"{'Precision':<15} | {p_micro:.4f}     | {p_macro:.4f}     | {p_weighted:.4f}\n"
        )
        f.write(
            f"{'Recall':<15} | {r_micro:.4f}     | {r_macro:.4f}     | {r_weighted:.4f}\n"
        )
        f.write(
            f"{'F1-Score':<15} | {f1_micro:.4f}     | {f1_macro:.4f}     | {f1_weighted:.4f}\n"
        )
        f.write("\n")

        f.write("Full Classification Report:\n")
        f.write("-" * 30 + "\n")
        f.write(report)
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write("-" * 30 + "\n")
        f.write(np.array2string(cm))
        f.write("\n")

    # Save Model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return acc, report


def train_all_models(train_path, val_path, output_dir):
    print("Loading data...")
    train_df = pd.read_csv(train_path, na_values=["nan"])
    val_df = pd.read_csv(val_path, na_values=["nan"])

    target_col = "檢傷分級"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]  # 1,2,3,4,5

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # Identify Categoricals
    # For Tree models (modern ones), it's often best to convert Object -> Category dtype
    # XGBoost and LightGBM handle 'category' dtype natively if specified.
    # HistGradientBoosting needs Ordinal Encoding or explicitly known categorical indices.

    # Let's standardize:
    # 1. Convert objects to 'category' dtype for XGB/LGBM
    # 2. For HistGradientBoosting, we will use Ordinal Encoding backup or just use the same dataframe if it supports 'category' (it nominally needs numerical encoding in older sklearn, but recent versions might take it. Best to Ordinal Encode for Sklearn just to be safe and robust).

    # Actually, to keep it simple and shared:
    # We will Ordinal Encode categorical columns for ALL models.
    # Why? All 3 support numerical input for categories. XGB/LGBM treat int as continuous unless specified,
    # BUT for high cardinalities, native category support is better.
    # However, '性別' (Sex) is low cardinality.
    # Let's use OrdinalEncoder for robust handling of 'nan' categories encoded as distinct integer or passed as NaN.

    print("Preprocessing...")

    # Detect object cols
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # For all models, let's just Ordinal Encode. It's the most stable common denominator.
    # We preserve NaN as NaN (OrdinalEncoder with handle_unknown or encoded as specific value).
    # Since these trees handle NaNs, we want to Keep NaN in X.
    # OrdinalEncoder(encoded_missing_value=NaN) is available in newer Sklearn.

    if cat_cols:
        # Standardize empty strings before encoding if any left
        # (Though ensure_nan.py dealt with it? No, that was for output csv.)
        # We just encode.
        # Using OrdinalEncoder to turn strings into 0,1,2... preserving NaNs?
        # Standard OrdinalEncoder doesn't preserve NaNs easily until recently.
        # Simple approach: Pandas 'category' codes.

        for col in cat_cols:
            # Train
            X_train[col] = X_train[col].astype("category")
            # Val (using train categories)
            X_val[col] = pd.Categorical(
                X_val[col], categories=X_train[col].cat.categories
            )

            # Convert to codes, but keep NaNs as NaNs
            # .cat.codes returns -1 for NaN. We want to optionally keep it as NaN or treat -1 as category.
            # Histogram-based methods usually handle NaNs fine. If we pass -1, it treats it as a category value -1.
            # This is actually fine and good for missing data handling in categorical features.

            X_train[col] = X_train[col].cat.codes
            X_val[col] = X_val[col].cat.codes

            # Now replace -1 with NaN if we really want native NaN handling?
            # Or just let -1 be a category "Missing".
            # HistGradientBoosting treats NaN as missing. It maps valid values to bins.
            # If we leave it as -1 (int), it's just a number.
            # Let's convert -1 back to NaN to be "pure" about "Native NaN support".

            X_train.loc[X_train[col] == -1, col] = np.nan
            X_val.loc[X_val[col] == -1, col] = np.nan

    # Categorical indices for Sklearn (indices of columns that were categorical)
    cat_cols_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    models = [
        ("HistGradientBoosting", train_hist_gradient_boosting),
        ("XGBoost", train_xgboost),
        ("LightGBM", train_lightgbm),
    ]

    results = []

    for name, train_func in models:
        try:
            print(f"--- Processing {name} ---")
            if name == "HistGradientBoosting":
                clf = train_func(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    output_dir,
                    cat_cols_idx=cat_cols_idx,
                )
            else:
                clf = train_func(X_train, y_train, X_val, y_val, output_dir)

            acc, report = evaluate_and_save(clf, name, X_val, y_val, output_dir)
            results.append((name, acc))
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            import traceback

            traceback.print_exc()

    # Aggregate Report
    summary_path = os.path.join(output_dir, "all_models_evaluation.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("All Models Evaluation Summary\n")
        f.write("=============================\n\n")
        f.write(f"{'Model Name':<25} | {'Accuracy':<10}\n")
        f.write("-" * 40 + "\n")

        for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
            f.write(f"{name:<25} | {acc:.4f}\n")

    print(f"\nTraining Complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data", "0128")

    train_path = os.path.join(data_dir, "train_data_nan.csv")
    val_path = os.path.join(data_dir, "val_data_nan.csv")

    if not os.path.exists(train_path):
        train_path = "data/0128/train_data_nan.csv"
        val_path = "data/0128/val_data_nan.csv"

    output_dir = script_dir  # Save in code/model/

    train_all_models(train_path, val_path, output_dir)

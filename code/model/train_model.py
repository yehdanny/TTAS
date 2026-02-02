import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer


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

    print("Training HistGradientBoostingClassifier (Balanced)...")
    clf = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=100,
        categorical_features=cat_cols_idx if cat_cols_idx else None,
        class_weight="balanced",  # Handle imbalance
    )
    clf.fit(X_train, y_train)
    return clf


def train_xgboost(X_train, y_train, X_val, y_val, output_dir):
    print("Training XGBoost (Balanced)...")
    # XGBoost handles NaNs.
    # Needs classes to be 0-indexed for multi-class classification.
    # Our Triage Level is 1-5. We need to map to 0-4.

    y_train_mapped = y_train - 1
    y_val_mapped = y_val - 1

    # Calculate sample weights for balanced training
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_mapped)

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

    clf.fit(
        X_train,
        y_train_mapped,
        sample_weight=sample_weights,  # Apply weights
        eval_set=[(X_val, y_val_mapped)],
        verbose=False,
    )
    return clf


def train_lightgbm(X_train, y_train, X_val, y_val, output_dir):
    print("Training LightGBM (Balanced)...")
    # LightGBM handles NaNs. Supports 'category' dtype.

    clf = lgb.LGBMClassifier(
        random_state=42,
        objective="multiclass",
        num_class=5,
        verbose=-1,
        class_weight="balanced",  # Handle imbalance
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=-1)],
    )  # turn off verbose logging
    return clf


def train_catboost(X_train, y_train, X_val, y_val, output_dir, cat_cols_idx=None):
    print("Training CatBoost (Balanced)...")
    # CatBoost handles NaNs and Categories natively

    # FOR CATBOOST: It prefers Strings or Integers for Categories.
    # X_train here contains NaNs in categorical columns (from other processing).
    # We should handle this. Natively CatBoost handles NaN in numeric, but for Categorical features it needs robust types.
    # We will let CatBoost detect features.

    # IMPORTANT: We need to make sure we don't pass float-like categoricals if we specify cat_features.
    # Since we are using a unified X_train which (in this script) has cat.codes (floats due to NaN),
    # we'll let CatBoost handle it as numerical features mostly, OR we fix it.

    # Strategy: Treat everything as numerical for CatBoost too (using the codes), BUT enable auto class weights.
    # If we want to use cat_features, we'd need to cast them to int and fillna.
    # Let's try treating them as numerical (since they are ordinal encoded) for simplicity in this shared pipeline. The codes preserve order.
    # This avoids the "Float vs String" error for cat_features.

    clf = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function="MultiClass",
        auto_class_weights="Balanced",  # Handle Imbalance
        nan_mode="Min",  # Handle NaNs
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    # Pass cat_features indices?
    # If we pass integers, we should tell CatBoost they are categorical if we want it to treat them as such.
    # However, for simplicity and consistency with other models consuming the same X_train (integers),
    # let's try letting CatBoost handle them or specify them.
    # Specifying cat_features usually gives better quality.

    clf.fit(
        X_train,
        y_train,
        # cat_features=cat_cols_idx,  # <-- Commenting out to treat as numeric (Ordinal) to avoid type issues
        eval_set=(X_val, y_val),
        verbose=False,
    )
    return clf


def evaluate_and_save(model, model_name, X_val, y_val, output_root=None):
    # If model is None (just ensemble predictions), predictions must be passed differently?
    # No, we assume 'model' implements predict().

    if output_root:
        model_dir = os.path.join(output_root, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    print(f"Evaluating {model_name}...")

    # Ensemble might be a wrapper or list of models.
    # If it's the Voting Ensemble, we handled it separately? No, let's treat it as a model object.

    y_pred = model.predict(X_val)

    # XGBoost prediction fix (if model predicts 0-4, we need to map back to 1-5)
    # But ONLY if it's the XGBClassifier directly.
    # VotingEnsemble using predict() might need care.
    # Actually, for simplified Ensemble, we can wrap the prediction logic.

    if hasattr(model, "feature_importances_") and isinstance(model, xgb.XGBClassifier):
        # Check if predictions are 0-4 (only if pure xgb)
        if np.min(y_pred) == 0 and np.max(y_pred) <= 4:
            y_pred = y_pred + 1

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_val, y_pred, average="micro"
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro"
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_val, y_pred, average="weighted"
    )

    report = classification_report(y_val, y_pred, output_dict=False)
    cm = confusion_matrix(y_val, y_pred)

    if output_root:
        # Save Report
        report_path = os.path.join(model_dir, "evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name} (Balanced)\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")

            f.write("Global Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(
                f"{'Metric':<15} | {'Micro':<10} | {'Macro':<10} | {'Weighted':<10}\n"
            )
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

        # Save Model (pickle)
        try:
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Could not save model pickle for {model_name}: {e}")

    return {
        "name": model_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "recall_macro": r_macro,
        "precision_macro": p_macro,
    }


class SoftVotingEnsemble:
    def __init__(self, estimators):
        self.estimators = estimators  # list of (name, model)

    def predict(self, X):
        # Average predict_proba
        # Weights? Equal weights for now.
        probas = []
        for name, clf in self.estimators:
            p = clf.predict_proba(X)

            # Align classes
            # XGBoost (0-4) -> classes are 0,1,2,3,4.
            # Others (1-5) -> classes are 1,2,3,4,5.
            # Proba matrix is (n_samples, 5).
            # Index 0 is Class 1 (or 0).
            # So they ARE aligned by index.

            probas.append(p)

        avg_proba = np.mean(probas, axis=0)
        # Argmax gives 0-4 index. Map to 1-5.
        preds = np.argmax(avg_proba, axis=1) + 1
        return preds

    def predict_proba(self, X):
        # Just return average
        probas = []
        for name, clf in self.estimators:
            probas.append(clf.predict_proba(X))
        return np.mean(probas, axis=0)


def train_all_models(train_path, val_path, output_dir):
    print("Loading data...")
    train_df = pd.read_csv(train_path, na_values=["nan"])
    val_df = pd.read_csv(val_path, na_values=["nan"])

    target_col = "檢傷分級"
    text_col = "病人主訴"

    # Preprocessing
    print("Preprocessing...")

    # 1. TF-IDF
    tfidf_dim = 100
    if text_col in train_df.columns:
        print(f"Generating TF-IDF features (Top {tfidf_dim})...")
        tfidf = TfidfVectorizer(max_features=tfidf_dim)

        # Fill NaN text
        train_text = train_df[text_col].fillna("")
        val_text = val_df[text_col].fillna("")

        # Fit Transform
        # We need dense arrays for most tree models (or dataframe columns)
        train_tfidf = tfidf.fit_transform(train_text).toarray()
        val_tfidf = tfidf.transform(val_text).toarray()

        # Create DataFrames
        tfidf_cols = [f"TFIDF_{i}" for i in range(train_tfidf.shape[1])]
        train_tfidf_df = pd.DataFrame(
            train_tfidf, columns=tfidf_cols, index=train_df.index
        )
        val_tfidf_df = pd.DataFrame(val_tfidf, columns=tfidf_cols, index=val_df.index)

        # Drop text col
        train_df = train_df.drop(columns=[text_col])
        val_df = val_df.drop(columns=[text_col])

        # Concat
        train_df = pd.concat([train_df, train_tfidf_df], axis=1)
        val_df = pd.concat([val_df, val_tfidf_df], axis=1)

    # Prepare X and y
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # Categoricals
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    if cat_cols:
        for col in cat_cols:
            # Train
            X_train[col] = X_train[col].astype("category")
            # Val (using train categories)
            X_val[col] = pd.Categorical(
                X_val[col], categories=X_train[col].cat.categories
            )

            # Convert to codes, but keep NaNs as NaNs
            X_train[col] = X_train[col].cat.codes
            X_val[col] = X_val[col].cat.codes

            # Restore NaNs
            X_train.loc[X_train[col] == -1, col] = np.nan
            X_val.loc[X_val[col] == -1, col] = np.nan

    cat_cols_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    models = [
        ("HistGradientBoosting", train_hist_gradient_boosting),
        ("XGBoost", train_xgboost),
        ("LightGBM", train_lightgbm),
        ("CatBoost", train_catboost),
    ]

    trained_models_list = []
    results_metrics = []

    # Train Individual Models
    for name, train_func in models:
        try:
            print(f"--- Processing {name} ---")
            if name == "HistGradientBoosting" or name == "CatBoost":
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

            # Save classifier for ensemble
            trained_models_list.append((name, clf))

            # Evaluate
            metrics = evaluate_and_save(clf, name, X_val, y_val, output_dir)
            results_metrics.append(metrics)

        except Exception as e:
            print(f"Failed to train {name}: {e}")
            import traceback

            traceback.print_exc()

    # Ensemble (Voting)
    try:
        print("--- Processing VotingEnsemble (Soft) ---")
        ensemble = SoftVotingEnsemble(trained_models_list)
        # Evaluate Ensemble
        metrics = evaluate_and_save(
            ensemble, "VotingEnsemble", X_val, y_val, output_dir
        )
        results_metrics.append(metrics)
    except Exception as e:
        print(f"Failed to create Ensemble: {e}")
        import traceback

        traceback.print_exc()

    # Aggregate Report
    summary_path = os.path.join(output_dir, "all_models_evaluation.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("All Models Evaluation Summary (Balanced + TFIDF)\n")
        f.write("==============================================\n\n")

        # Header
        headers = [
            "Model Name",
            "Accuracy",
            "F1 (Macro)",
            "Recall (Macro)",
            "Precision (Macro)",
        ]
        header_line = f"{headers[0]:<25} | {headers[1]:<10} | {headers[2]:<12} | {headers[3]:<12} | {headers[4]:<12}"
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")

        for m in sorted(results_metrics, key=lambda x: x["accuracy"], reverse=True):
            line = f"{m['name']:<25} | {m['accuracy']:.4f}     | {m['f1_macro']:.4f}       | {m['recall_macro']:.4f}       | {m['precision_macro']:.4f}"
            f.write(line + "\n")

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

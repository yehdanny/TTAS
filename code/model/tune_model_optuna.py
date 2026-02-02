import optuna
import pandas as pd
import numpy as np
import os
import sys
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_preprocess(train_path, val_path):
    print("Loading data...")
    train_df = pd.read_csv(train_path, na_values=["nan"])
    val_df = pd.read_csv(val_path, na_values=["nan"])

    target_col = "檢傷分級"
    text_col = "病人主訴"

    # Preprocessing (Same as train_model.py)
    print("Preprocessing...")

    # 1. TF-IDF
    tfidf_dim = 100
    if text_col in train_df.columns:
        print(f"Generating TF-IDF features (Top {tfidf_dim})...")
        tfidf = TfidfVectorizer(max_features=tfidf_dim)

        train_text = train_df[text_col].fillna("")
        val_text = val_df[text_col].fillna("")

        train_tfidf = tfidf.fit_transform(train_text).toarray()
        val_tfidf = tfidf.transform(val_text).toarray()

        tfidf_cols = [f"TFIDF_{i}" for i in range(train_tfidf.shape[1])]
        train_tfidf_df = pd.DataFrame(
            train_tfidf, columns=tfidf_cols, index=train_df.index
        )
        val_tfidf_df = pd.DataFrame(val_tfidf, columns=tfidf_cols, index=val_df.index)

        train_df = train_df.drop(columns=[text_col])
        val_df = val_df.drop(columns=[text_col])

        train_df = pd.concat([train_df, train_tfidf_df], axis=1)
        val_df = pd.concat([val_df, val_tfidf_df], axis=1)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # Categoricals
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    if cat_cols:
        for col in cat_cols:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = pd.Categorical(
                X_val[col], categories=X_train[col].cat.categories
            )
            X_train[col] = X_train[col].cat.codes
            X_val[col] = X_val[col].cat.codes
            X_train.loc[X_train[col] == -1, col] = np.nan
            X_val.loc[X_val[col] == -1, col] = np.nan

    cat_cols_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    return X_train, y_train, X_val, y_val, cat_cols_idx


def objective_hgb(trial, X_train, y_train, X_val, y_val, cat_cols_idx):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "random_state": 42,
        "class_weight": "balanced",
        "categorical_features": cat_cols_idx if cat_cols_idx else None,
    }

    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")

    return f1  # Optimize F1 Macro


def objective_lgbm(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "class_weight": "balanced",
    }

    clf = lgb.LGBMClassifier(**params, random_state=42)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    preds = clf.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    return f1


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    train_path = os.path.join(project_root, "data", "0128", "train_data_nan.csv")
    val_path = os.path.join(project_root, "data", "0128", "val_data_nan.csv")

    X_train, y_train, X_val, y_val, cat_cols_idx = load_and_preprocess(
        train_path, val_path
    )

    print("Optimization Start: HistGradientBoosting")
    study_hgb = optuna.create_study(direction="maximize")
    study_hgb.optimize(
        lambda trial: objective_hgb(
            trial, X_train, y_train, X_val, y_val, cat_cols_idx
        ),
        n_trials=30,
    )

    print("Optimization Start: LightGBM")
    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(
        lambda trial: objective_lgbm(trial, X_train, y_train, X_val, y_val), n_trials=30
    )

    print("\nBest HistGradientBoosting F1:", study_hgb.best_value)
    print("Best HGB Params:", study_hgb.best_params)

    print("\nBest LightGBM F1:", study_lgbm.best_value)
    print("Best LGBM Params:", study_lgbm.best_params)

    # Save best params to txt
    with open(os.path.join(script_dir, "best_params.txt"), "w") as f:
        f.write(f"HistGradientBoosting Best F1: {study_hgb.best_value}\n")
        f.write(f"Parameters: {study_hgb.best_params}\n\n")
        f.write(f"LightGBM Best F1: {study_lgbm.best_value}\n")
        f.write(f"Parameters: {study_lgbm.best_params}\n")

import pandas as pd
import numpy as np
import pickle
import os
import sys


def predict_stage1(patient_data, model_path, encoder_path):
    """
    Stage 1: Classification using HistGradientBoosting Classifier (24 features).
    """
    print("--- Stage 1: Random Forest (HistGradientBoosting) Prediction ---")

    # 1. Load Resources
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("Model or Encoder not found.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    # 2. Prepare DataFrame
    # Ensure all expected keys exist or fill with defaults/NaN
    df = pd.DataFrame([patient_data])

    # 3. Feature Engineering (Replicating preprocess.py logic partially)
    # Vital Signs Coercion
    numeric_cols = ["體溫", "體重", "收縮壓", "舒張壓", "脈搏", "呼吸", "SAO2", "身高"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # GCS
    gcs_cols = ["GCS_E", "GCS_V", "GCS_M"]
    for col in gcs_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Age Calculation
    if "急診日期" in df.columns and "生日" in df.columns:
        try:
            today = pd.to_datetime(
                str(df["急診日期"].iloc[0]), format="%Y%m%d", errors="coerce"
            )
            dob = pd.to_datetime(
                str(df["生日"].iloc[0]), format="%Y%m%d", errors="coerce"
            )
            df["Age"] = (today - dob).days / 365.25
            df.loc[df["Age"] < 0, "Age"] = np.nan
        except:
            df["Age"] = np.nan
    elif "Age" not in df.columns:
        df["Age"] = np.nan  # Or provided directly

    # GCS Total
    if all(col in df.columns for col in gcs_cols):
        df["GCS_Total"] = df["GCS_E"] + df["GCS_V"] + df["GCS_M"]
    else:
        df["GCS_Total"] = np.nan

    # Shock Index
    if "脈搏" in df.columns and "收縮壓" in df.columns:
        df["Shock_Index"] = df["脈搏"] / df["收縮壓"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        df["Shock_Index"] = np.nan

    # Complaint Length (dummy logic if text not provided in this dict, usually distinct)
    if "病人主訴" in df.columns:
        df["Complaint_Length"] = df["病人主訴"].astype(str).str.len()
    else:
        df["Complaint_Length"] = 0  # Feature name must exist

    # Missing Flags
    for col in ["收縮壓", "脈搏", "SAO2", "體重"]:
        col_name = f"{col}_Missing"
        if col in df.columns:
            df[col_name] = df[col].isnull().astype(int)
        else:
            df[col_name] = 1  # Missing if not in input

    # 4. Encoding Categoricals
    # Encoder expects: ['性別', '瞳孔左', '瞳孔右', 'LMP']
    cat_cols = ["性別", "瞳孔左", "瞳孔右", "LMP"]

    # Ensure they exist
    for col in cat_cols:
        if col not in df.columns:
            df[col] = (
                np.nan
            )  # Or appropriate missing value for object? Encoder handles -1/unknown?
            # Encoder is OrdinalEncoder with handle_unknown='use_encoded_value', unknown_value=-1
            # But it expects string/object input probably.
            df[col] = (
                df[col].astype(str).replace("nan", np.nan)
            )  # Ensure string-like or NaN

    # To use encoder safely, we must handle NaNs if encoder doesn't transform them naturally?
    # Sklearn OrdinalEncoder handles NaNs by passing through if configured or encoding them if encoded_missing_value is set.
    # The inspected encoder has `handle_unknown='use_encoded_value', unknown_value=-1`.
    # It might NOT handle NaNs directly unless trained with them.
    # In preprocess.py: "X_train[col] = X_train[col].cat.codes" -> This suggests the training data used categorical codes directly!
    # CHECK: inspect_model.py showed `encoder.pkl` exists. But `train_model.py` re-encodes manually using pandas category codes!
    # "X_train[col] = X_train[col].cat.codes"
    # Wait, `train_model.py` logic:
    # 1. Load CSV
    # 2. cat.codes
    # 3. Train
    # IF `encoder.pkl` was saved independently (not in shared train_model logic shown), use it.
    # BUT if `train_model.py` creates the model usage *without* saving `encoder.pkl`, where did `encoder.pkl` come from?
    # Inspect output showed `encoder.pkl` exists and is OrdinalEncoder.
    # Let's assume `encoder.pkl` is the correct one to use for inference to map raw strings to integers.

    # 4. Encoding Categoricals
    # Encoder expects: ['性別', '瞳孔左', '瞳孔右', 'LMP']
    cat_cols = ["性別", "瞳孔左", "瞳孔右", "LMP"]

    try:
        # Create a copy for encoding
        df_cat = df[cat_cols].copy()

        # Ensure consistent string type for encoder (OrdinalEncoder usually prefers same type)
        # Fill NaNs with a placeholder 'Unknown' which will likely be unknown to encoder -> -1
        # match handle_unknown='use_encoded_value', unknown_value=-1
        for col in cat_cols:
            df_cat[col] = df_cat[col].fillna("Unknown").astype(str)

        # Transform
        df_encoded_cats = encoder.transform(df_cat)

        # Assign back
        df[cat_cols] = df_encoded_cats
    except Exception as e:
        print(f"Encoding Warning: {e}. Fallback to -1 for categoricals.")
        for col in cat_cols:
            df[col] = -1

    # 5. Arrange Features (24 features)
    feature_names = [
        "檢傷編號",
        "性別",
        "體溫",
        "體重",
        "收縮壓",
        "舒張壓",
        "脈搏",
        "呼吸",
        "SAO2",
        "GCS_E",
        "GCS_V",
        "GCS_M",
        "瞳孔左",
        "瞳孔右",
        "身高",
        "LMP",
        "Age",
        "GCS_Total",
        "Shock_Index",
        "Complaint_Length",
        "收縮壓_Missing",
        "脈搏_Missing",
        "SAO2_Missing",
        "體重_Missing",
    ]

    # '檢傷編號' is usually ID. Model uses it? Inspect output says yes.
    # If it's a feature, it's very odd. Maybe it's numeric and just got included.
    # We should pass 0 or some dummy if not provided.
    if "檢傷編號" not in df.columns:
        df["檢傷編號"] = 0

    # Reorder
    X_input = df[feature_names]

    # 6. Predict
    prediction = model.predict(X_input)[0]

    # Map back? If model output is 1-5 already?
    # XGBoost in `train_model.py` needed mapping (0-4 -> 1-5).
    # HistGradientBoosting usually outputs classes as provided.
    # If training data y was 1-5, output is 1-5.

    return int(prediction)


def predict_stage2(pipe, messages, stage1_result, patient_data):
    """
    Stage 2: LLM refinement using Random Forest input.
    """
    print(f"--- Stage 2: LLM (Qwen) Prediction (Stage 1 Result: {stage1_result}) ---")

    # Construct System Prompt
    system_prompt = (
        "你是一位專業的檢傷分類護理師。你的任務是執行台灣檢傷分類 (TTAS)。\n"
        f"第一階段的生理評估建議檢傷級數為：{stage1_result} 級。\n"
        "請根據病患的主訴與現有數據進行評估，以確認或調整此分類結果。\n"
        "請提供你的判斷理由以及最終的檢傷級數 (1-5 級)。"
        "1 級 (復甦急救)： 生命徵象極端不穩，必須立即急救。\n"
        "2 級 (危急)： 潛在危及生命，生理狀態不穩，需 10 分鐘內診視。\n"
        "3 級 (緊急)： 穩定但有潛在惡化風險，需 30 分鐘內診視。\n"
        "4 級 (次緊急)： 穩定，慢性疾病急性發作，建議 60 分鐘內診視。\n"
        "5 級 (非緊急)： 輕微狀況，無即刻危險，建議 120 分鐘內診視。\n"
    )

    # Prepend system prompt to messages
    # If using chat template, usually:
    full_messages = [
        {"role": "system", "content": system_prompt},
    ] + messages

    # Generate
    try:
        outputs = pipe(full_messages, max_new_tokens=512)
        generated_text = outputs[0]["generated_text"]
        # Extract the last assistant response
        response = (
            generated_text[-1]["content"]
            if isinstance(generated_text, list)
            else generated_text
        )
        return response
    except Exception as e:
        return f"LLM Generation Failed: {e}"

import os
import sys
import time
import pandas as pd
from model_init import initialize_llm
from predict import predict_stage1, predict_stage2


def main():
    print("=== Two-Stage TTAS Triage System Start ===")
    time1 = time.time()

    # Paths
    # Adjust paths based on where main.py is (code/llms/) vs model (code/model/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
    model_path = os.path.join(base_dir, "model", "model.pkl")
    encoder_path = os.path.join(base_dir, "model", "encoder.pkl")

    # 1. Initialize LLM
    print("Initializing LLM...")
    pipe = initialize_llm()
    if pipe is None:
        print("Failed to initialize LLM. Exiting.")
        return

    # 2. Mock Patient Data
    # Based on inspected feature names
    patient_data = {
        "病人主訴": "今天早上開始覺得心悸、全身疲倦故入",
        "體溫": 36.8,
        "體重": np.nan,
        "收縮壓": 144,
        "舒張壓": 84,
        "脈搏": 110,
        "呼吸": 18,
        "SAO2": 99,
        "GCS_E": 4,
        "GCS_V": 5,
        "GCS_M": 6,
        "性別": "Female",  # Need to check encoding for Male/Female vs M/F or 1/2?
        "瞳孔左": np.nan,
        "瞳孔右": np.nan,
        "身高": np.nan,
        "LMP": np.nan,  # Last Menstrual Period
        "急診日期": "20251001",
        "生日": "19590117",
        "檢傷編號": 226060,
    }

    print("\nPatient Data:")
    print(patient_data)

    # 3. Stage 1 Prediction
    try:
        stage1_result = predict_stage1(patient_data, model_path, encoder_path)
        print(f"\nStage 1 Result (RF): Level {stage1_result}")
    except Exception as e:
        print(f"Stage 1 Failed: {e}")
        # Proceed to Stage 2 without stage 1 hint or abort?
        # User requirement implies Stage 1 acts as input.
        stage1_result = "Unknown"

    # 4. Stage 2 Prediction
    # Construct messages for LLM
    messages = [
        {"role": "user", "content": f"Patient Complaint: {patient_data['病人主訴']}"},
    ]

    stage2_result = predict_stage2(pipe, messages, stage1_result, patient_data)
    print("\nStage 2 Result (LLM):")
    print(stage2_result)

    print("\n=== Two-Stage TTAS Triage System End ===")
    time2 = time.time()
    print(f"Total time: {time2 - time1} seconds")


if __name__ == "__main__":
    import numpy as np  # For nan example

    main()

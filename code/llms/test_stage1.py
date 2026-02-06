import os
import sys
import pandas as pd
import numpy as np
import pickle
from predict import predict_stage1


def test_stage1():
    print("Testing Stage 1 (RF) only...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
    model_path = os.path.join(base_dir, "model", "model.pkl")
    encoder_path = os.path.join(base_dir, "model", "encoder.pkl")

    patient_data = {
        "病人主訴": "Chest pain",
        "體溫": 37.5,
        "體重": 70,
        "收縮壓": 150,
        "舒張壓": 90,
        "脈搏": 100,
        "呼吸": 22,
        "SAO2": 95,
        "GCS_E": 4,
        "GCS_V": 5,
        "GCS_M": 6,
        "性別": "Male",
        "瞳孔左": 3.0,
        "瞳孔右": 3.0,
        "身高": 175,
        "LMP": np.nan,
        "急診日期": 20230101,
        "生日": 19700101,
        "檢傷編號": 123456,
    }

    try:
        result = predict_stage1(patient_data, model_path, encoder_path)
        print(f"Stage 1 Prediction: {result}")
        print("Success!")
    except Exception as e:
        print(f"Stage 1 Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_stage1()

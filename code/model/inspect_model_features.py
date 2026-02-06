import pickle
import os
import sys
import numpy as np


def inspect_file(path):
    print(f"--- Inspecting {path} ---")
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        print(f"Type: {type(obj)}")

        if hasattr(obj, "feature_names_in_"):
            print(f"Feature Names Count: {len(obj.feature_names_in_)}")
            print(f"Feature Names In: {list(obj.feature_names_in_)}")

    except Exception as e:
        print(f"Error loading: {e}")


if __name__ == "__main__":
    base_dir = r"c:\Users\ygz08\Work\TTAS\code\model"
    inspect_file(os.path.join(base_dir, "model.pkl"))

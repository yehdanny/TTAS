import pickle
import os
import sys


def inspect_file(path):
    print(f"--- Inspecting {path} ---")
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        print(f"Type: {type(obj)}")
        print(f"String representation: {obj}")

        if hasattr(obj, "classes_"):
            print(f"Classes: {obj.classes_}")

        if hasattr(obj, "feature_names_in_"):
            print(f"Feature Names In: {obj.feature_names_in_}")

        if hasattr(obj, "get_feature_names_out"):
            try:
                print(f"Feature Names Out: {obj.get_feature_names_out()}")
            except:
                pass

        if isinstance(obj, dict):
            print("Keys:", obj.keys())

    except Exception as e:
        print(f"Error loading: {e}")


if __name__ == "__main__":
    base_dir = r"c:\Users\ygz08\Work\TTAS\code\model"
    inspect_file(os.path.join(base_dir, "model.pkl"))
    inspect_file(os.path.join(base_dir, "encoder.pkl"))

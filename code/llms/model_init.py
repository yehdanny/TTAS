from transformers import pipeline
import torch


def initialize_llm(model_id="Qwen/Qwen3-4B-Instruct-2507"):
    """
    Initializes the text-generation pipeline with the specified model.
    """
    print(f"Loading model: {model_id}...")

    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    try:
        pipe = pipeline(
            "text-generation", model=model_id, device=device, trust_remote_code=True
        )
        print("Model loaded successfully.")
        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__ == "__main__":
    # Test initialization
    pipe = initialize_llm()
    if pipe:
        print("Test passed: Pipeline created.")

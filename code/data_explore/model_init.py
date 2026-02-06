import torch
import chromadb
from transformers import pipeline
from utilities.chunk_list import chunks_list
import os

os.environ["HF_HUB_OFFLINE"] = "1"


# 1. 初始化 Qwen3
def initialize_llm(model_id="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-generation",
        model=model_id,
        trust_remote_code=True,
        device=device,
    )
    return pipe


# 2. 準備與儲存 Chunks
def setup_vector_db(chunks):
    client = chromadb.Client()
    # 建立集合，這裡可以使用默認的 embedding function
    collection = client.create_collection(name="ttas_knowledge")

    ids = [f"id_{i}" for i in range(len(chunks))]
    documents = [c["content"] + "\n備註: " + c["remarks"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def model_init():
    try:  # initialize llm
        pipe = initialize_llm()
        collection = setup_vector_db(chunks_list)
        print("[info] Model initialized successfully")
    except Exception as e:
        print(f"[error] Model initialization failed: {e}")

    return pipe, collection


if __name__ == "__main__":
    pipe, collection = model_init()
    collection = setup_vector_db(chunks_list)

    # qwen 2507 fp16 = 14.4 + 133.1  seconds
    # qwen 2507 fp4 = 9.0 + 78.9  seconds

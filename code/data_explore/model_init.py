"""
model_init.py

用途:
- 初始化model
- 初始化vector database

輸入:
- chunks_list: list : 檢傷標準的list，每個元素都是一個dict，包含metadata、content、remarks。

輸出:
- pipe: model的物件，供後續進行LLM生成預測結果。
- collection: vector database的物件，供後續進行relevant計算。

"""

import torch
import chromadb
from llama_cpp import Llama
from utilities.chunks import chunks_list
import os
import logging

os.environ["HF_HUB_OFFLINE"] = "1"

logger = logging.getLogger(__name__)


def initialize_llm():
    # CPU / GPU
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"[Hardware] 偵測到 GPU: {gpu_name} ({vram_gb:.2f}GB VRAM)")

        n_gpu_layers = -1 if vram_gb > 4 else 20
    else:
        logger.info("[Hardware] 未偵測到 GPU，使用 CPU 執行")
        n_gpu_layers = 0

    pipe = Llama(
        model_path=r"C:\Users\ygz08\Work\TTAS\code\data_explore\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        n_gpu_layers=n_gpu_layers,
        n_ctx=4096,
        n_threads=os.cpu_count(),
        f16_kv=True,
        verbose=False,
    )
    return pipe


# 2. 準備與儲存 Chunks
def setup_vector_db(chunks):
    client = chromadb.Client()
    collection = client.create_collection(name="ttas_knowledge")

    ids = [f"id_{i}" for i in range(len(chunks))]
    # 用 query_text 做為 Embedding 的對象
    documents = [c["query_text"] for c in chunks]

    # 將真正的 content 和 remarks 存進 metadata，方便檢索到後直接取用
    metadatas = []
    for c in chunks:
        meta = c["metadata"].copy()
        meta["full_content"] = c["content"]
        meta["full_remarks"] = c["remarks"]
        metadatas.append(meta)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def model_init():
    try:  # initialize llm
        pipe = initialize_llm()
        collection = setup_vector_db(chunks_list)
        logger.info("[info] Model initialized successfully")
    except Exception as e:
        logger.error(f"[error] Model initialization failed: {e}")

    return pipe, collection


if __name__ == "__main__":
    pipe, collection = model_init()
    collection = setup_vector_db(chunks_list)

    # qwen 2507 fp16 = 14.4 + 133.1  seconds
    # qwen 2507 fp4 = 9.0 + 78.9  seconds

    # qwen 2507 fp4 w/AutoModelForCausalLM = 9.3 + 86.9 seconds

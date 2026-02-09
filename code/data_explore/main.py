"""
main.py

用途:
- 主要進入點，進行LLM+RAG的檢傷預測。

輸入:
- model_init(): 初始化model和vector database
- get_patient_info(): 取得病患基本資料、目標族群、主訴
- model_predict(): 進行LLM+RAG的檢傷預測
- save2json(): 將檢傷預測結果儲存到json檔案

輸出:
- pipe: model的物件，進行LLM生成預測結果。
- collection: vector database的物件，儲存著ttas的embeddings。
- patient_info: dict : 病人生理資料
- target_group: str : 成人/兒童
- complaint: str : 病人主訴
- final_decision: dict : LLM檢傷預測結果 (包含檢傷級數、原因)

"""

from utilities.get_patient_info import get_patient_info
from utilities.save2json import save2json
from model_predict import model_predict
from model_init import model_init
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# init
start_time = time.time()
pipe, collection = model_init()
end_time = time.time()
logging.info(f"[time] Model initialized in {end_time - start_time:.2f} seconds")


# get data
start_time2 = time.time()
patient_info, target_group, complaint = (
    get_patient_info()
)  # 病人基本資料 + 成人/兒童 + 病人主訴
end_time2 = time.time()
logging.info(f"[time] Data get in {end_time2 - start_time2:.2f} seconds")


# run llm
start_time3 = time.time()
final_decision = model_predict(pipe, patient_info, collection, target_group, complaint)
end_time3 = time.time()
logging.info(f"[time] Model predicted in {end_time3 - start_time3:.2f} seconds")

# save to json
save2json(final_decision)

logging.info(f"[time] Total time: {time.time() - start_time:.2f} seconds")

"""
_validation.py

用途:
- 驗證LLM+RAG的準確性

計算:
- correct_count: 正確的檢傷數量
- incorrect_count: 錯誤的檢傷數量
- even_less_count: 檢傷級數過低的數量 (例如: 預測為3級，但實際為4級)，**可能可接受**
- even_more_count: 檢傷級數過高的數量 (例如: 預測為3級，但實際為2級)，不可接受
- total_accuracy: 正確檢傷數量 / 總檢傷數量

輸出:
- correct_count: int
- incorrect_count: int
- even_less_count: int
- even_more_count: int
- total_accuracy: float

實體檔案:
- result_log: 紀錄結果
"""

from utilities.get_patient_info import get_patient_info
from model_predict import model_predict
from model_init import model_init
import pandas as pd
import logging
import json
import re

# init
pipe, collection = model_init()

logging.basicConfig(
    filename=r"C:\Users\ygz08\Work\TTAS\code\data_explore\test_file\result.log",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(message)s",
    force=True,
)
test_csv = pd.read_csv(r"C:\Users\ygz08\Work\TTAS\data\0128\data.csv")
correct_count = 0
incorrect_count = 0
even_less_count = 0
even_more_count = 0
for i in range(len(test_csv)):
    row = test_csv.iloc[i]  # 0 is the first row

    # get data
    patient_info, target_group, complaint, ground_truth = get_patient_info(
        test=True, row=row
    )  # 病人基本資料 + 成人/兒童 + 主訴

    # run llm
    final_decision, retrieved_docs = model_predict(
        pipe, patient_info, collection, target_group, complaint
    )

    if isinstance(final_decision, str):
        clean_json_str = re.search(r"\{.*\}", final_decision, re.DOTALL).group()
        final_decision_dict = json.loads(clean_json_str)

    if int(final_decision_dict["level"]) == int(ground_truth):
        correct_count += 1
    else:
        incorrect_count += 1
        logging.error(
            f"No.{i} Incorrect: {final_decision_dict['level']} != {ground_truth}\n"
            f"Patient Info: {patient_info}\n"
            f"Retrieved Docs: {retrieved_docs}\n"
            f"Final Decision: {final_decision}\n"
            f"--------------------------------------------------\n"
        )
    if int(final_decision_dict["level"]) < int(ground_truth):
        even_less_count += 1
    if int(final_decision_dict["level"]) > int(ground_truth):
        even_more_count += 1

logging.log(f"Correct: {correct_count}")
logging.log(f"Incorrect: {incorrect_count}")
logging.log(f"Even Less: {even_less_count}")
logging.log(f"Even More: {even_more_count}")
total_accuracy = correct_count / len(test_csv)
logging.log(f"Total Accuracy: {total_accuracy}")

"""
model_predict.py

用途:
- 根據病患基本資料、RAG檢索到的檢傷標準，生成LLM檢傷預測結果

輸入:
- pipe: model的物件，進行LLM生成預測結果。
- collection: vector database的物件，儲存著ttas的embeddings，用(collection.query())進行relevant計算。(**調整n_results回傳前n個**)
- patient_info: dict : 病人生理資料
- target_group: str : 成人/兒童
- complaint: str : 主訴

輸出:
- final_decision: dict : LLM檢傷預測結果
- retrieved_docs: list : RAG檢索到的前**n_results**個檢傷標準 (**驗證時**才需要該變數)

"""

import logging

logger = logging.getLogger(__name__)


def generate_ttas_response(pipe, patient_info, top_content, top_remarks):

    context = "\n\n".join(top_content)
    remarks = "\n\n".join(top_remarks)

    # 這裡加入 System Prompt，讓模型進入「專家模式」
    system_prompt = """你是一位精通台灣急診檢傷系統（TTAS）的專業護理師。你必須嚴格遵守檢傷準則，並僅以 JSON 格式輸出結果。"""

    user_content = f"""【檢傷參考準則】: 
    {context}

    【備註】: 
    {remarks}

    【病患資訊】:
    {patient_info}

    【判定規則】:
    1. 指標需符合檢傷標準，才能判斷"level"分級。
    2. 可調節分級項目，若病患資訊符合備註中的可調節分級項目，則可適當調整分級。
    3. 考慮備註特殊情況，若病患資訊缺乏過多，則預設為5級。
    4. 輸出JSON格式，level為整數(X ∈ [1, 2, 3, 4, 5])，reason為字串：
    {{
        "level": X,
        "reason": "(根據檢傷標準與病患資訊，簡短說明檢傷結果)"
    }}
    """

    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    result = pipe(
        full_prompt,
        max_tokens=1024,
        temperature=0.1,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )

    return result["choices"][0]["text"].strip()


def model_predict(pipe, patient_info, collection, target_group, complaint):
    try:  # RAG
        if target_group == "成人":
            search_labels = ["成人", "成人/兒童"]
        else:
            search_labels = ["兒童", "成人/兒童"]

        results = collection.query(
            query_texts=[complaint],
            n_results=3,
            where={"target": {"$in": search_labels}},
        )
        logger.info("[info] RAG successfully")
    except Exception as e:
        logger.error(f"[error] RAG failed: {e}")

    try:  # llm quest
        top_query_text = results["documents"][0][0]  # 計算relevant的大項目
        top_content = results["metadatas"][0][0]["full_content"]  # 詳細5分級內容
        top_remarks = results["metadatas"][0][0]["full_remarks"]  # 備註
        final_decision = generate_ttas_response(
            pipe, patient_info, top_content, top_remarks
        )
        logger.info("[info] LLM prediction successfully")
        return final_decision, top_content, top_remarks, top_query_text
    except Exception as e:
        logger.error(f"[error] LLM prediction failed: {e}")


if __name__ == "__main__":
    from utilities.get_patient_info import get_patient_info
    from model_init import model_init

    pipe, collection = model_init()
    patient_info, target_group, complaint = get_patient_info(test=True)
    final_decision = model_predict(
        pipe, patient_info, collection, target_group, complaint
    )
    print(final_decision)

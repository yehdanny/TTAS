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


def generate_ttas_response(pipe, patient_info, top_content, top_remarks, judgement):

    context = "\n\n".join(top_content)
    remarks = "\n\n".join(top_remarks)

    # 這裡加入 System Prompt，讓模型進入「專家模式」
    system_prompt = """你是一位精通台灣急診檢傷系統（TTAS）的專業護理師。你必須嚴格遵守檢傷準則，並僅以 JSON 格式輸出結果。
    - TTAS分級: 一級 (復甦急救)、二級 (危急)、三級 (緊急)、四級 (次緊急)、五級 (非緊急)。
    """

    user_content = f"""【檢傷參考準則】: 
    {context}

    【備註】: 
    {remarks}

    【病患資訊】:
    {patient_info}

    【護理師初步判斷】:
    {judgement}

    【判定規則】:
    1. 從【檢傷參考準則】中找到最適合的分級方法。
    2. 使用【病患資訊】與【護理師初步判斷】來判斷 "level"。
    3. 可調節分級項目，若病患資訊符合備註中的可調節分級項目，則可適當調整 "level"。
    4. 若病患資訊未知過多，則設為5級。
    5. 輸出JSON格式，level為整數(X ∈ [1, 2, 3, 4, 5])，reason為字串：
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


def model_predict(pipe, patient_info, collection, target_group, complaint, judgement):
    """
    :param: pipe: model的物件，進行LLM生成預測結果。
    :param: patient_info: dict : 病人生理資料
    :param: collection: vector database的物件，儲存著ttas的embeddings，用(collection.query())進行relevant計算。(**調整n_results回傳前n個**)
    :param: target_group: str : 成人/兒童
    :param: complaint: str : 病人主訴 / "無提供主訴"
    :param: judgement: str : "檢傷名稱中文名"護理師初步判斷
    """
    try:  # 先將病患主訴做精準處理
        complaint_prompt = f"""
        你是一位精通台灣急診檢傷系統（TTAS）的專業護理師。將【病患主訴】中病患來急診的病因列出來。

        【病患主訴】:
        {complaint}

        【輸出範例】:
        ["發燒", "咳嗽有痰", "頭痛"]
        """
        full_prompt = (
            f"<|im_start|>user\n{complaint_prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        result = pipe(
            full_prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
        complaint = "".join(result["choices"][0]["text"].strip())
        print(complaint)
    except Exception as e:
        logger.error(f"[error] Complaint processing failed: {e}")

    try:  # RAG
        if target_group == "成人":
            search_labels = ["成人", "成人/兒童"]
        else:
            search_labels = ["兒童", "成人/兒童"]

        results = collection.query(
            query_texts=[complaint],
            n_results=2,
            where={"target": {"$in": search_labels}},
        )
        logger.info("[info] RAG successfully")
    except Exception as e:
        logger.error(f"[error] RAG failed: {e}")

    try:  # llm quest
        # Extract top 3 results
        top_query_texts = []
        top_contents = []
        top_remarks = []

        for i in range(
            len(results["ids"][0])
        ):  # Iterate through all results returned (which is limited by n_results=3 in query)
            top_query_texts.append(results["documents"][0][i])
            top_contents.append(results["metadatas"][0][i]["full_content"])
            top_remarks.append(results["metadatas"][0][i]["full_remarks"])

        # edge case
        if complaint == "無提供主訴":
            top_contents = []
            top_remarks = []

        final_decision = generate_ttas_response(
            pipe, patient_info, top_contents, top_remarks, judgement
        )
        logger.info("[info] LLM prediction successfully")
        return final_decision, top_contents, top_remarks, top_query_texts
    except Exception as e:
        logger.error(f"[error] LLM prediction failed: {e}")


if __name__ == "__main__":
    from utilities.get_patient_info import get_patient_info
    from model_init import model_init
    import pandas as pd

    pipe, collection = model_init()
    # get data
    test_csv = pd.read_csv(
        r"C:\Users\ygz08\Work\TTAS\code\data_explore\test_file\val_data.csv"
    )
    row = test_csv.iloc[12]
    patient_info, target_group, complaint, ground_truth, judgement = get_patient_info(
        test=True, row=row
    )

    # run llm
    final_decision, top_content, top_remarks, top_query_text = model_predict(
        pipe, patient_info, collection, target_group, complaint, judgement
    )
    print(final_decision)

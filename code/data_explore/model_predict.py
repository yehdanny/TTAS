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


def generate_ttas_response(pipe, patient_info, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    # 這裡加入 System Prompt，讓模型進入「專家模式」
    system_prompt = "你是一位精通台灣急診檢傷系統（TTAS）的專業護理師。你必須嚴格遵守檢傷準則，並僅以 JSON 格式輸出結果。"

    user_content = f"""【檢傷參考準則】:
    {context}

    【病患資訊】:
    {patient_info}

    【判定規則】:
    1. 若有多項指標判定出不同級數，採計「級數最小」者。
    2. 考慮備註特殊情況。
    3. 輸出JSON格式：
    {{
        "level": X,
        "reason": "(請列出符合準則的具體項目)"
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
        results = collection.query(
            query_texts=[complaint], n_results=3, where={"target": target_group}
        )
        print("[info] RAG successfully")
    except Exception as e:
        print(f"[error] RAG failed: {e}")

    # try:  # llm quest
    final_decision = generate_ttas_response(pipe, patient_info, results["documents"][0])
    print("[info] LLM prediction successfully")
    return final_decision, results["documents"][0]
    # except Exception as e:
    #     print(f"[error] LLM prediction failed: {e}")


if __name__ == "__main__":
    from utilities.get_patient_info import get_patient_info
    from model_init import model_init

    pipe, collection = model_init()
    patient_info, target_group, complaint = get_patient_info(test=True)
    final_decision = model_predict(
        pipe, patient_info, collection, target_group, complaint
    )
    print(final_decision)

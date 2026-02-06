def generate_ttas_response(pipe, patient_info, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    # TTAS 核心邏輯：級數最小者優先 [cite: 3, 14, 28]
    prompt = f"""
    你是一位專業的急診檢傷護理師。請根據提供的【檢傷參考準則】與【病患資訊】，判定該病患的 TTAS 等級。
    
    【檢傷參考準則】:
    {context}
    
    【病患資訊】:
    {patient_info}
    
    【判定規則】:
    1. 若有多項指標（如呼吸、血壓、疼痛）判定出不同級數，請採計「最緊急（級數最小）」的結果。
    2. 必須考慮備註中的特殊情況（如 COPD 或 免疫缺陷）。
    3. 輸出JSON格式：
    {{
        "level": X,
        "reason": "(請列出符合準則的具體項目)"
    }}
    """

    messages = [{"role": "user", "content": prompt}]
    # 使用 Qwen3 進行生成
    gen_kwargs = {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1,
    }

    # 執行推論
    result = pipe(messages, **gen_kwargs)
    return result[0]["generated_text"][-1]["content"]


def model_predict(pipe, patient_info, collection, target_group, complaint):
    try:  # RAG
        results = collection.query(
            query_texts=[complaint], n_results=3, where={"target": target_group}
        )
        print("[info] RAG successfully")
    except Exception as e:
        print(f"[error] RAG failed: {e}")

    try:  # llm quest
        final_decision = generate_ttas_response(
            pipe, patient_info, results["documents"][0]
        )
        print("[info] LLM prediction successfully")
        return final_decision
    except Exception as e:
        print(f"[error] LLM prediction failed: {e}")


if __name__ == "__main__":
    from utilities.get_patient_info_csv import get_patient_info
    from model_init import model_init

    pipe, collection = model_init()
    patient_info, target_group, complaint = get_patient_info()
    final_decision = model_predict(
        pipe, patient_info, collection, target_group, complaint
    )
    print(final_decision)

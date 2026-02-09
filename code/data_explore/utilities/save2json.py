"""
save2json.py

用途:
- 將final_decision存成json檔

輸入:
- final_decision: dict or str

輸出:
- out.json

"""

import logging
import json
import re

logger = logging.getLogger(__name__)


def save2json(final_decision):

    try:  # check format
        if isinstance(final_decision, str):
            clean_json_str = re.search(r"\{.*\}", final_decision, re.DOTALL).group()
            final_decision_dict = json.loads(clean_json_str)
        else:
            final_decision_dict = final_decision
    except Exception as e:
        logger.error(f"解析失敗->防呆機制: {e}")
        final_decision_dict = {"raw_output": final_decision}

    # 2. 正確存檔
    output_path = r"C:\Users\ygz08\Work\TTAS\code\data_explore\test_file\out.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_decision_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    final_decision = '{\n    "level": 4,\n    "reason": "根據提供的檢傷參考準則，針對13歲兒童進行TTAS等級評估：\\n\\n1. 意識狀態：GCS = 15（E4, V5, M6），屬於正常範圍，且在14-15之間，依據備註說明，此情形應依主訴項目判斷，不直接歸類為一級或二級。\\n\\n2. 疼痛狀況：主訴為雙下肢疼痛，但未提供疼痛評估工具（如FLACC或臉譜量表）的量化結果。然而，生理數據顯示體溫37°C（正常）、心率100次/分（正常）、血壓與血氧均正常，無生命徵象異常，且無意識障礙或抽搐等症狀。\\n\\n3. 體溫：37°C，在正常範圍內（<38°C），無發燒，亦無低體溫，因此不符合二級或三級體溫指標。\\n\\n4. 所有生理指標均穩定，意識清楚，無急性惡化跡象，無抽搐、無呼吸困難、無休克征兆，也無明顯病容或危及生命的症狀。\\n\\n綜合分析：\\n- 意識狀態正常 → 不符合一級或二級標準。\\n- 疼痛雖存在，但未達重度（8-10分），且無其他嚴重併發症或生命危險因素。\\n- 體溫正常，無發燒或低體溫。\\n\\n根據「兒童疼痛檢傷標準」，疼痛若為中度（4-7分）屬三級，輕度（<4分）屬四級；但本案例缺乏疼痛量表評估，僅主訴疼痛，需結合臨床情境。\\n\\n由於患者為13歲，已超過3歲，疼痛若未明確達到中度以上，且無其他危急指標，應視為輕度或非緊急。\\n\\n最終判定：所有指標皆未觸發一級或二級，疼痛未達危急程度，生理穩定，意識清楚，無急性惡化或危機。\\n\\n因此，依照「最緊急（級數最小）」原則'
    save2json(final_decision)

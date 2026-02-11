"""
get_patient_info.py

用途:
- 取得病患基本資料
- 格式化病患基本資料

輸入:
get_patient_info() :
- test=True: 測試用，用csv讀取row當輸入
- test=False: 使用api 讀取json format 的病患基本資料 (正式pipeline輸入)

輸出:
- patient_text: str : 病人生理資料
- target_group: str : 成人/兒童
- complaint: str : 病人主訴
- ground_truth: int (only when test=True) : 檢傷分級 (**驗證時**才需要該變數)

"""

from datetime import datetime
import pandas as pd
import logging
import json

logger = logging.getLogger(__name__)


def clean_val(val, unit="", default="未知"):
    # 輔助函式：處理未知與單位
    s_val = str(val).strip().lower()
    # 判斷是否為空、空格、nan 或 0 (針對身高體重)
    if s_val in ["", "nan", "none", "0", "0.0"]:
        return default
    return f"{val}{unit}"


def format_patient_info_csv(row):
    # 日期與對象判定
    visit_date = datetime.strptime(str(row["急診日期"]), "%Y%m%d")
    birth_date = datetime.strptime(str(row["生日"]), "%Y%m%d")
    age = (
        visit_date.year
        - birth_date.year
        - ((visit_date.month, visit_date.day) < (birth_date.month, birth_date.day))
    )
    target_group = "兒童" if age < 18 else "成人"

    # 1. 基本資料處理
    complaint = clean_val(row["病人主訴"], default="無提供主訴")

    if row["性別"] == "M":
        gender = "男"
    elif row["性別"] == "F":
        gender = "女"
    else:
        gender = "無提供"

    # 2. 生理數據處理 (使用 clean_val 確保未知時不加單位)
    height = clean_val(row["身高"], unit="cm")
    weight = clean_val(row["體重"], unit="kg")
    temp = clean_val(row["體溫"], unit="°C")
    hr = clean_val(row["脈搏"], unit=" 次/分")
    rr = clean_val(row["呼吸"], unit=" 次/分")
    sao2 = clean_val(row["SAO2"], unit="%")

    # 3. 血壓與 MAP 邏輯
    sbp_raw = str(row["收縮壓"]).strip()
    dbp_raw = str(row["舒張壓"]).strip()

    if sbp_raw in ["", "nan", "0"] or dbp_raw in ["", "nan", "0"]:
        bp_str = "未知"
        map_val = "未知"
    else:
        try:
            sbp = float(sbp_raw)
            dbp = float(dbp_raw)
            bp_str = f"{int(sbp)}/{int(dbp)} mmHg"
            map_val = round((sbp - dbp) / 3 + dbp, 1)
        except:
            bp_str = "未知"
            map_val = "未知"

    # 4. GCS 處理
    try:
        e, v, m = int(row["GCS_E"]), int(row["GCS_V"]), int(row["GCS_M"])
        gcs_sum = f"{e + v + m} 分 (E{e}, V{v}, M{m})"
    except:
        gcs_sum = "未知"

    allergy = clean_val(row["藥物過敏史"], default="無")

    # 組合字串
    info_str = f"""
    【病患基本資料】
    - 對象：{target_group} (年齡: {age}歲, 性別: {gender})
    - 主訴：{complaint}

    【生理數據】
    - 身高：{height}
    - 體重：{weight}
    - 體溫：{temp}
    - 血壓：{bp_str} (MAP: {map_val})
    - 心跳/脈搏：{hr}
    - 呼吸：{rr}
    - 血氧 (SaO2)：{sao2}
    - 意識狀態 (GCS)：{gcs_sum}
    - 藥物過敏史：{allergy}
    """

    return (
        info_str,
        target_group,
        complaint,
        row["檢傷分級"],
        row["檢傷名稱中文名"],
    )


def format_patient_info_json(json_data):

    # 日期與年齡計算
    visit_date = datetime.strptime(str(json_data["急診日期"]), "%Y%m%d")
    birth_date = datetime.strptime(str(json_data["生日"]), "%Y%m%d")
    age = (
        visit_date.year
        - birth_date.year
        - ((visit_date.month, visit_date.day) < (birth_date.month, birth_date.day))
    )
    target_group = "兒童" if age < 18 else "成人"

    # 1. 基本資料處理
    complaint = clean_val(json_data["病人主訴"], default="無提供主訴")

    gender_map = {"M": "男", "F": "女"}
    gender = gender_map.get(str(json_data["性別"]).strip().upper(), "無提供")

    # 2. 生理數據處理 (未知時不加單位)
    height = clean_val(json_data["身高"], "cm", default="無提供")
    weight = clean_val(json_data["體重"], "kg", default="無提供")
    temp = clean_val(json_data["體溫"], "°C")
    hr = clean_val(json_data["脈搏"], " 次/分")
    rr = clean_val(json_data["呼吸"], " 次/分")
    sao2 = clean_val(json_data["SAO2"], "%")

    # 3. 血壓與 MAP 邏輯
    sbp_val = str(json_data["收縮壓"]).strip()
    dbp_val = str(json_data["舒張壓"]).strip()

    if sbp_val in ["", "nan", "0"] or dbp_val in ["", "nan", "0"]:
        bp_str = "未知"
        map_val = "未知"
    else:
        try:
            sbp = float(sbp_val)
            dbp = float(dbp_val)
            bp_str = f"{int(sbp)}/{int(dbp)} mmHg"
            map_val = round((sbp - dbp) / 3 + dbp, 1)
        except:
            bp_str = "未知"
            map_val = "未知"

    # 4. GCS 意識狀態
    try:
        e = int(json_data["GCS_E"])
        v = int(json_data["GCS_V"])
        m = int(json_data["GCS_M"])
        gcs_sum = f"{e + v + m} 分 (E{e}, V{v}, M{m})"
    except:
        gcs_sum = "未知"

    allergy = clean_val(json_data["藥物過敏史"], default="無")

    info_str = f"""
    【病患基本資料】
    - 對象：{target_group} (年齡: {age}歲, 性別: {gender})
    - 主訴：{complaint}

    【生理數據】
    - 身高：{height}
    - 體重：{weight}
    - 體溫：{temp}
    - 血壓：{bp_str} (MAP: {map_val})
    - 心跳/脈搏：{hr}
    - 呼吸：{rr}
    - 血氧 (SaO2)：{sao2}
    - 意識狀態 (GCS)：{gcs_sum}
    - 藥物過敏史：{allergy}
    """

    # 回傳時保持 json_data["病人主訴"] 原始值或處理過的值，視您後續 RAG 需求而定
    return info_str, target_group, complaint, json_data["檢傷名稱中文名"]


def get_patient_info(test=False, row=None):
    if test:
        patient_text, target, complaint, ground_truth, judgement = (
            format_patient_info_csv(row)
        )
        return patient_text, target, complaint, ground_truth, judgement
    else:
        try:  # get json
            json_data = get_json_api()
        except Exception as e:
            logger.error(f"[error] Get JSON failed: {e}")

        try:  # format patient info
            patient_text, target, complaint, judgement = format_patient_info_json(
                json_data
            )
        except Exception as e:
            logger.error(f"[error] Format patient info failed: {e}")

        logger.info("[info] Get patient info successfully")
        return patient_text, target, complaint, judgement


# TODO: 串api get json
def get_json_api():
    with open(
        r"C:\Users\ygz08\Work\TTAS\code\data_explore\test_file\test.json",
        "r",
        encoding="utf-8",
    ) as f:
        json_data = json.load(f)
    return json_data


if __name__ == "__main__":
    use_json = True
    use_csv = False

    if use_json:
        json_data = {
            "急診日期": "20251001",
            "生日": "20120329",
            "性別": "M",
            "病人主訴": "雙下肢疼痛、入急診就診未小便故入",
            "體溫": 37,
            "收縮壓": 110,
            "舒張壓": 70,
            "脈搏": 100,
            "呼吸": 18,
            "SAO2": 99,
            "GCS_E": 4,
            "GCS_V": 5,
            "GCS_M": 6,
            "藥物過敏史": "無",
        }
        patient_text, target, complaint = format_patient_info_json(json_data)
        print(patient_text)
    if use_csv:
        test_csv = pd.read_csv(r"C:\Users\ygz08\Work\TTAS\data\0128\data.csv")
        row = test_csv.iloc[0]
        patient_text, target, complaint = format_patient_info_csv(row)
        print(patient_text)

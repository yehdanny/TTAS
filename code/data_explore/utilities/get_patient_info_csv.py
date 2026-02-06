from datetime import datetime
import pandas as pd
import json


def format_patient_info_csv(row):
    visit_date = datetime.strptime(str(row["急診日期"]), "%Y%m%d")
    birth_date = datetime.strptime(str(row["生日"]), "%Y%m%d")
    age = (
        visit_date.year
        - birth_date.year
        - ((visit_date.month, visit_date.day) < (birth_date.month, birth_date.day))
    )
    target_group = "兒童" if age < 18 else "成人"

    try:
        sbp = float(row["收縮壓"])
        dbp = float(row["舒張壓"])
        map_val = round((sbp - dbp) / 3 + dbp, 1)
    except:
        map_val = "未知"

    try:
        gcs_sum = int(row["GCS_E"]) + int(row["GCS_V"]) + int(row["GCS_M"])
    except:
        gcs_sum = "未知"

    info_str = f"""
    【病患基本資料】
    - 對象：{target_group} (年齡: {age}歲, 性別: {row["性別"]})
    - 主訴：{row["病人主訴"]}


    【生理數據 (Vital Signs)】
    - 體溫：{row["體溫"]}°C
    - 血壓：{row["收縮壓"]}/{row["舒張壓"]} mmHg (MAP: {map_val})
    - 心跳/脈搏：{row["脈搏"]} 次/分
    - 呼吸：{row["呼吸"]} 次/分
    - 血氧 (SaO2)：{row["SAO2"]}%
    - 意識狀態 (GCS)：{gcs_sum} 分 (E{row["GCS_E"]}, V{row["GCS_V"]}, M{row["GCS_M"]})
    - 藥物過敏史：{row["藥物過敏史"]}
    """
    return info_str, target_group, row["病人主訴"]


def format_patient_info_json(json_data):
    visit_date = datetime.strptime(str(json_data["急診日期"]), "%Y%m%d")
    birth_date = datetime.strptime(str(json_data["生日"]), "%Y%m%d")
    age = (
        visit_date.year
        - birth_date.year
        - ((visit_date.month, visit_date.day) < (birth_date.month, birth_date.day))
    )
    target_group = "兒童" if age < 18 else "成人"

    try:
        sbp = float(json_data["收縮壓"])
        dbp = float(json_data["舒張壓"])
        map_val = round((sbp - dbp) / 3 + dbp, 1)
    except:
        map_val = "未知"

    try:
        gcs_sum = (
            int(json_data["GCS_E"]) + int(json_data["GCS_V"]) + int(json_data["GCS_M"])
        )
    except:
        gcs_sum = "未知"

    info_str = f"""
    【病患基本資料】
    - 對象：{target_group} (年齡: {age}歲, 性別: {json_data["性別"]})
    - 主訴：{json_data["病人主訴"]}


    【生理數據 (Vital Signs)】
    - 體溫：{json_data["體溫"]}°C
    - 血壓：{json_data["收縮壓"]}/{json_data["舒張壓"]} mmHg (MAP: {map_val})
    - 心跳/脈搏：{json_data["脈搏"]} 次/分
    - 呼吸：{json_data["呼吸"]} 次/分
    - 血氧 (SaO2)：{json_data["SAO2"]}%  
    - 意識狀態 (GCS)：{gcs_sum} 分 (E{json_data["GCS_E"]}, V{json_data["GCS_V"]}, M{json_data["GCS_M"]})
    - 藥物過敏史：{json_data["藥物過敏史"]}
    """
    return info_str, target_group, json_data["病人主訴"]


def get_patient_info(test=False):
    if test:
        test_csv = pd.read_csv(r"C:\Users\ygz08\Work\TTAS\data\0128\data.csv")
        row = test_csv.iloc[0]  # 0 is the first row
        patient_text, target, complaint = format_patient_info_csv(row)
        return patient_text, target, complaint
    else:
        try:  # get json
            json_data = get_json_api()
        except Exception as e:
            print(f"[error] Get JSON failed: {e}")

        try:  # format patient info
            patient_text, target, complaint = format_patient_info_json(json_data)
        except Exception as e:
            print(f"[error] Format patient info failed: {e}")

        print("[info] Get patient info successfully")
        return patient_text, target, complaint


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

from utilities.get_patient_info import get_patient_info
from model_predict import model_predict
from model_init import model_init

# init
pipe, collection = model_init()

# get data
patient_info, target_group, complaint = (
    get_patient_info()
)  # 病人基本資料 + 成人/兒童 + 主訴

# run llm
final_decision = model_predict(pipe, patient_info, collection, target_group, complaint)

print(final_decision)

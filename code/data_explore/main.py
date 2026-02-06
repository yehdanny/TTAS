from utilities.get_patient_info_csv import get_patient_info
from utilities.save2json import save2json
from model_predict import model_predict
from model_init import model_init
import time


# init
start_time = time.time()
pipe, collection = model_init()
end_time = time.time()
print(f"[time] Model initialized in {end_time - start_time:.2f} seconds")


# get data
start_time2 = time.time()
patient_info, target_group, complaint = (
    get_patient_info()
)  # 病人基本資料 + 成人/兒童 + 主訴
end_time2 = time.time()
print(f"[time] Data get in {end_time2 - start_time2:.2f} seconds")


# run llm
start_time3 = time.time()
final_decision = model_predict(pipe, patient_info, collection, target_group, complaint)
end_time3 = time.time()
print(f"[time] Model predicted in {end_time3 - start_time3:.2f} seconds")

# save to json
save2json(final_decision)

print(f"[time] Total time: {time.time() - start_time:.2f} seconds")

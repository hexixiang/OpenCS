# 将data添加到instruction_id中
import json

index_list = {}

exp_type = "alpaca"   # alpaca, dolly, wizard

if exp_type == "alpaca":
    file_path_1 = r"data\alpaca\processed_tag_data\alpaca_tag_total_with_index_good.jsonl"
    file_path_2 = r"src\tag_based_cluster\exp_tmp\instruction_id_alpaca.json"
    file_path_3 = r"src\tag_based_cluster\exp_tmp\instruction_id_alpaca_updated.json"
elif exp_type == "dolly":
    file_path_1 = r"data\dolly_15K\processed_tag_data\dolly_15k_tag_total_with_index_good.jsonl"
    file_path_2 = r"src\tag_based_cluster\exp_tmp\instruction_id_dolly.json"
    file_path_3 = r"src\tag_based_cluster\exp_tmp\instruction_id_dolly_updated.json"
elif exp_type == "wizard":
    file_path_1 = r"data\Wizard_70K\processed_tag_data\wizardlm-70k_tag_total_with_index_good.jsonl"
    file_path_2 = r"src\tag_based_cluster\exp_tmp\instruction_id_wizard.json"
    file_path_3 = r"src\tag_based_cluster\exp_tmp\instruction_id_wizard_updated.json"
else:
    print("Error! No such exp_type")
    raise TypeError
    

with open(file_path_1,"r",encoding="utf-8") as f:
    for line in f:
        line_dict = json.loads(line)
        index = line_dict["index"]
        index_list[index] = line_dict["data"]

with open(file_path_2, "r", encoding="utf-8") as f:
    # 创建一个新的文件来存储更新后的数据
    with open(file_path_3, "w", encoding="utf-8") as fw:
        for line in f:
            line_dict = json.loads(line)[0]
            index = line_dict["index"]
            # 添加 data 字段
            line_dict["data"] = index_list[index]
            # 将更新后的字典转换为 JSON 字符串
            updated_line = json.dumps(line_dict, ensure_ascii=False)
            # 写入新的文件
            fw.write(updated_line + "\n")

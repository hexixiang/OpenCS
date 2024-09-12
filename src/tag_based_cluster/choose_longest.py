# 对于labels_tuple相同的做聚集，然后挑选instruction最长的一条
import json

file_path = r"src\tag_based_cluster\exp_tmp\instruction_id_alpaca_updated.json"

labels_tuple_dict = {}

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        labels_list = sorted(data["labels"])
        labels_tuple = tuple(labels_list)
        
        response = data["data"]["output"]
        response_length = len(response)
        if labels_tuple not in labels_tuple_dict:
            labels_tuple_dict.update({labels_tuple:(response_length,data)})
        else:
            val = labels_tuple_dict[labels_tuple]
            if response_length > val[0]:
                del labels_tuple_dict[labels_tuple]
                labels_tuple_dict.update({labels_tuple:(response_length,data)})

with open(r"src\tag_based_cluster\exp_tmp\instruction_id_alpaca_longest.json", "w", encoding="utf-8") as f:
    for key,value in labels_tuple_dict.items():
        line_dict = value[1]
        updated_line = json.dumps(line_dict, ensure_ascii=False)
        # 写入新的文件
        f.write(updated_line + "\n")

        
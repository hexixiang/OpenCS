# 根据instruction_id生成映射表
import json

label_tag_dict = {}
tag_list = []
index_list = {}

with open(r"src\tag_based_cluster\exp_tmp\instruction_id_alpaca_longest.json","r",encoding="utf-8") as f:
    for line in f:
        line_dict = json.loads(line)
        index = line_dict["index"]
        index_list[index] = line_dict

once = 0
def print_once(value):
    global once
    if once ==0:
        print(value)
        once += 1
    else:
        pass

with open(r"src\tag_based_cluster\exp_tmp\instruction_id_alpaca_longest.json","r",encoding="utf-8") as f:
    for line in f:
        line_dict = json.loads(line)
        index = line_dict["index"]
        # labels_list = line_dict["labels"]
        # tags = [i["tag"] for i in line_dict["tags"]]
        tags = [i["tag_orgin"] for i in line_dict["tags"]]
        instruction = index_list[index]
        print_once(instruction)
        labels_list = [i["label"] for i in line_dict["tags"]]
        for tag in tags:
            if tag not in tag_list:
                tag_list.append(tag)
        for label,tag in zip(labels_list,tags):
            if label not in label_tag_dict:
                label_tag_dict[label] = [instruction]
            else:
                # 这一行代码很迷惑，感觉完全没必要额外做判断
                if instruction not in label_tag_dict[label]:
                    label_tag_dict[label].append(instruction)

# with open(r"src\tag_based_cluster\exp_tmp\instruction_id_alpaca.json","r",encoding="utf-8") as f:
#     for line in f:
#         line_dict = json.loads(line)[0]
#         index = line_dict["index"]
#         # labels_list = line_dict["labels"]
#         # tags = [i["tag"] for i in line_dict["tags"]]
#         tags = [i["tag_orgin"] for i in line_dict["tags"]]
#         instruction = index_list[index]
#         print_once(instruction)
#         labels_list = [i["label"] for i in line_dict["tags"]]
#         for tag in tags:
#             if tag not in tag_list:
#                 tag_list.append(tag)
#         for label,tag in zip(labels_list,tags):
#             if label not in label_tag_dict:
#                 label_tag_dict[label] = [instruction]
#             else:
#                 # 这一行代码很迷惑，感觉完全没必要额外做判断
#                 if instruction not in label_tag_dict[label]:
#                     label_tag_dict[label].append(instruction)

with open(r"src\tag_based_cluster\exp_tmp\alpaca_longest_5085.json","w",encoding="utf-8") as f:
    json.dump(label_tag_dict,f,indent=4,ensure_ascii=False)










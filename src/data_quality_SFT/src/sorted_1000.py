import json
import re

file_path = r"E:\shixi\论文\data\wizardlm-70k\score\MY\wizardlm_myprompt_score.json"
out_path = r"E:\shixi\论文\data\wizardlm-70k\score\MY\result_myprompt_1000.json"
index = []
datas = []
result = {}
num_0 = 0
num_101 = 0
with open(file_path,'r',encoding = 'utf-8') as f:
    for line in f:
        data = json.loads(line)
        datas.append(data)
for data in datas:
    key = data["key"]
    if data["key"] not in index:
        index.append(data["key"])
        result[key] = []
    context = data["context"]
    data1 = context["data1"]
    data2 = context["data2"]
    resu = data["result"]
    scores_1 = re.findall(r'\[Response 1\] Score: (\d+)', resu)
    scores_2 = re.findall(r'\[Response 2\] Score: (\d+)', resu)

# 将提取的分数转换为整数
    if len(scores_1) < 1 :
        scores_1.append(0)
    if len(scores_2) < 1 :
        scores_2.append(0)
        num_0 += 1
    response_scores_1 = [int(score) for score in scores_1]
    response_scores_2 = [int(score) for score in scores_2]

    mid1 = {}
    mid1["key"] = key
    mid1["data"] = data1
    mid1["score"] = response_scores_1[0]
    mid2 = {}
    mid2["key"] = key
    mid2["data"] = data2
    mid2["score"] = response_scores_2[0]
    result[key].append(mid1)
    result[key].append(mid2)
result_new = []
for key,value in result.items():
    item = {}
    sorted_value = sorted(value, key=lambda x: x['score'], reverse=True)
    length_of_sorted_value  = len(sorted_value)     ###py_prompt打分范围是0-100，这里按照类的数据数量排序
    item["label"] = key
    item["length_56"] = length_of_sorted_value 
    item["datas"] = sorted_value
    """
    len = 0                         ###what的打分范围是0-6，这里按照5和6的数量排序
    for i in value:
        if i["score"] == 5 or i["score"] == 6:
            len += 1
    item["label"] = key
    item["length_56"] = len
    item["datas"] = sorted_value
    """


    result_new.append(item)

sorted_result = sorted(result_new, key=lambda x: x['length_56'])
data_list = []
with open(out_path,'w',encoding = 'utf-8') as f:
    for item in sorted_result:
        i = 0
        values = item["datas"]
        for value in values:
            if (i < 20) and value["data"] not in data_list and value["score"] <= 100:
                data_list.append(value["data"])
                i += 1
                f.write(json.dumps(value) + '\n')




    

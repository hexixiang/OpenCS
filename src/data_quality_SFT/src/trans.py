import json

file_path = r"/home/lc/fuwuqi/wizardlm-evol-instruction-70k/wizardlm_score/WHAT/result_what-1000.json"
out_path = r"/home/lc/fuwuqi/wizardlm-evol-instruction-70k/wizardlm_score/my_prompt/result_what_1000_trans.json"
data_list = []
with open(file_path,'r',encoding = 'utf-8') as f:
    for line in f:
        tmp = {}
        data = json.loads(line)
        tmp = data["data"]
        tmp["instruction"] = data["data"]["instruction"]
        tmp["input"] = data["data"]["input"]
        tmp["output"] = data["data"]["output"]
        data_list.append(tmp)

with open(out_path,'w', encoding = 'utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

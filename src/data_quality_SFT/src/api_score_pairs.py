import json
import random
import os
from api_client import llm_openai
from tqdm import tqdm
import numpy as np
"""

"""

system_alpagasus = "You are a helpful and precise assistant for checking the quality of the answer."
def api_result(prompts, out_file, client):
    batch_generate_and_save(prompts, out_file, client)

def simple_double(dataset1, output_path, client):
    #提示词读取
    prompt_dou = open(r"/home/lc/fuwuqi/prompt/pair_alpagasus.txt", 'r', encoding='utf-8').read()
    # prompt_fs_dou = open(r"E:\shixi\论文\prompt\dafen\double_few-shot.txt", 'r', encoding='utf-8').read()

    output_fs_dou = []
    output_dou_trans = []
    result_dict = {}         #调用api
    #读取josn文件
    with open(dataset1,'r', encoding = 'utf-8') as f1:
        datas = json.load(f1)
        for (key,value) in datas.items():
            # print(key)
            pairs = []
            if len(value)%2==1:
                value = value[:-1]
            for i in range(0,len(value),2):
                mid = {}
                data1 = value[i]
                data2 = value[i+1]
                mid["data1"] = data1["data"]
                mid["data1"]["index"] = data1["index"]
                mid["data2"] = data2["data"]
                mid["data2"]["index"] = data2["index"]
                pro_dou = prompt_dou
                # instruction_dou = samples[i]["instruction"] + samples[i]["input"]
                response1 = data1["data"]["output"]
                response2 = data2["data"]["output"]
                # pro_dou = pro_dou.replace("<Instruction>", str(instruction_dou))
                """
                pro_dou = pro_dou.replace("<Instruction 1>", str(data1["data"]["instruction"] + data1["data"]["input"]))
                pro_dou = pro_dou.replace("<Instruction 2>", str(data2["data"]["instruction"] + data2["data"]["input"]))    
                pro_dou = pro_dou.replace("<Response 1>", str(response1))
                pro_dou = pro_dou.replace("<Response 2>", str(response2))
                """

                pro_dou = pro_dou.replace("[Question 1]", str(data1["data"]["instruction"] + data1["data"]["input"]))
                pro_dou = pro_dou.replace("[Question 2]", str(data2["data"]["instruction"] + data2["data"]["input"]))    
                pro_dou = pro_dou.replace("{answer_1}", str(response1))
                pro_dou = pro_dou.replace("{answer_2}", str(response2))
                # pro_dou = pro_dou.replace("[message1]", str(samples[i]))
                # pro_dou = pro_dou.replace("[message2]", str(samples[i+1]))
                mid["prompt"] = pro_dou
                pairs.append(mid)
            result_dict[key] = pairs
            
            


    """
    #在每个json文件中随机抽取样例
    samples = random.sample(source_data, 40)
    if len(samples)%2==1:
        samples = samples[:-1]
    """

    #替换待评分数据

    #api调用并保存结果
    api_result(result_dict, output_path, client)
    # api_result(output_fs_dou, output_file2, client)
    # api_result(output_dou_trans, output_file3, client)

#循环利用prompts内的数据调用api，并保存到对应文件
def batch_generate_and_save( result_dict, output_file, client):
    key_list_1 = list(range(10))
    key_list_2 = list(range(10,20))
    key_list_3 = list(range(20,30))
    key_list_4 = list(range(30,40))
    key_list_5 = list(range(40,50))
    key_4 = [33, 38, 30, 35]
    key_5 = [48, 47, 46]
    with open(output_file, 'w') as f:
        for key,value in tqdm(result_dict.items(),desc="Processing"):
            if int(key) not in key_5:
                continue
            for i,prompt in enumerate(value):
                pro = prompt["prompt"]
                data1 = prompt["data1"]
                data2 = prompt["data2"]
                source = {}
                source["data1"] = data1
                source["data2"] = data2
                try:
                    messages = [
                        {"role": "system", "content": system_alpagasus},
                        {"role": "user", "content": pro},
                ]
                    response = client.chat(messages)
                    result = {}
                    result["key"] = key
                    result["context"] = source
                    result["result"] = response
                    if response:
                        f.write(json.dumps(result) + '\n')
                        print(f"KEY {key} saved {i} response to {output_file}")
                        # print(response)
                    else:
                        print(f"{i} no response")
                except Exception as e:
                    print(e)

if __name__ == "__main__":
    dataset1 = r"/home/lc/fuwuqi/wizardlm-evol-instruction-70k/Wizard_70K-9994.json"      #待处理目录
    # dataset2 = r"E:\shixi\论文\data\databricks-dolly-15k.jsonl"      #待处理目录
    output_path = r"/home/lc/fuwuqi/wizardlm-evol-instruction-70k/wizardlm_score/alpagasus/wizardlm-result_5_last.json"     #目标目录

    openai_api_key = "sk-o2kUzhSEod2FWcmL582c1c68864e4eD5827aB573139e226e"  # key-6
    openai_api_base = "https://rerverseapi.workergpt.cn/v1"

    # 创建会话
    client = llm_openai(openai_api_key, openai_api_base, r"gpt-4o")
    simple_double(dataset1,  output_path, client)
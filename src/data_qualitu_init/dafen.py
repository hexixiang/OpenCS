import json
import random
import os
from api import llm_openai
from tqdm import tqdm

"""

"""
def api_result(prompts, out_file, client):
    batch_generate_and_save(prompts, out_file, client)

def simple_double(IN_DIR, OUT_DIR, client, SKIP):
    #提示词读取
    prompt_sim = open(r"E:\shixi\论文\prompt\dafen\simple_en.txt", 'r', encoding='utf-8').read()
    prompt_dou = open(r"E:\shixi\论文\prompt\dafen\double.txt", 'r', encoding='utf-8').read()
    prompt_fs_sim = open(r"E:\shixi\论文\prompt\dafen\simple_few-shot.txt", 'r', encoding='utf-8').read()
    prompt_fs_dou = open(r"E:\shixi\论文\prompt\dafen\double_few-shot.txt", 'r', encoding='utf-8').read()

    output_sim = []
    output_dou = []
    output_fs_sim = []
    output_fs_dou = []
    output_dou_trans = []
    #遍历json文件夹
    for file_name in os.listdir(IN_DIR):
        source_data = []
        if file_name in SKIP:
            continue
        #拼接目标文件路径
        file_path = os.path.join(IN_DIR,file_name)
        base,exc = os.path.splitext(file_name)
        output_file = os.path.join(OUT_DIR, base)
        out_path_sim = output_file + "_sim" +"_result.json"
        out_path_dou = output_file + "_dou" + "_result.josn"
        out_path_fs_sim = output_file + "_fs_dou" + "_result.josn"
        out_path_fs_dou = output_file + "_fs_dou" + "_result.josn"
        out_path_dou_trans = output_file + "_dou_trans" + "_result.josn"

        #读取josn文件到数组
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                source_data.append(data)
        
        #在每个json文件中随机抽取样例
        samples = random.sample(source_data, 10)
        if len(samples)%2==1:
            samples = samples[:-1]

        #替换待评分数据
        for da in samples:
            pro_sim = prompt_sim
            pro_sim = pro_sim.replace("[message]", str(da))
            output_sim.append(pro_sim)

        for da in samples:
            pro_fs_sim = prompt_fs_sim
            pro_fs_sim = pro_fs_sim.replace("[message]", str(da))
            output_fs_sim.append(pro_fs_sim)
        
        for i in range(0,len(samples),2):
            pro_dou = prompt_dou
            pro_dou = pro_dou.replace("[message1]", str(samples[i]))
            pro_dou = pro_dou.replace("[message2]", str(samples[i+1]))
            output_dou.append(pro_dou)

        for i in range(0,len(samples),2):
            pro_dou_trans = prompt_dou
            pro_dou_trans = pro_dou_trans.replace("[message2]", str(samples[i]))
            pro_dou_trans = pro_dou_trans.replace("[message1]", str(samples[i+1]))
            output_dou_trans.append(pro_dou_trans)

        for i in range(0,len(samples),2):
            pro_fs_dou = prompt_fs_dou
            pro_fs_dou = pro_fs_dou.replace("[message1]", str(samples[i]))
            pro_fs_dou = pro_fs_dou.replace("[message2]", str(samples[i+1]))
            output_fs_dou.append(pro_fs_dou)
        
        #api调用并保存结果
        api_result(output_sim, out_path_sim, client)
        api_result(output_dou, out_path_dou, client)
        api_result(output_fs_sim, out_path_fs_sim, client)
        api_result(output_fs_dou, out_path_fs_dou, client)
        api_result(output_dou_trans, out_path_dou_trans, client)

#循环利用prompts内的数据调用api，并保存到对应文件
def batch_generate_and_save( prompts, output_file, client):
        with open(output_file, 'w') as f:
            for i,prompt in enumerate(prompts):
                messages = [
                {"role": "system", "content": prompt},
            ]
                response = client.chat(messages)
                if response:
                    f.write(response + '\n')
                    print(f"Saved {i} response to {output_file}")
                    print(response)
                else:
                    print(f"{i} no response")

if __name__ == "__main__":
    IN_DIR = r"E:\shixi\论文\data"      #待处理目录
    OUT_DIR = r"E:\shixi\论文\result\dafen"     #目标目录

    openai_api_key = "sk-49VAm6uCdWSxb7xQEa64732596644dE8Aa30A58cFc73E146"
    openai_api_base = "http://rerverseapi.workergpt.cn/v1"

    # 创建会话
    client = llm_openai(openai_api_key, openai_api_base, r"gpt-4o")
    simple_double(IN_DIR, OUT_DIR, client, SKIP = ["alpaca.csv"])

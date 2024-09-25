# 由大模型生成每一条指令的tag
#-*- coding:utf-8 -*-
from openai import OpenAI
import json
from tqdm import tqdm
import time
# Set the proxy URL and port


inst = "You are a helpful assistant. Please identify tags of user intentions in the following user query and explain each tag. Please respond in the JSON format {“tag”: str, “explanation”: str}. Query: <query-to-tag>Assistant: <tagging-result>"

def api_demo(message):
    proxy_url = 'https://rerverseapi.workergpt.cn/v1'
    key = 'sk-49VAm6uCdWSxb7xQEa64732596644dE8Aa30A58cFc73E146'

# Set the http_proxy and https_proxy environment variables
# os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
# os.environ['https_proxy'] = f'{proxy_url}'
    """instruction = message[0]["system"]
    input = message[1]["context"]
    output = message[1]["target"]"""
    pro = ''
    result = {}
    result["data"] = message

    pro = prompt
    instruction = message["instruction"]
    position = pro.find("{instruction}")
    pro = pro[:position] + instruction + pro[position+len("{instruction}"):]
    
    # pro = pro.replace("{instruction}",instruction)
    # print(pro)
    client = OpenAI(api_key=key, base_url=proxy_url,)
    
    for i in range(1):
        
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": inst,
            },
            {
                "role": "user",
                "content": pro,
            }
        ],
        temperature=0.6,
        max_tokens=1024,
        top_p=1
        )
        result[f"key_{i}"] = (response.choices[0].message.content)

    # print(response.choices[0].message.content)
    
    return result

def all_data_api(data_list, output_file):
    with open(output_file,'w',encoding = 'utf-8') as f:
        a = 0    # noqa: F841
        for index,prompt in tqdm(enumerate(data_list)):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = api_demo(prompt)
                    f.write(json.dumps(response,ensure_ascii = False))
                    f.write('\n')
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print("尝试次数：",attempt+1)
                        time.sleep(10)  # 等待10秒后重试
                    else:
                        with open('bad_case.jsonl', 'a', encoding='utf-8') as f:
                            f.write(json.dumps(input_file[index]) + '\n')
                        # 记录错误日志或其他处理
                        print(f"Error: {e}, prompt saved to ad_case.jsonl")
                        response = None


input_file = r'data\alpaca\alpaca_22884_to_end.jsonl'
output_file = r'data\alpaca\tag_data\result_22884_to_end.json'
with open(r'prompts\biaoqian_pro.txt','r',encoding='utf-8') as f:
    prompt = f.read()
res_data = []
with open(input_file,'r',encoding = 'utf-8') as fp:
    # lines = fp.read()
    for line in fp:
        tmp = {}
        # print(line)
        data = json.loads(line.strip())
        res_data.append(data)

all_data_api(res_data,output_file)
from openai import OpenAI
import json
from tqdm import tqdm
import re
import my_prompts
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import collections
from itertools import cycle

openai_api_key = [
    "sk-LORwSL0exKUgxtlh6aDc3e9f49Bd4cE6969a7813Ef354336",
    "sk-r25G77V3Q5mulB5c542dFdC1799a499098Cf550bA760383c",
    "sk-PcEbbrgvYjCkw5Zj223792E9B4734bA68901F3E1398fE630",
    "sk-DUKSMPwRYanDO0Gj69F914Ff29E4429eA5B3181cC3507403",
    "sk-kxIkgnGCNZc5cxr653Ee6b4a0cFc4d50A51d249f1c1b221b"
    ]
openai_api_base = "https://rerverseapi.workergpt.cn/v1"

response_func = {
    "type": "object",
    "properties": {
        "explanation":
            {"type": "string", "description": "the explanation why the response is not good for the given instruction"}
        ,
        "better_response":
            {"type": "string", "description": "The Better response"}
        
    },
    "required":["explanation","better_response"]
}

instruction_func = {
    "type": "object",
    "properties": {
        "explanation":
            {"type": "string", "description": "the explanation why the instruction is not good for the given response"}
        ,
        "better_instruction":
            {"type": "string", "description": "The Better instruction"}
    },
    "required":["explanation","better_instruction"]
}

def get_response(message, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url=openai_api_base,
    )
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=message,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    stream=False
    )
    return response.choices[0].message.content

def call_openai_format_data(messages, function_parameters, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url=openai_api_base,
    )
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        functions=[
            {
                "name": "format_output",
                "description": "按照要求格式化输出内容",
                "parameters": function_parameters,
            }
        ],
        function_call={"name": "format_output"}
    )


    json_data = json.loads(response.choices[0].message.function_call.arguments)
    return json_data



def get_data(file_path):
    with open(file_path,"r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 使用正则表达式提取'```json\n'与'\n```'之间的内容
def extract_json(text):
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def extract_step1_text(text):
    match = re.search(r'Step 1:(.*?)Step 2:', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def process_item(item, api_key):
    instruction = item["instruction"]
    input = item["input"]
    response = item["output"]
    # 对于response的优化
    user_prompt_response = my_prompts.user_prompt_response.replace("{Instruction}", instruction).replace("{Response}", response)
    message_response = [
        {"role":"system","content":my_prompts.system_prompt_response},
        {"role":"user","content":user_prompt_response}
    ]
    # content_response = call_openai_format_data(message_response,response_func, api_key)
    content_response = get_response(message_response,api_key)
    try:
        new_response = str(json.loads(extract_json(content_response))["better_response"])
        print("normal:",new_response)
    except Exception:
        new_response = response
        print("error:",response)
    

    # 对于instruction的优化
    user_prompt_instruction = my_prompts.user_prompt_instruction.replace("{Instruction}", instruction).replace("{Response}", new_response)
    message_instruction = [
        {"role":"system","content":my_prompts.system_prompt_instruction},
        {"role":"user","content":user_prompt_instruction}
    ]
    # content_instruction = call_openai_format_data(message_instruction,instruction_func, api_key)
    # new_instruction = content_instruction["better_instruction"]        
    content_instruction = get_response(message_instruction,api_key)
    try:
        new_instruction = json.loads(extract_json(content_instruction))["better_instruction"]
    except Exception:
        new_instruction = instruction
        print(instruction)
    return {"instruction":new_instruction,"input":input,"output":new_response}

def main():
    data = get_data(r"src\data_augment\data\raw\test_1000_myprompt.json")[0:10]
    # print(len(data))
    good_case = []

    api_keys_cycle = cycle(openai_api_key)
    with ThreadPoolExecutor(max_workers=len(openai_api_key)) as executor:
        futures = [executor.submit(process_item, item, next(api_keys_cycle)) for item in data]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                good_case.append(result)
            except Exception as e:
                print(f"处理项目时发生错误: {e}")

    with open(r"src\data_augment\data\aug\test_10.jsonl", "w", encoding="utf-8") as f:
        json.dump(good_case, f, ensure_ascii=False,indent=4)


if __name__ == "__main__":
    main()
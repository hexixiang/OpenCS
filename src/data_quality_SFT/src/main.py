from prompts import article_prompt  # noqa: F401
from api import llm_openai
import pandas as pd
import json
from tqdm import tqdm

if __name__ == "__main__":
    # 定义常量
    openai_api_key = "sk-49VAm6uCdWSxb7xQEa64732596644dE8Aa30A58cFc73E146"
    openai_api_base = "http://rerverseapi.workergpt.cn/v1"

    # 创建会话
    client = llm_openai(openai_api_key, openai_api_base, r"gpt-4")

    # prompts模板
    system_prompt = """
    We would like to request your feedback on the performance of AI assistant in response to the instruction
    and the given input displayed following.
    Instruction: [Instruction]
    Input: [Input]
    Response: [Response]
    """
    user_prompt = """
    Please rate according to the [dimension] of the response to the instruction and the input. Each assistant
    receives a score on a scale of 0 to 5, where a higher score indicates higher level of the [dimension]. Please
    first output a single line containing the value indicating the scores. In the subsequent line, please provide a
    comprehensive explanation of your evaluation, avoiding any potential bias.
    """ 
    # 读取数据集然后填充模板，之后生成回复
    df = pd.read_csv(r"U:\codes\data_quality_SFT\data\alpaca_data.csv")
    # 按条数随机抽n条
    count = 10
    df = df.sample(n=count)

    for index, row in tqdm(df.iterrows(), total=count, desc="处理中"):
        try:
            input_tmp = "" if pd.isna(row["input"]) else row["input"]
            system = system_prompt.replace("[Instruction]", str(row["instruction"])) \
                                .replace("[Input]", str(input_tmp)) \
                                .replace("[Response]", str(row["output"]))
            user = user_prompt.replace("[dimension]", "[accuracy]")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            response = client.chat(messages=messages)
            response_dict = {
                "index": index,
                "system": system,
                "user": user,
                "response": response
            }
            # 将response_dict写入到results.jsonl中
            with open(rf"U:\codes\data_quality_SFT\results\results_{count}.jsonl", "a", encoding="utf-8") as f:
                f.write(f"{json.dumps(response_dict, indent=4)}\n")

        except Exception as e:
            print(f"Error: {e}")
            print(f"{index} 已记录到bad_case.csv中")
            with open(rf"U:\codes\data_quality_SFT\results\bad_case_{count}.csv", "a", encoding="utf-8") as f:
                # 记录原本csv对应的index的那一行
                f.write(f"{df.iloc[index]}\n")
            continue    
        
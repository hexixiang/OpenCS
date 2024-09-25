"""
Generate tags for each instruction using a large language model
"""
import openai
import json
from tqdm import tqdm
import time
import argparse

# Global variables
PROXY_URL = 'https://xxxx/v1'
API_KEY = 'sk-xxxx'
PROMPT_TEMPLATE = "You are a helpful assistant. Please identify tags of user intentions in the following user query and explain each tag. Please respond in the JSON format {“tag”: str, “explanation”: str}. Query: <query-to-tag>Assistant: <tagging-result>"

def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def llm_api(message, prompt):
    result = {"data": message}
    instruction = message["instruction"]
    prompt = prompt.replace("{instruction}", instruction)
    
    client = openai.OpenAI(api_key=API_KEY, base_url=PROXY_URL)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=1024,
        top_p=1
    )
    result["response"] = response.choices[0].message.content
    return result

def all_data_api(data_list, output_file, prompt):
    with open(output_file, 'w', encoding='utf-8') as f:
        for index, prompt_data in tqdm(enumerate(data_list)):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = llm_api(prompt_data, prompt)
                    f.write(json.dumps(response, ensure_ascii=False) + '\n')
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt number: {attempt + 1}")
                        time.sleep(10)
                    else:
                        with open('bad_case.jsonl', 'a', encoding='utf-8') as bad_case_file:
                            bad_case_file.write(json.dumps(prompt_data) + '\n')
                        print(f"Error: {e}, prompt saved to bad_case.jsonl")

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tags for instructions using a large language model")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file')
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to the prompt file')
    args = parser.parse_args()
    prompt = load_prompt(args.prompt_file)
    res_data = load_data(args.input_file)
    all_data_api(res_data, args.output_file, prompt)
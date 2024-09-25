"""
description: GPT-4 evaluation script
"""

import argparse
import json
import logging
import os
import time
from typing import List

import openai
from tqdm import tqdm
import concurrent.futures

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name to use")
    parser.add_argument("--input_dir", type=str, default="/path/to/converge_results", help="Input directory containing JSON files")
    parser.add_argument("--output_dir", type=str, default="/opecs/results/compare_judgement", help="Output directory for results")
    parser.add_argument("-k1", "--key_1", type=str, default="model_name1")
    parser.add_argument("-k2", "--key_2", type=str, default="model_name2")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens produced in the output",
    )
    return parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def api_generation(messages: List[dict], model: str, max_tokens: int, api_key: str):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
        )
        return response
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}. Retrying...")
        time.sleep(10)
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
        )
        return response

def parse_score(review: str):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise ValueError("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\nYou must manually fix the score pair."
        )
        return [-1, -1]

def gen_prompt(ques, ans1, ans2):

    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    prompt_template = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, criteria=criteria
    )
    return sys_prompt, prompt

def process_file(input_file: str, output_dir: str, model: str, max_tokens: int, key_1: str, key_2: str, api_key: str):
    
    with open(input_file, 'r') as f:
        qa_jsons = json.load(f)

    base_filename = os.path.basename(input_file)
    output_files = [
        os.path.join(output_dir, f"{key_1}-{key_2}-{base_filename}"),
        os.path.join(output_dir, f"{key_2}-{key_1}-{base_filename}")
    ]

    for key_1, key_2, output_file in [(key_1, key_2, output_files[0]), (key_2, key_1, output_files[1])]:
        message_list = []
        for qa in qa_jsons:
            if "sinstruct" in input_file:
                prompt = qa["prompt"]
            else:
                prompt = qa["text"]
            ans1 = qa[key_1]
            ans2 = qa[key_2]

            sys_prompt, prompt_text = gen_prompt(prompt, ans1, ans2)
            message = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_text},
            ]
            message_list.append(message)


        predictions = []
        pbar = tqdm(total=len(message_list), desc=f"Processing {output_file}")
        for idx, messages in enumerate(tqdm(message_list)):
            prediction = api_generation(messages=messages, model=model, max_tokens=max_tokens, api_key=api_key)
            predictions.append(prediction)
            review = prediction['choices'][0]['message']['content']
            scores = parse_score(review)
            print(f"Scores: {scores}")
            qa_jsons[idx]["review"] = review
            qa_jsons[idx]["score"] = scores
            time.sleep(0.1) 
            pbar.update(1)
        pbar.close()
        with open(output_file, "w") as f_out:
            json.dump(qa_jsons, f_out, indent=4)

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    input_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir) if file.endswith('.jsonl')]
    api_keys = [
        "sk-xx1", "sk-xx2", "sk-xx3", "sk-xx4", "sk-xx5"
    ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, input_file in enumerate(input_files):
            api_key = api_keys[i % len(api_keys)]
            print(input_file)
            futures.append(executor.submit(process_file, input_file, args.output_dir, args.model, args.max_tokens, args.key_1, args.key_2, api_key))
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")
if __name__ == "__main__":
    main()

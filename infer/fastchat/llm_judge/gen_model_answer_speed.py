"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
import glob
import ray

def run_eval(
    model_path,
    model_id,
    question_file,
    answer_file,
    column,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
):
    questions = load_questions(question_file, begin=None, end=None)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                column,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    column,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
):
    model, tokenizer = load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for question in tqdm(questions):
        temperature = 0.7
        result = []
        if "sinstruct" in column:
            instances = question['instances']
            assert len(instances) == 1
            if instances[0]['input'] != "":
                input_data = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\r\n\r\n"
                f"### Instruction:\r\n{question[column]}\r\n\r\n### Input:\n{instances[0]['input']}\r\n\r\n### Response:"
                )
            else:
                input_data = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\r\n\r\n"
                f"### Instruction:\r\n{question[column]}\r\n\r\n### Response:"
                )
        else:
            input_data = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\r\n\r\n"
                f"### Instruction:\r\n{question[column]}\r\n\r\n### Response:"
                )
        input_ids = tokenizer(input_data).input_ids
        # some models may error out when generating long outputs
        try:
            output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                    )
            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]) :]
            output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()
        except RuntimeError as e:
            print("ERROR question ID: ", question["question_id"])
            output = "ERROR"
        
        ans_json = {
            "question_id": question["question_id"],
            "model_id": model_id,
            "column": question[column],
            "answer": output
        }
        result.append(ans_json)
        
    # dump answers
    with open(answer_file, "w") as json_file:
        json.dump(result, json_file, indent=4)



def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/hxx/model/output/Llama-2-7b-hf-cluster_pair_score_myprompt_sharegpt-e15lr1e-5",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, default="Llama-2-7b-hf-cluster_pair_score_myprompt_sharegpt-e15lr1e-5", help="A custom name for the model."
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="/home/hxx/long-is-more-for-alignment/data/test/",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/home/hxx/long-is-more-for-alignment/output/test/",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=4,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=8, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()


    os.makedirs(args.save_path, exist_ok=True)
    test_data = ['koala_test_set.jsonl', 'sinstruct_test_set.jsonl', 'wizardlm_test_set.jsonl', 'vicuna_test_set.jsonl', 'lima_test.jsonl']
    col = ['prompt', 'instruction', 'Instruction', 'text', 'conversations']
    if args.model_path.endswith("/"):
        model_name = args.model_path.split("/")[-2]
    else:
        model_name = args.model_path.split('/')[-1]
    for i in range(len(test_data)):
        path = args.test_path + test_data[i]
        name = test_data[i].split('_')[0]
        sv_path = args.save_path + model_name + "_" + name + ".jsonl"

        print(f"Processing {path} -> {sv_path}")

        run_eval(
            model_path=args.model_path,
            model_id=args.model_id,
            question_file=path,
            answer_file=sv_path,
            column=col[i],
            max_new_token=args.max_new_token,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            max_gpu_memory=args.max_gpu_memory,
            dtype=str_to_torch_dtype(args.dtype),
            revision=args.revision,
        )

if __name__ == "__main__":
    main()
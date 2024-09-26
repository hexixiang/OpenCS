import json
import random
import os
from api import llm_openai
from tqdm import tqdm

def api_result(prompts, out_file, client):
    """Generate responses from the API and save to file."""
    batch_generate_and_save(prompts, out_file, client)

def simple_double(IN_DIR, OUT_DIR, client, SKIP):
    """Process JSON files and generate outputs based on prompts."""
    
    # Read prompts from files
    prompt_sim = open("xxx", 'r', encoding='utf-8').read()
    prompt_dou = open("xxx", 'r', encoding='utf-8').read()
    prompt_fs_sim = open("xxx", 'r', encoding='utf-8').read()
    prompt_fs_dou = open("xxx", 'r', encoding='utf-8').read()

    # Initialize output lists
    output_sim, output_dou, output_fs_sim, output_fs_dou, output_dou_trans = [], [], [], [], []

    # Iterate through JSON files in the input directory
    for file_name in os.listdir(IN_DIR):
        if file_name in SKIP:
            continue
        
        file_path = os.path.join(IN_DIR, file_name)
        base, _ = os.path.splitext(file_name)
        output_file = os.path.join(OUT_DIR, base)

        # Output file paths
        out_paths = {
            "sim": f"{output_file}_sim_result.json",
            "dou": f"{output_file}_dou_result.json",
            "fs_sim": f"{output_file}_fs_sim_result.json",
            "fs_dou": f"{output_file}_fs_dou_result.json",
            "dou_trans": f"{output_file}_dou_trans_result.json"
        }

        # Read JSON data into an array
        with open(file_path, 'r', encoding='utf-8') as f:
            source_data = [json.loads(line) for line in f]

        # Randomly sample data
        samples = random.sample(source_data, min(10, len(source_data)))
        if len(samples) % 2 == 1:
            samples = samples[:-1]

        # Replace placeholders in prompts
        for da in samples:
            output_sim.append(prompt_sim.replace("[message]", str(da)))
            output_fs_sim.append(prompt_fs_sim.replace("[message]", str(da)))

        for i in range(0, len(samples), 2):
            output_dou.append(prompt_dou.replace("[message1]", str(samples[i]))
                               .replace("[message2]", str(samples[i + 1])))
            output_dou_trans.append(prompt_dou.replace("[message2]", str(samples[i]))
                                     .replace("[message1]", str(samples[i + 1])))
            output_fs_dou.append(prompt_fs_dou.replace("[message1]", str(samples[i]))
                                  .replace("[message2]", str(samples[i + 1])))

        # Call the API and save results
        api_result(output_sim, out_paths["sim"], client)
        api_result(output_dou, out_paths["dou"], client)
        api_result(output_fs_sim, out_paths["fs_sim"], client)
        api_result(output_fs_dou, out_paths["fs_dou"], client)
        api_result(output_dou_trans, out_paths["dou_trans"], client)

def batch_generate_and_save(prompts, output_file, client):
    """Generate responses for each prompt and save them to a file."""
    with open(output_file, 'w') as f:
        for i, prompt in enumerate(prompts):
            messages = [{"role": "system", "content": prompt}]
            response = client.chat(messages)
            if response:
                f.write(response + '\n')
                print(f"Saved response {i} to {output_file}")
            else:
                print(f"No response for prompt {i}")

if __name__ == "__main__":
    IN_DIR = "xxx"  # Input directory
    OUT_DIR = "xxx"  # Output directory

    # Sensitive information placeholders
    openai_api_key = "xxx"
    openai_api_base = "http://xxx/v1"

    # Create client session
    client = llm_openai(openai_api_key, openai_api_base, r"gpt-4o")
    simple_double(IN_DIR, OUT_DIR, client, SKIP=["alpaca.csv"])

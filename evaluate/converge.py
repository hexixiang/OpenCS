"""
Description:
This script is a useful tool for merging JSON files containing model outputs, facilitating comparison and analysis of different models' performance on the same set of questions.


Usage:
python3 converge.py --folder1 /path/to/first/folder --folder2 /path/to/second/folder --output_folder /path/to/output/folder
"""


import json
import os
import argparse

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def merge_json_files(folder1, folder2, output_folder, test_data):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for test_name in test_data:
        # Find the file containing the test set name
        file1 = next((f for f in os.listdir(folder1) if test_name in f), None)
        file2 = next((f for f in os.listdir(folder2) if test_name in f), None)

        if file1 is None or file2 is None:
            print(f"Matching test set not found in the folders: {test_name}")
            continue

        # Load data from both files
        data1 = load_json(os.path.join(folder1, file1))
        data2 = load_json(os.path.join(folder2, file2))

        merged_list = []

        for item1, item2 in zip(data1, data2):
            if item1["question_id"] != item2["question_id"]:
                print(f"Question IDs do not match in the two files: {item1['question_id']} vs {item2['question_id']}")
                continue
            merged_item = {
                "id": item1["question_id"],
                "text": item1["text"],
                "prompt": item1["prompt"],
                item1["model_id"]: item1["output"],
                item2["model_id"]: item2["output"],
            }
            merged_list.append(merged_item)

        # Output the merged file
        output_file = os.path.join(output_folder, f"merged_{test_name}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)

        print(f"Merging completed, results saved in {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON files from two folders into one.")
    parser.add_argument("--folder1", default="path/to/folder1", help="Path to the first folder containing JSON files.")
    parser.add_argument("--folder2", default="path/to/folder2", help="Path to the second folder containing JSON files.")
    parser.add_argument("--output_folder", default="path/to/output_folder", help="Path to the output folder to save merged JSON files.")
    
    args = parser.parse_args()

    test_data = ['koala_test_set.jsonl', 'sinstruct_test_set.jsonl', 'wizardlm_test_set.jsonl', 'vicuna_test_set.jsonl', 'lima_test_set.jsonl']

    merge_json_files(args.folder1, args.folder2, args.output_folder, test_data)

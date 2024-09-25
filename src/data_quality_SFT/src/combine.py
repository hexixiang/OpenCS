import os
import json

def merge_json_files(src_dir, output_file):
    """
    遍历 src_dir 目录及其子目录，将所有并行打分的多个 JSON 文件合并到一个 JSON 文件中。
    
    :param src_dir: 源目录路径
    :param output_file: 输出 JSON 文件路径
    """
    all_data = []

    # 遍历目录及其子目录
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith('.jsonl'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                # 读取 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        data = json.loads(line)
                        all_data.append(data)

    # 将所有数据写入到输出 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"All JSON files have been merged into {output_file}")

# 示例用法
source_directory = 'E:\shixi\论文\data'  # 修改为你的源目录路径
output_json_file = 'E:\shixi\论文\data\output_file.json'  # 修改为你的输出 JSON 文件路径

merge_json_files(source_directory, output_json_file)
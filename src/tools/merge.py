# 合并文件
from utils import count_lines
import json     # noqa: F401
import os   # noqa: F401

# 合并两个jsonl文件
def merge_jsonl(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2, open(output_file, 'w', encoding='utf-8') as out:
        for line in f1:
            out.write(line)
        for line in f2:
            out.write(line)

# l = count_lines(r'data\alpaca\tag_data\tag_alpaca_2501_to_end_only_instruction.jsonl')+2501
merge_jsonl(r'data\alpaca\raw_tag_data\tag_alpaca_1_to_22884_only_instruction.jsonl', r'data\alpaca\raw_tag_data\result_22884_to_end.jsonl', r'data\alpaca\raw_tag_data\alpaca_tag_total.jsonl')



print("合并文件完成")



        



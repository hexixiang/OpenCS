# 截断数据集的一部分
import json  # noqa: F401


def split_dataset_jsonl(input_file, output_file, start_index, end_index):
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if start_index <= i < end_index:
                f_out.write(line)


end_index = len(open('data/alpaca/alpaca.jsonl', 'r', encoding='utf-8').readlines())    
split_dataset_jsonl('data/alpaca/alpaca.jsonl', 'data/alpaca/alpaca_22884_to_end.jsonl', 22884, end_index)

print("Done!")








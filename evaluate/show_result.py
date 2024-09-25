import argparse
import json
import os
from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="/path/to/model1_model2", help="Path to the folder containing the JSON files.")
    parser.add_argument("--output_path", type=str, default="/path/graph/model1_model2", help="The output path for the generated plot.")
    return parser.parse_args()

def result_process(folder_path: str):
    results = defaultdict(lambda: {'win': 0, 'tie': 0, 'lose': 0})
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            model1 = "model_name1"
            model2 = "model_name2"
            dataset = filename.split("_")[-3]
            
            path = os.path.join(folder_path, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                
            for item in data:
                score1, score2 = item['score']
                if score1 > score2:
                    results[(model1, model2)]['win'] += 1
                elif score1 < score2:
                    results[(model1, model2)]['lose'] += 1
                else:
                    results[(model1, model2)]['tie'] += 1

    return results

def graph(results: dict, output_path: str):
    labels = list(results.keys())
    win_data = [results[label]['win'] for label in labels]
    tie_data = [results[label]['tie'] for label in labels]
    lose_data = [results[label]['lose'] for label in labels]

    total = [win + tie + lose for win, tie, lose in zip(win_data, tie_data, lose_data)]
    win_percent = [win / tot * 100 for win, tot in zip(win_data, total)]
    tie_percent = [tie / tot * 100 for tie, tot in zip(tie_data, total)]
    lose_percent = [lose / tot * 100 for lose, tot in zip(lose_data, total)]

    height = 0.8
    fig, ax = plt.subplots(figsize=(18,12))
    left = np.zeros(len(labels))

    category_colors = plt.colormaps['tab20c'](np.linspace(0., 0.15, 3))
    idx = 0
    label_colors = ['white', 'black', 'black']

    for judge, judge_count in zip(['Win', 'Tie', 'Lose'], [win_percent, tie_percent, lose_percent]):
        p = ax.barh(labels, judge_count, height, label=judge, left=left, color=category_colors[idx, :])
        left += judge_count
        ax.bar_label(p, label_type='center', fontsize=16, fmt="%g", color=label_colors[idx])
        idx += 1

    plt.xticks([])
    plt.yticks(fontsize=16)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.22), fontsize=16, columnspacing=0.8)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def main():
    args = parse_args()
    results = result_process(args.folder_path)
    graph(results, args.output_path)

if __name__ == "__main__":
    main()

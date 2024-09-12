# 目前想找一个好的挑选数量的方法
import os
import json
import sys  # noqa: F401
import numpy as np
from transformers import BertTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
from kcenter_greedy import kCenterGreedy
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 配置实验参数
bert_model_path = r"res\bert-base-uncased"
bert_embedding_path = r"data\alpaca\bert_embedding.npy"

# set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
@torch.no_grad()
def bert_embedding(texts,batch=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model = AutoModel.from_pretrained(bert_model_path).to(device)
    # 将文本转化为BERT模型可识别的token序列
    encoded_texts = tokenizer(texts,return_tensors="pt",truncation=True,padding=True,max_length=96)
    encoded_texts =  encoded_texts.to(device)
    cls_hid_li = []
    # 使用BERT模型对每个文本序列进行编码,提取其语义向量
    # 每个序列的 [CLS] token 的隐藏状态
    i= 0
    for i in tqdm(range(0,len(texts),batch)):
        last_hids = model(input_ids=encoded_texts["input_ids"][i:i+batch],
                          attention_mask=encoded_texts["attention_mask"][i:i+batch])['last_hidden_state']
        cls_hids = last_hids[:,0,:].squeeze()
        cls_hid_li.append(cls_hids)
        i+= batch
        torch.cuda.empty_cache()  # 清理 CUDA 缓存
    # 将所有文本的embedding连成特征矩阵
    cls_hids_tensor = torch.concat(cls_hid_li, dim=0)
    return np.array(cls_hids_tensor.cpu())

# 数据采样
def sample_func(text_list,K):
    result = []
    if os.path.exists(bert_embedding_path):
        # print('load bert embedding')
        text_embedding = np.load(bert_embedding_path)
    else:
        print("bert embedding not exist, start embedding")
        text_embedding = bert_embedding(text_list)
        np.save(bert_embedding_path,text_embedding)
    
    result = []

    k_center = kCenterGreedy(text_embedding)
    
    already_selected = None
    #for _ in range(K):
    result = k_center.select_batch_(text_embedding,already_selected,K)
        #result = result + new_data
        #already_selected += new_data
    return result

def calculate_total_distance(embeddings, selected_indices):
    total_distance = 0
    for i in range(len(embeddings)):
        min_distance = float('inf')
        for j in selected_indices:
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            if distance < min_distance:
                min_distance = distance
        total_distance += min_distance
    return total_distance

def calculate_silhouette_score(embeddings, selected_indices):
    labels = np.zeros(len(embeddings))
    for idx in selected_indices:
        labels[idx] = 1
    score = silhouette_score(embeddings, labels)
    return score

def find_elbow_point(distances):
    # 使用肘部法则找到最佳的K值
    n_points = len(distances)
    all_coords = np.vstack((range(n_points), distances)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.linalg.norm(vec_to_line, axis=1)
    best_k = np.argmax(dist_to_line)
    return best_k

# def determine_best_k(text_list, max_k):
#     if os.path.exists(r"/root/jt_ws/sft_data_selection/data/alpaca/bert_embedding.npy"):
#         # print('load bert embedding')
#         text_embedding = np.load(r"/root/jt_ws/sft_data_selection/data/alpaca/bert_embedding.npy")
#     else:
#         print("bert embedding not exist, start embedding")
#         text_embedding = bert_embedding(text_list)
#         np.save(r"/root/jt_ws/sft_data_selection/data/alpaca/bert_embedding.npy",text_embedding)
#     distances = []
#     for K in tqdm(range(1, max_k + 1,5), desc="Calculating distances"):
#         selected_indices = sample_func(text_list, K)
#         total_distance = calculate_total_distance(text_embedding, selected_indices)
#         distances.append(total_distance)
    
#     best_k = find_elbow_point(distances)
    
#     plt.plot(range(1, max_k + 1, 5), distances, 'bx-')
#     plt.xlabel('K')
#     plt.ylabel('Total Distance')
#     plt.title('Elbow Method For Optimal K')
#     plt.show()
#     # 保存图片到指定路径
#     plt.savefig(r"/root/jt_ws/sft_data_selection/results/kcenter_greedy_test/elbow_method.png")
#     print(f"Best K: {best_k}")
#     return best_k

def determine_best_k(text_list, max_k):
    if os.path.exists(bert_embedding_path):
        # print('load bert embedding')
        text_embedding = np.load(bert_embedding_path)
    else:
        print("bert embedding not exist, start embedding")
        text_embedding = bert_embedding(text_list)
        np.save(bert_embedding_path,text_embedding)
    scores = []
    k_values = list(range(2, max_k + 1, 100))
    for K in tqdm(k_values):
        selected_indices = sample_func(text_list, K)
        score = calculate_silhouette_score(text_embedding, selected_indices)
        scores.append(score)
    
    best_k = k_values[np.argmax(scores)]
    
    plt.plot(k_values, scores, 'bx-')  # 使用k_values作为x轴
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score For Optimal K')
    plt.show()
    # 保存图片到指定路径
    plt.savefig(r"results\kcenter_greedy_test\elbow_method.png")
    print(f"Best K: {best_k}")
    
    return best_k  # 返回最佳的K值

def batch_exp(output_file_prefix,candidate_list):
    """sumary_line
    Keyword arguments:
    output_file_prefix -- 输出文件的前缀
    candidate_list -- 需要实验的K列表
    Return: 无返回值，但会将结果保存到指定路径
    """
    for k in tqdm(candidate_list):
        output_file = f"{output_file_prefix}_{k}.json"
        res = sample_func(text_list = instruction_list, K = k)
        data_li = []
        for index in res:
            try:
                data_li.append(data_dict[int(index)])
            except KeyError:
                print(f"Index {index} not found in the DataFrame")
        json.dump(obj=data_li,fp=open(output_file,"w",encoding="utf-8"),indent=4,ensure_ascii=False)
        print(f"{k} done!")


    



if __name__ == "__main__":
    input_file = r"data\alpaca\alpaca.csv"
    data = pd.read_csv(input_file)
    data = data.fillna("")  # 将NaN替换为空字符串
    data_dict = data.to_dict(orient="records")
    instruction_list = [item["instruction"] for item in data_dict]
    output_file_prefix = r"results\kcenter_greedy_test\alpaca_kcenter_greedy"
    # 单次实验
    # max_k = 10000  # 你可以根据数据集大小调整这个值
    # K = determine_best_k(instruction_list, max_k)
    # 批量实验
    # 分别取data的1%，5%，10%，20%，30%，40%，50%
    data_len = len(data_dict)
    candidate_list = [int(data_len*0.01),int(data_len*0.05),int(data_len*0.1),int(data_len*0.2),int(data_len*0.3),int(data_len*0.4),int(data_len*0.5)]
    print("实验开始")
    batch_exp(output_file_prefix,candidate_list)


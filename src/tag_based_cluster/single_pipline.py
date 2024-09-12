# 构建一条处理的流水线，不封装
# 导入必要的库
import json
import re
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,KMeans
from nltk.stem import PorterStemmer
import torch
from transformers import BertTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from mlxtend.frequent_patterns import fpgrowth
from tqdm import tqdm
import numpy as np
import pandas as pd

def default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class TAG:
    def __init__(self,raw_tag):
        self.raw_tag = raw_tag  # 原始tag
        self.tag = None  # 词性还原后的tag
        self.count = None  # 出现次数
        self.label = None  # 聚类标签
        self.instrucion_id = None  # 对应指令的id-tags映射


class TAG_CLUSTER:
    def __init__(self):
        # data\dolly_15K\raw_tag_data\dolly_15k_tag_total_with_index.jsonl
        # data\alpaca\raw_tag_data\alpaca_tag_total_with_index.jsonl
        # data\Wizard_70K\raw_tag_data\wizardlm-70k_tag_total_with_index.jsonl
        self.dataset_type = "alpaca"    # dolly_15K, alpaca, Wizard_70K
        self.dataset = ""
        self.raw_data_path = os.path.join(rf'data\{self.dataset_type}\raw_tag_data',self.dataset)
        self.processsed_tag_save_path = rf'data\{self.dataset_type}\processed_tag_data'
        self.bert_model_path = r'res\bert-base-uncased'
        self.embedding_path = rf'data\{self.dataset_type}\total_embeddings_phrasebert_alpha_20.npy'
        self.phrasebert_model_path = r'res\phrase-bert'
        self.route(dateset_type=self.dataset_type)  # 根据dataset_type初始化参数
        self.alpha = 20
        self.embedding_type = "phrasebert" #bert,phrasebert
        self.methods = 'kmeans' #kmeans,dbscan
        self.n_clusters = 50
        self.eps = 0.5
        self.min_samples = 5
        self.instruction_id = []
        self.label_tag_mapping = {}

    def execute(self):
        # 1. 读取数据jsonl
        data = self.read_data()
        # 2. 去除不符合格式的数据并添加到bad_case中
        self.normalize_tag_format(data)
        # 3. 得到所有good_case_tag_data标签与对应解释
        self.instruction_id = self.get_good_case_tag_data()
        # 4. 统计所有good_case_tag_data中所有标签的数量
        tag_count = self.count_tag_frequency(self.instruction_id)
        # 5. 基于频率进行标签的筛选
        tag_count = self.filter_tag_by_frequency(tag_count,alpha=self.alpha)
        # 6. 基于规则的过滤(数量可能减少，因为处理后tag可能有重复)
        tag_count = self.filter_tag_by_rule(tag_count)
        # 7. 语义聚合
        text_embedding = self.embeding_tag(tag_count)
        # 聚合
        self.cluster_text_embeddings(text_embedding, methods=self.methods, n_clusters=self.n_clusters,eps=self.eps, min_samples=self.min_samples)
        # 将instruction_id写入文件
        existing_labels = set()  # 创建一个集合来存储已经存在的labels

        with open(rf"src/tag_based_cluster/instruction_id_{self.dataset_type}.json", "w", encoding="utf-8") as f:
            for item in self.instruction_id:
                # 原本的代码逻辑
                # labels_tuple = tuple(item[1])  # 将labels列表转换为元组，因为列表不能作为集合的元素
                # if labels_tuple not in existing_labels:  # 检查当前labels是否已经存在
                #     json_data = json.dumps(
                #         [{
                #             "tags": item[0],
                #             "labels": item[1],
                #             "index": item[2]
                #         }],
                #         ensure_ascii=False, default=default
                #     )
                #     f.write(json_data + "\n")
                #     existing_labels.add(labels_tuple)  # 将新的labels添加到集合中

                # 改为同一类的取最长的，修改后的代码逻辑
                labels_tuple = tuple(item[1])  # 将labels列表转换为元组，因为列表不能作为集合的元素
                
                json_data = json.dumps(
                    [{
                        "tags": item[0],
                        "labels": item[1],
                        "index": item[2],
                        "labels_tuple": labels_tuple    # 添加一个标识
                    }],
                    ensure_ascii=False, default=default
                )
                f.write(json_data + "\n")
                if labels_tuple not in existing_labels:  # 检查当前labels是否已经存在
                    existing_labels.add(labels_tuple)  # 将新的labels添加到集合中               
        print(f"最终数据集的长度为{len(existing_labels)}")
        # 8. 关联聚类(暂时不写)
        # self.fp_growth(tag_count)
        # 9. 根据tags对于instruction进行聚类抽样
        # self.cluster_instruction_by_tags(tag_count)

    def route(self,dateset_type):
        if self.dataset_type == "alpaca":
            self.dataset = r"alpaca_tag_total_with_index.jsonl"
        elif self.dataset_type == "dolly":
            self.dataset = r"dolly_15k_tag_total_with_index.jsonl"
        elif self.dataset_type == "wizard":
            self.dataset = r"wizardlm-70k_tag_total_with_index.jsonl"
        else:
            print("No such dateset")
            raise TypeError


    # 1. 读取数据jsonl
    def read_data(self):
        data = []
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    # 2. 去除不符合格式的数据并添加到bad_case中
    def normalize_tag_format(self,data):
        good_case = []
        bad_case = []
        for item in data:
            flag = False
            try:
                raw_tags = json.loads(self.extract_content(item["key_0"]))
                item["tags"] = raw_tags
                del item["key_0"]
                flag = True
            except Exception:
                try: 
                    raw_tags = json.loads(item["key_0"])
                    item["tags"] = raw_tags
                    del item["key_0"]
                    flag = True
                except Exception as e:
                    print(e)
            if flag:
                good_case.append(item)
            else:
                bad_case.append(item)


        with open(os.path.join(self.processsed_tag_save_path, self.dataset.split('.')[0]+'_good'+"."+self.dataset.split('.')[1]), 'w', encoding='utf-8') as f:
            for item in good_case:
                f.write(json.dumps(item) + '\n')

        with open(os.path.join(self.processsed_tag_save_path, self.dataset.split('.')[0]+'_bad'+"."+self.dataset.split('.')[1]), 'w', encoding='utf-8') as f:
            for item in bad_case:
                f.write(json.dumps(item) + '\n')

    # 3.得到所有good_case_tag_data标签与对应解释
    def get_good_case_tag_data(self):
        good_case_tag_data = []
        with open(os.path.join(self.processsed_tag_save_path, self.dataset.split('.')[0]+'_good'+"."+self.dataset.split('.')[1]), 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 把index加上方便后续处理
                good_case_tag_data.append((item["tags"],item["index"]))
        # 统计平均每一条数据有多少tag以及最大最小值
        tag_counts = [len(item[0]) for item in good_case_tag_data]
        avg_tag_count = sum(tag_counts) / len(tag_counts)
        max_tag_count = max(tag_counts)
        min_tag_count = min(tag_counts)
        print("***********初始情况记录日志***************")
        print(f"instruction_id原始长度为{len(good_case_tag_data)}")
        print(f"每一条数据平均有{avg_tag_count}个tag，最大{max_tag_count}个，最小{min_tag_count}个")
        print("***********初始情况记录日志***************")
        print("\n")

        # plt.hist(tag_counts, bins=range(min_tag_count, max_tag_count + 2), edgecolor='black', align='left')
        # plt.xlabel('tag_appearrance_time')
        # plt.ylabel('data_count')
        # plt.title('tag_appearrance_time_distribution')
        # plt.xticks(range(min_tag_count, max_tag_count + 1))

        # # 获取直方图的柱子和边界
        # counts, bins, patches = plt.hist(tag_counts, bins=range(min_tag_count, max_tag_count + 2), edgecolor='black', align='left')

        # # 在每个柱子上标注出现次数
        # for count, bin in zip(counts, bins):
        #     if count >= 0:  # 只标注非零的柱子
        #         plt.text(bin, count, str(int(count)), ha='center', va='bottom')

        # plt.show()

        return good_case_tag_data


    # 4.统计所有good_case_tag_data中所有标签的数量
    def count_tag_frequency(self,good_case_tag_data):
        # 所有的标签以及对应数量
        tag_count = {}  
        for item in good_case_tag_data:
            for tag in item[0]:
                if tag["tag"] in tag_count:
                    tag_count[tag["tag"]] += 1
                else:
                    tag_count[tag["tag"]] = 1
        print("***********计数日志***************")
        print(f"原始不同tag个数为{len(tag_count)}个")
        # 找到tag_count中的最大最小，以及平均值
        max_count = max(tag_count.values())
        min_count = min(tag_count.values())
        avg_count = sum(tag_count.values()) / len(tag_count)
        print(f"标签最多出现次数为{max_count}，最小{min_count}次，平均{avg_count}次")
        print("***********计数日志***************")
        print("\n")
        # 绘制tag_count的直方图
        # tag_counts = list(tag_count.values())
        # plt.bar(range(len(tag_counts)), tag_counts)
        # plt.xlabel('Tag')
        # plt.ylabel('Count')
        # plt.title('Tag Count Distribution')
        # plt.show()

        # 画饼状图
        # plt.pie(list(tag_count.values()), labels=list(tag_count.keys()), autopct='%1.1f%%')
        # plt.axis('equal')
        # plt.show()
        return tag_count

    # 5.基于频率进行标签的筛选
    # 论文描述：过滤掉在整个标注数据集中出现次数少于α次的长尾标签。α是与数据集规模相关的超参数
    def filter_tag_by_frequency(self, tag_count, alpha=None):
        # 删除tag_count中出现次数小于alpha的标签
        for tag in list(tag_count.keys()):
            if tag_count[tag] < alpha:
                del tag_count[tag]
        
        # 筛选并过滤self.instruction_id中的标签
        self.instruction_id = [
            (tags, index) for tags, index in self.instruction_id
            if any(tag_dict["tag"] in tag_count for tag_dict in tags)
        ]
        
        # 过滤每个tags列表中的标签
        self.instruction_id = [
            ([tag_dict for tag_dict in tags if tag_dict["tag"] in tag_count], index)
            for tags, index in self.instruction_id
        ]
        print("***********基于频率过滤日志***************")
        print(f"基于频率过滤后的instruction_id长度为{len(self.instruction_id)}")
        print(f"基于频率过滤后tag数量为{len(tag_count)},alpha为{alpha}")
        print("***********基于频率过滤日志***************")
        print("\n")
        return tag_count
# 6. 基于规则的过滤(数量可能减少，因为处理后tag可能有重复)
# 论文描述：将所有标签都转换为小写字符，以避免大小写的影响。我们还将所有特殊字符替换为空格，以进一步聚合标记。最后，我们在NLTK的支持下对每个标签应用词干提取
    def filter_tag_by_rule(self,tag_count):
        ps = PorterStemmer()
        tag_mapping = {}

        for tag in list(tag_count.keys()):
            # 将标签转换为小写
            new_tag = tag.lower()
            # 替换所有特殊字符为空格
            new_tag = re.sub(r'\W+', ' ', new_tag)
            # 应用词干提取
            new_tag = ps.stem(new_tag)
            
            # 更新字典
            tag_mapping[tag] = new_tag
            tag_count[new_tag] = [tag,tag_count.pop(tag)]

        # 更新instruction_id中的tags
        self.instruction_id = [
            ([{"tag": tag_mapping.get(tag_dict["tag"], tag_dict["tag"]),"tag_orgin":tag_dict["tag"],"explanation": tag_dict["explanation"]} for tag_dict in tags], index)
            for tags, index in self.instruction_id
        ]
        print("***********基于规则过滤日志***************")
        print(f"基于规则过滤后tag数量为{len(tag_count)}")
        print(f"基于规则过滤后instruction_id长度为{len(self.instruction_id)}")
        print("***********基于规则过滤日志***************")
        print("\n")
        # print(self.instruction_id[7])
        return tag_count


# 7.语义聚合
# 论文方法：使用PHRASEBERT得到标签的嵌入，之后使用DBSCAN算法进行语义聚类
# 使用bert_embedding得到标签的嵌入
# 加载预训练的BERT模型和tokenizer
    def embeding_tag(self,tag_count):
        print("***********语义聚类日志***************")
        tag_list = [tag_dict["tag"] for tags, _ in self.instruction_id for tag_dict in tags]

        if os.path.exists(self.embedding_path):
            text_embedding = np.load(self.embedding_path)
        else:
            print("embedding not exist, start embedding")
            if self.embedding_type == "bert":
                text_embedding = self.bert_embedding(tag_list)
            elif self.embedding_type == "phrasebert":
                text_embedding = self.phrasebert_embedding(tag_list)
            np.save(self.embedding_path,text_embedding)
        print("语义聚类完成！")
        print("***********语义聚类日志***************")
        print("\n")
        return text_embedding

# 使用DBSCAN算法基于text_embedding进行语义聚类得到聚类标签
    def cluster_text_embeddings(self,text_embeddings, methods='kmeans',eps=0.5, min_samples=5, n_clusters=None):
        print("***********聚类日志***************")
        # 创建聚类模型
        if methods == 'dbscan':
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        elif methods == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters)
        print(f"聚类方法为{methods}")
        # 进行聚类
        if methods == 'dbscan':
            labels = dbscan.fit_predict(text_embeddings)
        elif methods == 'kmeans':
            labels = kmeans.fit_predict(text_embeddings)

        # 根据labels更新instruction_id
        tag_list = [tag_dict["tag"] for tags, _ in self.instruction_id for tag_dict in tags]
        tag_label_mapping = dict(zip(tag_list, labels))
        # 得到一张label到tag的映射表
        self.label_tag_mapping = {}
        for i in range(len(labels)):
            if labels[i] not in self.label_tag_mapping:
                self.label_tag_mapping[labels[i]] = [tag_list[i]]
            else:
                self.label_tag_mapping[labels[i]].append(tag_list[i])

        # 更新instruction_id，添加标签对应的label序列
        i = 0
        for tags, index in self.instruction_id:
            labels_list = []
            for tag_dict in tags:
                tag_dict["label"] = tag_label_mapping.get(tag_dict["tag"], None)
                if tag_dict["label"] not in labels_list:
                    labels_list.append(tag_dict["label"])
            # 对labels_list进行排序
            labels_list.sort()
            # 更新instruction_id
            self.instruction_id[i] = (tags, labels_list, index)
            i+=1
        print(self.instruction_id[0])
        print("***********聚类日志***************")
        print("\n")
        return labels


    # 8.关联聚类
    # 论文方法：采用FP-Growth算法挖掘标签之间的关联规则，然后根据上述关联规则递归合并关联标签，减少了冗余
    def fp_growth(self):
        # 对于instruction_id的label进行FP-Growth
        df = pd.DataFrame(self.instruction_id,columns=['tags','labels','index'])
        result = fpgrowth(df, min_support=0.1, use_colnames=True)
        return result

    # 根据tags对于instruction进行聚类抽样
    def cluster_instruction_by_tags(self,tag_count):
        pass

    # 使用正则表达式提取'```json\n'与'\n```'之间的内容
    def extract_content(self,text):
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1)
        return None
    
    # 输入一个文本list
    @torch.no_grad()
    def bert_embedding(self,texts,batch=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        model = AutoModel.from_pretrained(self.bert_model_path).to(device)
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

    # 使用phrasebert进行embedding
    def phrasebert_embedding(self,texts,batch=100):
        model = SentenceTransformer(self.phrasebert_model_path)
        phrase_embs = model.encode(texts, show_progress_bar=True)
        phrase_embs = np.array(phrase_embs)
        return phrase_embs

    # 将tag_count中的tag与instruction进行关联
    def tag_instruction_association(self,tag_count,instruction_data):
        pass

        

if __name__ == '__main__':
    tag_cluster = TAG_CLUSTER()
    tag_cluster.execute()





























    




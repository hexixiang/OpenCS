import json
import re
import os
import argparse
from sklearn.cluster import DBSCAN, KMeans
from nltk.stem import PorterStemmer
import torch
from transformers import BertTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

def default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class TagCluster:
    def __init__(self, dataset_type="alpaca", dataset="", bert_model_path="res/bert-base-uncased", 
                 phrasebert_model_path="res/phrase-bert", alpha=20, embedding_type="phrasebert", 
                 methods='kmeans', n_clusters=50, eps=0.5, min_samples=5):
        self.dataset_type = dataset_type    # dolly_15K, alpaca, Wizard_70K
        self.dataset = dataset
        self.raw_data_path = os.path.join(rf'data\{self.dataset_type}\raw_tag_data', self.dataset)
        self.processed_tag_save_path = rf'data\{self.dataset_type}\processed_tag_data'
        self.bert_model_path = bert_model_path
        self.embedding_path = rf'data\{self.dataset_type}\total_embeddings_phrasebert_alpha_{alpha}.npy'
        self.phrasebert_model_path = phrasebert_model_path
        self.set_dataset_path(dataset_type=self.dataset_type)
        self.alpha = alpha
        self.embedding_type = embedding_type # bert, phrasebert
        self.methods = methods # kmeans, dbscan
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.instruction_id = []
        self.label_tag_mapping = {}

    def execute(self):
        # Read data from jsonl
        data = self.read_data()
        # Remove data that does not meet the format and add to bad_case
        self.normalize_tag_format(data)
        # Get all good_case_tag_data tags and corresponding explanations
        self.instruction_id = self.get_good_case_tag_data()
        # Count the number of all tags in good_case_tag_data
        tag_count = self.count_tag_frequency(self.instruction_id)
        # Filter tags based on frequency
        tag_count = self.filter_tag_by_frequency(tag_count, alpha=self.alpha)
        # Filter based on rules
        tag_count = self.filter_tag_by_rule(tag_count)
        # Semantic aggregation
        text_embedding = self.embed_tags(tag_count)
        self.cluster_text_embeddings(text_embedding, methods=self.methods, n_clusters=self.n_clusters, eps=self.eps, min_samples=self.min_samples)
        # Write instruction_id to file
        existing_labels = set() 

        with open(rf"src/tag_based_cluster/instruction_id_{self.dataset_type}.json", "w", encoding="utf-8") as f:
            for item in self.instruction_id:
                labels_tuple = tuple(item[1])
                
                json_data = json.dumps(
                    [{
                        "tags": item[0],
                        "labels": item[1],
                        "index": item[2],
                        "labels_tuple": labels_tuple
                    }],
                    ensure_ascii=False, default=default
                )
                f.write(json_data + "\n")
                if labels_tuple not in existing_labels:
                    existing_labels.add(labels_tuple)              
        print(f"The final dataset length is {len(existing_labels)}")

    def set_dataset_path(self, dataset_type):
        if self.dataset_type == "alpaca":
            self.dataset = r"alpaca_tag_total_with_index.jsonl"
        elif self.dataset_type == "dolly":
            self.dataset = r"dolly_15k_tag_total_with_index.jsonl"
        elif self.dataset_type == "wizard":
            self.dataset = r"wizardlm-70k_tag_total_with_index.jsonl"
        else:
            print("No such dataset")
            raise TypeError

    def read_data(self):
        data = []
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def normalize_tag_format(self, data):
        
        def parse_json_key(key):
            try:
                return json.loads(self.extract_content(key))
            except Exception:
                return json.loads(key)
        
        good_case = []
        bad_case = []
        for item in data:
            try:
                item["tags"] = parse_json_key(item["key_0"])
                del item["key_0"]
                good_case.append(item)
            except Exception as e:
                print(e)
                bad_case.append(item)

        with open(os.path.join(self.processed_tag_save_path, self.dataset.split('.')[0]+'_good'+"."+self.dataset.split('.')[1]), 'w', encoding='utf-8') as f:
            for item in good_case:
                f.write(json.dumps(item) + '\n')

        with open(os.path.join(self.processed_tag_save_path, self.dataset.split('.')[0]+'_bad'+"."+self.dataset.split('.')[1]), 'w', encoding='utf-8') as f:
            for item in bad_case:
                f.write(json.dumps(item) + '\n')

    def get_good_case_tag_data(self):
        good_case_tag_data = []
        with open(os.path.join(self.processed_tag_save_path, self.dataset.split('.')[0]+'_good'+"."+self.dataset.split('.')[1]), 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                good_case_tag_data.append((item["tags"], item["index"]))
        tag_counts = [len(item[0]) for item in good_case_tag_data]
        avg_tag_count = sum(tag_counts) / len(tag_counts)
        max_tag_count = max(tag_counts)
        min_tag_count = min(tag_counts)
        print("*********** Initial Log ***************")
        print(f"The original length of instruction_id is {len(good_case_tag_data)}")
        print(f"Each data entry has an average of {avg_tag_count} tags, with a maximum of {max_tag_count} and a minimum of {min_tag_count}")
        print("*********** Initial Log ***************")
        print("\n")
        return good_case_tag_data

    def count_tag_frequency(self, good_case_tag_data):
        tag_count = {}  
        for item in good_case_tag_data:
            for tag in item[0]:
                if tag["tag"] in tag_count:
                    tag_count[tag["tag"]] += 1
                else:
                    tag_count[tag["tag"]] = 1
        print("*********** Count Log ***************")
        print(f"The original number of different tags is {len(tag_count)}")
        max_count = max(tag_count.values())
        min_count = min(tag_count.values())
        avg_count = sum(tag_count.values()) / len(tag_count)
        print(f"The maximum occurrence of a tag is {max_count}, the minimum is {min_count}, and the average is {avg_count}")
        print("*********** Count Log ***************")
        print("\n")
        return tag_count

    def filter_tag_by_frequency(self, tag_count, alpha=None):
        for tag in list(tag_count.keys()):
            if tag_count[tag] < alpha:
                del tag_count[tag]
        self.instruction_id = [
            (tags, index) for tags, index in self.instruction_id
            if any(tag_dict["tag"] in tag_count for tag_dict in tags)
        ]
        self.instruction_id = [
            ([tag_dict for tag_dict in tags if tag_dict["tag"] in tag_count], index)
            for tags, index in self.instruction_id
        ]
        print("*********** Frequency-Based Filtering Log ***************")
        print(f"The length of instruction_id after frequency-based filtering is {len(self.instruction_id)}")
        print(f"The number of tags after frequency-based filtering is {len(tag_count)}, alpha is {alpha}")
        print("*********** Frequency-Based Filtering Log ***************")
        print("\n")
        return tag_count

    def filter_tag_by_rule(self, tag_count):
        ps = PorterStemmer()
        tag_mapping = {}

        for tag in list(tag_count.keys()):
            # Convert tags to lowercase
            new_tag = tag.lower()
            # Replace all special characters with spaces
            new_tag = re.sub(r'\W+', ' ', new_tag)
            # Apply stemming
            new_tag = ps.stem(new_tag)
            tag_mapping[tag] = new_tag
            tag_count[new_tag] = [tag, tag_count.pop(tag)]

        self.instruction_id = [
            ([{"tag": tag_mapping.get(tag_dict["tag"], tag_dict["tag"]), "tag_origin": tag_dict["tag"], "explanation": tag_dict["explanation"]} for tag_dict in tags], index)
            for tags, index in self.instruction_id
        ]
        print("*********** Rule-Based Filtering Log ***************")
        print(f"The number of tags after rule-based filtering is {len(tag_count)}")
        print(f"The length of instruction_id after rule-based filtering is {len(self.instruction_id)}")
        print("*********** Rule-Based Filtering Log ***************")
        print("\n")
        return tag_count

    def embed_tags(self, tag_count):
        print("*********** Semantic Clustering Log ***************")
        tag_list = [tag_dict["tag"] for tags, _ in self.instruction_id for tag_dict in tags]

        if os.path.exists(self.embedding_path):
            text_embedding = np.load(self.embedding_path)
        else:
            print("Embedding not exist, start embedding")
            if self.embedding_type == "bert":
                text_embedding = self.bert_embedding(tag_list)
            elif self.embedding_type == "phrasebert":
                text_embedding = self.phrasebert_embedding(tag_list)
            np.save(self.embedding_path, text_embedding)
        print("Semantic clustering completed!")
        print("*********** Semantic Clustering Log ***************")
        print("\n")
        return text_embedding

    def cluster_text_embeddings(self, text_embeddings, methods='kmeans', eps=0.5, min_samples=5, n_clusters=None):
        print("*********** Clustering Log ***************")
        if methods == 'dbscan':
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        elif methods == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters)
        print(f"Clustering method is {methods}")
        if methods == 'dbscan':
            labels = dbscan.fit_predict(text_embeddings)
        elif methods == 'kmeans':
            labels = kmeans.fit_predict(text_embeddings)

        tag_list = [tag_dict["tag"] for tags, _ in self.instruction_id for tag_dict in tags]
        tag_label_mapping = dict(zip(tag_list, labels))
        self.label_tag_mapping = {}
        for i in range(len(labels)):
            if labels[i] not in self.label_tag_mapping:
                self.label_tag_mapping[labels[i]] = [tag_list[i]]
            else:
                self.label_tag_mapping[labels[i]].append(tag_list[i])
        i = 0
        for tags, index in self.instruction_id:
            labels_list = []
            for tag_dict in tags:
                tag_dict["label"] = tag_label_mapping.get(tag_dict["tag"], None)
                if tag_dict["label"] not in labels_list:
                    labels_list.append(tag_dict["label"])
            labels_list.sort()
            self.instruction_id[i] = (tags, labels_list, index)
            i += 1
        print(self.instruction_id[0])
        print("*********** Clustering Log ***************")
        print("\n")
        return labels

    # Use regular expressions to extract content between '```json\n' and '\n```'
    def extract_content(self, text):
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1)
        return None
    
    @torch.no_grad()
    def bert_embedding(self, texts, batch=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        model = AutoModel.from_pretrained(self.bert_model_path).to(device)
        encoded_texts = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=96)
        encoded_texts = encoded_texts.to(device)
        cls_hid_li = []
        for i in tqdm(range(0, len(texts), batch)):
            last_hids = model(input_ids=encoded_texts["input_ids"][i:i+batch],
                            attention_mask=encoded_texts["attention_mask"][i:i+batch])['last_hidden_state']
            cls_hids = last_hids[:,0,:].squeeze()
            cls_hid_li.append(cls_hids)
            torch.cuda.empty_cache()
        cls_hids_tensor = torch.concat(cls_hid_li, dim=0)
        return np.array(cls_hids_tensor.cpu())

    # use phrasebert for embedding
    def phrasebert_embedding(self, texts, batch=100):
        model = SentenceTransformer(self.phrasebert_model_path)
        phrase_embs = model.encode(texts, show_progress_bar=True)
        phrase_embs = np.array(phrase_embs)
        return phrase_embs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TagCluster arguments")
    parser.add_argument('--dataset_type', type=str, default="alpaca", help='Dataset type: dolly_15K, alpaca, Wizard_70K')
    parser.add_argument('--dataset', type=str, default="", help='Dataset file name')
    parser.add_argument('--bert_model_path', type=str, default="res/bert-base-uncased", help='Path to BERT model')
    parser.add_argument('--phrasebert_model_path', type=str, default="res/phrase-bert", help='Path to PhraseBERT model')
    parser.add_argument('--alpha', type=int, default=20, help='Alpha value for filtering')
    parser.add_argument('--embedding_type', type=str, default="phrasebert", help='Embedding type: bert, phrasebert')
    parser.add_argument('--methods', type=str, default='kmeans', help='Clustering method: kmeans, dbscan')
    parser.add_argument('--n_clusters', type=int, default=50, help='Number of clusters for kmeans')
    parser.add_argument('--eps', type=float, default=0.5, help='Epsilon value for DBSCAN')
    parser.add_argument('--min_samples', type=int, default=5, help='Minimum samples for DBSCAN')

    args = parser.parse_args()

    tag_cluster = TagCluster(
        dataset_type=args.dataset_type,
        dataset=args.dataset,
        bert_model_path=args.bert_model_path,
        phrasebert_model_path=args.phrasebert_model_path,
        alpha=args.alpha,
        embedding_type=args.embedding_type,
        methods=args.methods,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples
    )
    tag_cluster.execute()
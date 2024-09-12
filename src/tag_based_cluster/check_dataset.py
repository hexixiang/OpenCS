# 对于构造好的tag_cluster_dataset进行统计
import json
import matplotlib.pyplot as plt

# 统计维度包括：
# 不重复tag的总数量
# 出现最多，最少次数的tag数目
# 每条指令包括的tag数目分布（直方图）
# 平均tag数量
# 最大最小tag数

tag_count_dict = {}
tag_counts = []

with open(r'results\tag_cluster_result_k_50_alpha_20\instruction_id_5047_k_50_updated.json', 'r', encoding='utf-8') as f:
    for line in f:
        content = json.loads(line)
        tags = content['tags']
        tag_counts.append(len(tags))
        index = content['index']
        for tag in tags:
            tag = tag['tag']
            if tag not in tag_count_dict:
                tag_count_dict[tag] = 0
            tag_count_dict[tag] += 1


# 先统计tag的指标
print(f'不同tag的总数量: {len(tag_count_dict)}')
print(f'出现最多，最少次数的tag数目: {max(tag_count_dict.values())}, {min(tag_count_dict.values())}')
# tag平均出现次数
print(f'tag平均出现次数: {round(sum(tag_count_dict.values()) / len(tag_count_dict), 2)}')

# 每条指令包含tag的数量最大，最小，平均值（保留两位小数）
print(f'每条指令包含tag的数量最大，最小值: {max(tag_counts)}, {min(tag_counts)}')
print(f'每条指令包含tag的数量平均值: {round(sum(tag_counts) / len(tag_counts), 2)}')
min_tag_count = min(tag_counts)
max_tag_count = max(tag_counts)
# 指令包含tag的数量直方图
plt.hist(tag_counts, bins=range(min_tag_count, max_tag_count + 2), edgecolor='black', align='left')
plt.xlabel('tag_appearrance_time')
plt.ylabel('data_count')
plt.title('tag_appearrance_time_distribution')
plt.xticks(range(min_tag_count, max_tag_count + 1))

# 获取直方图的柱子和边界
counts, bins, patches = plt.hist(tag_counts, bins=range(min_tag_count, max_tag_count + 2), edgecolor='black', align='left')

# 在每个柱子上标注出现次数
for count, bin in zip(counts, bins):
    if count >= 0:  # 只标注非零的柱子
        plt.text(bin, count, str(int(count)), ha='center', va='bottom')

plt.show()

# 直方图画出每条指令包括的tag数目分布












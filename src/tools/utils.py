# 基本函数

# 统计文件行数
def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f)
import requests
import random
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_embeddings():
    with open('write_task_synthesis.jsonl', 'r', encoding='utf-8') as f:
        write_task_data = [json.loads(line)['response'] for line in f]

    url = "https://api.siliconflow.cn/v1/embeddings"

    for i in range(len(write_task_data)):
        print(i + 1)
        payload = {
            "model": "Qwen/Qwen3-Embedding-8B",
            "input": write_task_data[i]
        }
        headers = {
            "Authorization": "Bearer sk-fdvbhjujrlslkaowvmwgumqppjeogbqwrjvwwzwilajhcjki",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers).json()
        with open('write_task_embeddings.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(response, ensure_ascii=False) + '\n')


def dedup():
    # 读取embeddings
    embeds = []
    with open('write_task_embeddings.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            try:
                embeds.append(data['data'][0]['embedding'])
            except Exception:
                print(data)
                print(i)
                exit()
    embeddings = np.array(embeds)
    print(embeddings.shape)  # (15900, 4096)
    SIMILARITY_THRESHOLD = 0.5

    print("开始计算相似度矩阵...")
    
    # 1. 计算所有向量两两之间的余弦相似度矩阵
    # 使用 sklearn.metrics.pairwise.cosine_similarity 更高效
    # 结果是一个 (15900, 15900) 的矩阵
    # 请确保您已经导入了 from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    print("相似度矩阵计算完成。")
    print(f"相似度矩阵形状: {similarity_matrix.shape}")
    
    # 2. 初始化用于去重的变量
    N = embeddings.shape[0]
    # 用于标记哪些数据已经被保留（True）或丢弃（False）
    is_kept = np.full(N, True) 
    # 存储最终保留数据的原始索引
    kept_indices = []
    
    print("开始执行去重逻辑...")
    
    # 3. 遍历所有数据，执行贪婪去重
    for i in range(N):
        if is_kept[i]:
            # 如果当前数据 i 还没有被标记为丢弃，则保留它
            kept_indices.append(i)
            
            # 找到所有与当前数据 i 相似度超过阈值的数据 j
            # j > i 是为了只看矩阵的上三角部分，避免重复比较和将 i 自己标记为丢弃
            # 我们只需要检查 i 之后的元素，因为 i 之前的元素已经处理过了
            # 并且如果 i 相似于 k，则 k 在处理时会丢弃 i。
            # 这里我们查找所有与 i 相似的 '未来' 元素，并将它们标记为丢弃。
            # 遍历 j 从 i+1 到 N
            # similarity_matrix[i, j] 是 i 和 j 的相似度
            
            # 获取当前行中，索引大于 i 且相似度大于阈值的索引 j
            # numpy.where 返回一个元组，我们需要第一个元素（即索引数组）
            duplicate_indices = np.where(
                (similarity_matrix[i, i+1:] >= SIMILARITY_THRESHOLD) 
            )[0]
            
            # 因为我们取的是 i+1: 范围内的索引，所以需要加上 i+1 才是它们在整个数据集中的实际索引
            actual_duplicate_indices = duplicate_indices + (i + 1)
            
            # 将这些重复项标记为 False (丢弃)
            is_kept[actual_duplicate_indices] = False

    # 读取write_task_synthesis
    all_data = []
    with open('write_task_synthesis.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        all_data.append(json.loads(line)['response'])
    
    kept_data = [all_data[i] for i in kept_indices]
    
    # 3. 打乱数据的顺序
    print("打乱数据顺序...")
    random.shuffle(kept_data)
    
    # 4. 保存到新的 JSONL 文件
    output_filename = 'write_task_synthesis_deduped_shuffled.jsonl'
    print(f"保存数据到 {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for item in kept_data:
            # 将字典 item 转换回 JSON 字符串，并写入文件，每行一个
            f.write(json.dumps({"instruction": item}, ensure_ascii=False) + '\n')
            
    print(f"数据保存成功! 文件: {output_filename}, 包含 {len(kept_data)} 条记录。")

    return kept_indices

if __name__ == '__main__':
    # get_embeddings()  # OK
    dedup()

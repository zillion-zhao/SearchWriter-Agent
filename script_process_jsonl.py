import json
import os

def transform_jsonl(input_filepath: str, output_filepath: str):
    """
    读取 JSONL 文件，将 'response' 键转换为 'instruction'，并移除 'index' 键。

    Args:
        input_filepath: 输入 JSONL 文件的路径。
        output_filepath: 输出 JSONL 文件的路径。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_filepath):
        print(f"❌ 错误：输入文件未找到 -> {input_filepath}")
        return

    processed_count = 0
    
    print(f"▶️ 开始处理文件: {input_filepath}")
    
    # 使用 'w' 模式打开输出文件，如果文件已存在则会覆盖
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        # 逐行读取输入文件
        for line in infile:
            try:
                # 1. 解析 JSON 行
                data = json.loads(line.strip())
                
                # 2. 提取 'response' 字段并重命名为 'instruction'
                # 这里的逻辑是确保 'response' 键存在，如果不存在则跳过该行或报错
                if "response" in data:
                    new_data = {
                        "instruction": data["response"]
                    }
                    
                    # 3. 将新的字典对象序列化为 JSON 字符串并写入输出文件
                    # ensure_ascii=False 确保中文字符正确显示
                    json_line = json.dumps(new_data, ensure_ascii=False)
                    outfile.write(json_line + '\n')
                    processed_count += 1
                else:
                    print(f"⚠️ 警告：跳过一行缺少 'response' 键的数据: {line.strip()[:50]}...")

            except json.JSONDecodeError:
                # 处理非法的 JSON 行
                print(f"❌ 错误：跳过一行无效的 JSON 数据: {line.strip()[:50]}...")
            except Exception as e:
                print(f"❌ 发生未知错误: {e}")

    print(f"✅ 处理完成！共转换了 {processed_count} 行数据。")
    print(f"🚀 新文件已保存至: {output_filepath}")


if __name__ == "__main__":
    # 假设您的输入文件名为 input.jsonl
    INPUT_FILE = "new_syn.jsonl"
    # 输出文件名为 output.jsonl
    OUTPUT_FILE = "output.jsonl"
    transform_jsonl(INPUT_FILE, OUTPUT_FILE)
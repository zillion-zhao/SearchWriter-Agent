import os
import time
import json
import http.client
from openai import OpenAI
from datetime import datetime
from google.genai import Client
from google.genai import types
from langdetect import detect
from config import *

# -----------------------------------------------------------------LLM API Utils-----------------------------------------------------------------

def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


def load_prompt_from_file(file_path):
    return open(file_path, 'r', encoding='utf-8').read()


def get_start_idx(file_name):
    if not os.path.exists(file_name):
        return 0
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return len(lines)


# 当stage=plan (长度16000) /refine (长度64000) / other (未明确限制长度) 时为正常生成，当stage=write时遇到<END_SERACH_QUERY>停止
def openai_generate(client: OpenAI, query, stage='plan', model="deepseek-ai/DeepSeek-V3"):
    max_retries = 3
    for t in range(max_retries):
        try:
            if stage == 'plan':
                response = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': query}],
                    stream=False,
                    max_completion_tokens=16000,
                    temperature=0.5,
                    top_p=0.8
                )
            elif stage == 'write':
                response = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': query}],
                    stream=False,
                    max_completion_tokens=16000,
                    temperature=0.5,
                    top_p=0.8,
                    stop=['<END_SEARCH_QUERY>', '</END_SEARCH_QUERY>']
                )
            elif stage == 'refine':
                response = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': query}],
                    stream=False,
                    max_completion_tokens=64000,
                    temperature=0.5,
                    top_p=0.8
                )
            elif stage == 'other':
                response = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': query}],
                    stream=False,
                    temperature=0.5,
                    top_p=0.8
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {t + 1} failed: {str(e)}")
            time.sleep(3)
    return ""


def gemini_generate(client: Client, query, stage='plan', model="gemini-2.5-flash"):
    max_retries = 3
    for t in range(max_retries):
        try:
            if stage == 'plan':
                response = client.models.generate_content(
                    model=model,
                    contents=query,
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        top_p=0.8,
                        max_output_tokens=16000
                    )
                )
            elif stage == 'write':
                response = client.models.generate_content(
                    model=model,
                    contents=query,
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        top_p=0.8,
                        max_output_tokens=16000,
                        stop_sequences=['<END_SEARCH_QUERY>', '</END_SEARCH_QUERY>']
                    )
                )
            elif stage == 'refine':
                response = client.models.generate_content(
                    model=model,
                    contents=query,
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        top_p=0.8,
                        max_output_tokens=64000
                    )
                )
            elif stage == 'other':
                response = client.models.generate_content(
                    model=model,
                    contents=query,
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        top_p=0.8
                    )
                )
            return response.text
        except Exception as e:
            print(f"Attempt {t + 1} failed: {str(e)}")
            time.sleep(3)
    return ""


def openai_batch_generate(client, query_set, model, save_path):
    start_idx = get_start_idx(save_path)

    for i in range(start_idx, len(query_set)):
        print(i + 1)
        prompt = query_set[i]
        response = openai_generate(client, prompt, model=model)
        with open(save_path, 'a', encoding='utf-8') as f:
            json_data = {"index": i + 1, "response": response}
            json.dump(json_data, f)
            f.write('\n')


def gemini_batch_generate(client, query_set, model, save_path):
    start_idx = get_start_idx(save_path)

    for i in range(start_idx, len(query_set)):
        print(i + 1)
        prompt = query_set[i]
        response = gemini_generate(client, prompt, model=model)
        with open(save_path, 'a', encoding='utf-8') as f:
            json_data = {"index": i + 1, "response": response}
            json.dump(json_data, f)
            f.write('\n')


# -----------------------------------------------------------------Search Utils----------------------------------------------------------------

class MitaSearch:
    def __init__(self) -> None:
        self.conn = http.client.HTTPSConnection("metaso.cn")

    def mita_search(query):
        # 秘塔网络搜索
        conn = http.client.HTTPSConnection("metaso.cn")
        payload = json.dumps({"q": query, "scope": "webpage", "includeSummary": False, "size": "10", "includeRawContent": False, "conciseSnippet": False})
        headers = {
            'Authorization': 'Bearer mk-5B7900DB1721A46334E9A9F30265A60B',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/api/v1/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(data.decode("utf-8"))
        return data.decode("utf-8")

    def mita_parse(url):
        # 秘塔网页内容读取
        conn = http.client.HTTPSConnection("metaso.cn")
        payload = json.dumps({"url": url})
        headers = {
            'Authorization': 'Bearer mk-5B7900DB1721A46334E9A9F30265A60B',
            'Accept': 'text/plain',
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/api/v1/reader", payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(data.decode("utf-8"))
        return data.decode("utf-8")


class GoogleSearch:
    def __init__(self) -> None:
        self.google_search_prompt_cn = load_prompt_from_file('./prompts/google_search_cn.txt')
        self.google_search_prompt = load_prompt_from_file('./prompts/google_search_en.txt')
        self.genai_client = Client(api_key=GEMINI_KEY)
    
    def search(self, query, model="gemini-2.5-flash", lang="cn"):
        if lang == "cn":
            prompt = self.google_search_prompt_cn
        else:
            prompt = self.google_search_prompt
        current_date = str(get_current_date())
        formatted_prompt = prompt.replace("<CURRENT_DATE>", current_date).replace("<RESEARCH_TOPIC>", query)
        # send request
        max_retries = 10
        for t in range(max_retries):
            try:
                response = self.genai_client.models.generate_content(
                    model=model,
                    contents=formatted_prompt,
                    config={
                        "tools": [{"google_search": {}}],
                        "temperature": 0.2,
                    },
                )
                return response.candidates[0].content.parts[-1].text
            except Exception as e:
                print(f"Attempt {t + 1} failed: {str(e)}")
                time.sleep(3)
        return ""


# -----------------------------------------------------------------Data Utils-----------------------------------------------------------------

# from huggingface_hub import snapshot_download

# # 从Huggingface下载模型和数据
# snapshot_download(repo_id="xlangai/BRIGHT", repo_type="dataset", # {'dataset', 'model'}
#                   local_dir="BRIGHT",
#                   local_dir_use_symlinks=False, resume_download=True,
#                   token='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',
#                   endpoint='https://hf-mirror.com')  # 如果不能翻墙，可以添加这个参数，从而在hf-mirror上下载（不需要翻墙，默认huggingface需要外网）

class DataLoader:
    def __init__(self) -> None:
        self.writingbench_path = 'benchmarks/WritingBench/benchmark_query/benchmark_all.jsonl'
        self.search_write_task_path = 'benchmarks/SearchWriteBench.jsonl'

    def load_writingbench(self):
        """
        index, domain1, domain2, lang, query, checklist (用于评价)
        """
        with open(self.writingbench_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
        return records

    def load_search_write_task(self):
        with open(self.search_write_task_path, 'r', encoding='utf-8') as f:
            search_write_task_data = [json.loads(line)['instruction'] for line in f]
        return search_write_task_data


# -----------------------------------------------------------------Other Utils-----------------------------------------------------------------
def detect_language(text):
    if not text or not text.strip():
        return "en"
    try:
        # detect() 函数返回语言的 ISO 639-1 代码 (如 'en', 'zh', 'ja')
        lang_code = detect(text)
        
        if lang_code in ['zh-cn', 'zh-tw']:
            return 'cn'
        else:
            return 'en'
    except Exception:
        # 如果文本太短或包含太多无法识别的字符，可能会抛出异常，这时返回en即可
        return 'en'

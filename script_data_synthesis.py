import random
import json
from config import *
from utils import *


def search_write_task_synthesis(prompt, save_path):
    # 调用API来为每个bench进行inference
    gemini_llm_client = Client(api_key=GEMINI_KEY)

    in_len = ['0-10', '0-10', '0-10', '0-10', '0-10', '10-30', '10-30', '10-30', '30-60', '30-60', '30-60', '60-100', '60-100', '100-200', '100-200', '200-500', '500-1000', '1000-2000']
    lang = ['an English', 'a Chinese']

    persona = []
    with open('persona.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        persona.append(json.loads(line)['persona'])
    print(len(persona))

    for i in range(106300, 107000):
        print(i - 106300 + 1)
        p = persona[i]
        in_l = random.choice(in_len)
        la = random.choice(lang)
        in_prompt = prompt.replace('[ROLE]', p).replace('[LANGUAGE]', la).replace('[INPUT_LEN]', in_l)
        print(in_prompt)
        response = gemini_generate(gemini_llm_client, in_prompt, stage='plan', model='gemini-2.5-pro')
        with open(save_path, 'a', encoding='utf-8') as f:
            json_data = {"index": i, "response": response}
            f.write(json.dumps(json_data) + '\n')


def write_task_synthesis(prompts, save_path):
    dl = DataLoader()

    # (1) 读取WritingBench
    writingbench_data = dl.load_writingbench()
    writingbench_queries = [x['query'] for x in writingbench_data]
    print(len(writingbench_queries))

    # (2) 读取HelloBench
    hellobench_data = dl.load_hellobench()
    hellobench_chat_queries = [x['instruction'] for x in hellobench_data['chat']]
    hellobench_htg_queries = [x['instruction'] for x in hellobench_data['htg']]
    hellobench_qa_queries = [x['instruction'] for x in hellobench_data['qa']]
    hellobench_sum_queries = [x['instruction'] for x in hellobench_data['sum']]
    hellobench_comp_queries = [x['instruction'] for x in hellobench_data['comp']]

    hellobench_queries = hellobench_chat_queries + hellobench_htg_queries + hellobench_qa_queries + hellobench_sum_queries + hellobench_comp_queries
    print(len(hellobench_queries))

    examples = hellobench_queries + writingbench_queries

    # 调用API来为每个bench进行inference
    gemini_llm_client = Client(api_key=GEMINI_KEY)
    uiui_llm_client = OpenAI(api_key=UIUI_KEY, base_url="https://sg.uiuiapi.com/v1")
    silicon_llm_client = OpenAI(api_key=SILICON_KEY, base_url="https://api.siliconflow.cn/v1")

    lang = ['an English', 'a Chinese']

    persona = []
    with open('persona.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        persona.append(json.loads(line)['persona'])
    print(len(persona))
    persona = random.sample(persona, 20000)  # 只随机选取部分persona

    for i in range(5484, len(persona)):
        print(i + 1)
        p = persona[i]
        la = random.choice(lang)
        prompt = random.choice(prompts)
        prompt = prompt.replace('[ROLE]', p).replace('[LANGUAGE]', la)
        ex = random.sample(examples, 3)
        if '[EXAMPLE1]' in prompt:
            print('Using example 1')
            prompt = prompt.replace('[EXAMPLE1]', ex[0])
        if '[EXAMPLE2]' in prompt:
            print('Using example 2')
            prompt = prompt.replace('[EXAMPLE2]', ex[1])
        if '[EXAMPLE3]' in prompt:
            print('Using example 3')
            prompt = prompt.replace('[EXAMPLE3]', ex[2])

        response = gemini_generate(gemini_llm_client, prompt, stage='plan', model='gemini-2.5-flash')
        with open(save_path, 'a', encoding='utf-8') as f:
            json_data = {"index": i + 1, "response": response}
            json.dump(json_data, f)
            f.write('\n')


if __name__ == '__main__':
    # 合成需要搜索的写作任务指令
    search_write_task_prompt = """Suppose that you are [ROLE]. Please devise a user query that deliver a writing task that should necessitate multiple Internet **search** behaviours. Please directly generate an **[LANGUAGE]** query with the length of about [INPUT_LEN] words and do not generate anything else."""
    search_write_task_synthesis(search_write_task_prompt, 'new_syn.jsonl')  # OK

    # 合成通用写作任务指令（利用好现在的数据集）
    w0 = """Suppose that you are [ROLE]. Please devise a user query that deliver a writing task. Please directly generate **[LANGUAGE]** query without anything else."""

    w1 = """Suppose that you are [ROLE]. Please devise a user query that deliver a writing task. Please directly generate **[LANGUAGE]** query without anything else.

Here is an example of user query that you can refer to:

[EXAMPLE1]"""

    w2 = """Suppose that you are [ROLE]. Please devise a user query that deliver a writing task. Please directly generate **[LANGUAGE]** query without anything else.
    
Here are some examples of user query that you can refer to:

Example 1:
[EXAMPLE1]

Example 2:
[EXAMPLE2]"""

    w3 = """Suppose that you are [ROLE]. Please devise a user query that deliver a writing task. Please directly generate **[LANGUAGE]** query without anything else.

Here are some examples of user query that you can refer to:

Example 1:
[EXAMPLE1]

Example 2:
[EXAMPLE2]

Example 3:
[EXAMPLE3]"""

#     write_task_synthesis([w0, w0, w1, w1, w2, w3], 'write_task_synthesis.jsonl')

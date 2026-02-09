import re
import json
import time
from config import *
from utils import *


class SearchWriterAgent:
    def __init__(self) -> None:
        self.outline_generation_prompt_cn = load_prompt_from_file('./prompts/outline_generation_cn.txt')
        self.outline_generation_prompt = load_prompt_from_file('./prompts/outline_generation_en.txt')
        self.write_prompt_cn = load_prompt_from_file('./prompts/write_cn.txt')
        self.write_prompt = load_prompt_from_file('./prompts/write_en.txt')
        self.write_nosearch_prompt_cn = load_prompt_from_file('./prompts/write_nosearch_cn.txt')
        self.write_nosearch_prompt = load_prompt_from_file('./prompts/write_nosearch_en.txt')
        self.refine_prompt_cn = load_prompt_from_file('./prompts/refine_cn.txt')
        self.refine_prompt = load_prompt_from_file('./prompts/refine_en.txt')

        self.silicon_llm_client = OpenAI(api_key=SILICON_KEY, base_url="https://api.siliconflow.cn/v1")
        self.uiui_llm_client = OpenAI(api_key=UIUI_KEY, base_url="https://sg.uiuiapi.com/v1")
        self.gemini_llm_client = Client(api_key=GEMINI_KEY)
        self.search_client = GoogleSearch()

    def outline_generation(self, inst, lang="cn"):
        if lang == "cn":
            prompt = self.outline_generation_prompt_cn
        else:
            prompt = self.outline_generation_prompt
        prompt = prompt.replace("<USER_QUESTION>", inst)
        # ------------------------- plan阶段 调用LLM API -------------------------
        # return openai_generate(self.silicon_llm_client, prompt, stage='plan', model='deepseek-ai/DeepSeek-V3')
        # return openai_generate(self.uiui_llm_client, prompt, stage='plan', model='gpt-4o')
        return gemini_generate(self.gemini_llm_client, prompt, stage='plan', model='gemini-2.5-pro')

    def google_search(self, q, lang="cn"):
        return self.search_client.search(q, model="gemini-2.5-flash", lang=lang)

    def wide_search(self, text, lang="cn"):
        print('-------------------- 正在执行 Wide Search --------------------')
        pattern = r'(<BEGIN_SEARCH_QUERY>)(.+?)(<END_SEARCH_QUERY>)'

        # 使用 re.sub 进行替换，对每个匹配项执行一个函数 (replacer)
        def replacer(match):
            # match.group(0) 是完整的匹配字符串 (包括标签), match.group(2) 是捕获的查询内容
            query = match.group(2)
            # --- 执行搜索（调用google搜索API） ---
            simulated_result = self.google_search(query, lang)
            print('对query: "' + str(query) + '" 调用搜索结果完成')
            # 构造替换字符串：完整的原始匹配 + BEGIN_SEARCH_RESULT + 结果 + END_SEARCH_RESULT
            return f"{match.group(0)} <BEGIN_SEARCH_RESULT> {simulated_result} <END_SEARCH_RESULT>"

        # 执行替换
        modified_text = re.sub(pattern, replacer, text, flags=re.DOTALL)
        return modified_text

    def write_one_part(self, inst, part_outline, before_written_text, search_doc, lang="cn"):
        if lang == "cn":
            raw_prompt = self.write_prompt_cn
            nosearch_prompt = self.write_nosearch_prompt_cn
            current_written_text = '空'
        else:
            raw_prompt = self.write_prompt
            nosearch_prompt = self.write_nosearch_prompt
            current_written_text = 'EMPTY'

        writing_count = 0
        current_written_text_with_search = ''

        while True:
            if writing_count < 3:
                prompt = raw_prompt.replace('<USER_QUESTION>', inst)
                prompt = prompt.replace('<CURRENT_PART>', part_outline)
                prompt = prompt.replace('<BEFORE_WRITTEN_TEXT>', before_written_text)
                prompt = prompt.replace('<CURRENT_WRITTEN_TEXT>', current_written_text)
                prompt = prompt.replace('<SEARCH_DOC>', search_doc)
            else:
                print('\n---------- ！本段落已使用三次查询，不再抛出查询！ ----------\n')
                prompt = nosearch_prompt.replace('<USER_QUESTION>', inst)
                prompt = prompt.replace('<CURRENT_PART>', part_outline)
                prompt = prompt.replace('<BEFORE_WRITTEN_TEXT>', before_written_text)
                prompt = prompt.replace('<CURRENT_WRITTEN_TEXT>', current_written_text)
                prompt = prompt.replace('<SEARCH_DOC>', search_doc)
            print('\n---------- 目前可用的搜索结果如下 ----------\n' + search_doc)
            print('\n---------- 目前可用的搜索结果展示完毕 ----------\n')

            if writing_count == 0:
                current_written_text = ''

            # ------------------------- write阶段 调用LLM API -------------------------
            # response = openai_generate(self.silicon_llm_client, prompt, stage='write', model='deepseek-ai/DeepSeek-V3.2')
            # response = openai_generate(self.uiui_llm_client, prompt, stage='write', model='gpt-4o')
            response = gemini_generate(self.gemini_llm_client, prompt, stage='write', model='gemini-2.5-pro')
            print(response)

            writing_count += 1
            if writing_count <= 3:
                if '<BEGIN_SEARCH_QUERY>' in response:
                    written, query = response.split('<BEGIN_SEARCH_QUERY>')
                    current_written_text += written
                    print('写作停止，搜索query: ' + query)
                    search_doc = self.google_search(query, lang=lang).strip()
                    current_written_text_with_search += (response + '<END_SEARCH_QUERY> <BEGIN_SEARCH_RESULT> ' + search_doc + ' <END_SEARCH_RESULT>')
                else:
                    current_written_text += response
                    current_written_text_with_search += response
                    break
            else:
                current_written_text += response
                current_written_text_with_search += response
                break

        print('\n\n写作完毕！')
        return current_written_text, current_written_text_with_search

    def split_outline_by_part(self, text):
        # (1) 移除可能存在的 Markdown 代码块标记（如果有）
        if text.startswith("```") and text.endswith("```"):
            text = text.strip("```").strip()

        # (2) 使用正则表达式 (Part X) 作为分隔符进行分割，并保留分隔符
        parts = re.split(r'(\(Part\s\d+\))', text)

        # 清理结果：
        # 1. 移除空字符串（通常是由于文本开头或分隔符旁边的空格导致）
        # 2. 将分隔符与其后面的内容重新组合起来
        cleaned_parts = [p.strip() for p in parts if p.strip()]
        final_list = []

        # 重新组合：将 (Part X) 标记与其后的文本内容合并
        i = 0
        while i < len(cleaned_parts):
            if re.match(r'\(Part\s\d+\)', cleaned_parts[i]):
                # 找到一个 (Part X) 标记
                part_content = cleaned_parts[i]
                if i + 1 < len(cleaned_parts) and not re.match(r'\(Part\s\d+\)', cleaned_parts[i+1]):
                    # 如果下一个元素不是另一个 (Part X) 标记，则将其内容合并
                    part_content += " " + cleaned_parts[i+1]
                    i += 2
                else:
                    i += 1
                final_list.append(part_content)
            else:
                # 理论上，在正确的分割下，这里不应该有非 (Part X) 开头的元素遗留
                # 但作为安全措施，将其加入列表
                final_list.append(cleaned_parts[i])
                i += 1

        return final_list

    def plan(self, inst, save_path):
        # 判断语言
        lang = detect_language(inst)
        # plan阶段
        print('-------------------- START PLANNING --------------------\n')
        # (1) 生成outline
        original_o = self.outline_generation(inst, lang)
        print('-------------------- 初始大纲生成如下 --------------------')
        print(original_o)
        # (2) 抽取query并调用gemini google search，用返回文档doc替换query，的到最终outline
        search_o = self.wide_search(original_o, lang)
        print('\n-------------------- 融合Wide Search结果后的大纲如下 --------------------')
        print(search_o)
        search_o_list = self.split_outline_by_part(search_o)  # 将outline按照(Part X)进行分割，形成列表
        print('\n-------------------- PLANNING ENDED --------------------')
        # 保存结果到文件中
        with open(save_path, 'a', encoding='utf-8') as f:
            json_data = {"search_outline": search_o, "outline": original_o, "lang": lang}
            f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
        return search_o, search_o_list, original_o, lang

    def write(self, inst, text_save_path, search_o_list=None, lang="cn"):
        # write阶段
        print('\n-------------------- START WRITING --------------------\n')
        written_text = ''
        written_text_with_search = ''

        for i in range(len(search_o_list)):
            print('正在写作 (Part' + str(i + 1) + ')')
            outline = search_o_list[i]
            part, part_with_search = self.write_one_part(inst, outline, written_text, '空', lang=lang)
            written_text += part + '\n\n'
            written_text_with_search += part_with_search + '\n\n'
        print('\n-------------------- WRITING ENDED --------------------')
        # 两个写作结果（带搜索/不带搜索）保存到文件中
        with open(text_save_path, 'a', encoding='utf-8') as f:
            json_data = {'written_text': written_text, 'written_text_with_search': written_text_with_search, 'lang': lang}
            f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
        return written_text, written_text_with_search, lang

    def refine(self, inst, text, save_path, lang="cn"):
        if lang == "cn":
            raw_prompt = self.refine_prompt_cn
        else:
            raw_prompt = self.refine_prompt
        print('\n-------------------- START REFINING --------------------\n')
        prompt = raw_prompt.replace('<WRITTEN_TEXT>', text)
        prompt = prompt.replace('<INST>', inst)
        # ------------------------- refine阶段 调用LLM API -------------------------
        # response = openai_generate(self.silicon_llm_client, prompt, stage='refine', model='deepseek-ai/DeepSeek-V3')
        # response = openai_generate(self.uiui_llm_client, prompt, stage='refine', model='gpt-4o')
        response = gemini_generate(self.gemini_llm_client, prompt, stage='refine', model='gemini-2.5-pro')
        # response = nlpir_generate(self.nlpir_llm_client, prompt, stage='refine', model='DeepSeek-V3.1-Terminus')
        print('\n-------------------- REFINING ENDED --------------------')
        with open(save_path, 'a', encoding='utf-8') as f:
            json_data = {'refine_text': response}
            f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
        return response

    def single_test(self, inst, idx):
        print('--- 单元测试，写作任务指令如下：')
        print(inst)
        print()
        _, search_o_list, _, lang = self.plan(inst, save_path=str(idx) + '-plan.jsonl')
        _, written_text_with_search, lang = self.write(inst, text_save_path=str(idx) + '-write.jsonl', search_o_list = search_o_list, lang=lang)
        res = self.refine(inst, written_text_with_search, save_path=str(idx) + '-refine.jsonl', lang=lang)
        print('--- 单元测试写作结束！')
        return res


def load_bench():
    dl = DataLoader()

    # (1) 读取WritingBench
    writingbench_data = dl.load_writingbench()
    writingbench_queries = [x['query'] for x in writingbench_data]
    print(len(writingbench_queries))

    # (2) 读取SearchWriteTask
    search_write_bench_queries = dl.load_search_write_task()
    print(len(search_write_bench_queries))

    return writingbench_queries, search_write_bench_queries


def load_sft_dpo_queries():
    sft_insts, dpo_insts = [], []
    with open('SFT-data.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sft_insts.append(json.loads(line)['instruction'])
    with open('DPO-data.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        dpo_insts.append(json.loads(line)['instruction'])
    print(len(sft_insts), len(dpo_insts))
    return sft_insts, dpo_insts


def plan_inference(agent: SearchWriterAgent, queries, save_path, start_idx=0):
    for i in range(start_idx, len(queries)):
        print(i)
        inst = queries[i]
        agent.plan(inst=inst, save_path=save_path)
        time.sleep(1)


def load_plan_inference_res(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    search_os = []
    langs = []
    for line in lines:
        json_data = json.loads(line)
        search_os.append(json_data['outline'])
        langs.append(json_data['lang'])
    print(len(search_os), len(langs))
    return search_os, langs


def write_interence(agent: SearchWriterAgent, queries, search_os, langs, text_save_path, start_idx=0):
    for i in range(start_idx, len(queries)):
        inst = queries[i]
        search_o_list = agent.split_outline_by_part(search_os[i])
        lang = langs[i]
        agent.write(inst=inst, text_save_path=text_save_path, search_o_list=search_o_list, lang=lang)
        time.sleep(1)


def load_write_inference_res(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    written_texts_with_search, langs = [], []
    for line in lines:
        json_data = json.loads(line)
        written_texts_with_search.append(json_data['written_text_with_search'])
        langs.append(json_data['lang'])
    print(len(written_texts_with_search), len(langs))
    return written_texts_with_search, langs


def refine_inference(agent: SearchWriterAgent, queries, texts, langs, save_path):
    print(len(queries), len(texts), len(langs))
    for i in range(len(texts)):
        print(i + 1)
        agent.refine(queries[i], texts[i], lang=langs[i], save_path=save_path)
        time.sleep(2)


if __name__ == '__main__':
    agent = SearchWriterAgent()
    writingbench_queries, searchwritebench_queries = load_bench()
    # agent.single_test(writingbench_queries[330], 331)

    # -------------------- WritingBench inference (OK) --------------------
    # WritingBench用的是老plan和write的prompt，未优化，已经达到SuperWriter一致的效果，后期可以再用新的prompt重新跑一次，争取尽可能逼近LongWwriter-Zero
    # (1) plan inference (OK)
    # plan_inference(agent, writingbench_queries, 'results/SearchWriterInfer/writingbench_plan.jsonl')
    # (2) write inference (OK)
    # wb_search_os, wb_langs = load_plan_inference_res('results/SearchWriterInfer/writingbench_plan.jsonl')
    # write_interence(agent, writingbench_queries, wb_search_os, wb_langs, text_save_path='results/SearchWriterInfer/writingbench_write.jsonl')
    # (3) refine inference (OK)
    # wb_texts, wb_langs = load_write_inference_res('results/SearchWriterInfer/writingbench_write.jsonl')
    # refine_inference(agent, writingbench_queries, wb_texts, wb_langs, save_path='results/SearchWriterInfer/writingbench_refine.jsonl')

    # -------------------- SearchWriteBench inference --------------------
    # (1) plan inference (OK)
    # plan_inference(agent, searchwritebench_queries, 'results/SearchWriterInfer/searchwritebench_plan.jsonl')
    # (2) write inference (OK)
    # swb_search_os, swb_langs = load_plan_inference_res('results/SearchWriterInfer/searchwritebench_plan.jsonl')
    # write_interence(agent, searchwritebench_queries, swb_search_os, swb_langs, text_save_path='results/SearchWriterInfer/searchwritebench_write.jsonl', start_idx=649)
    # (3) refine inference (ING)
    swb_texts, swb_langs = load_write_inference_res('results/SearchWriterInfer/searchwritebench_write.jsonl')
    refine_inference(agent, searchwritebench_queries, swb_texts, swb_langs, save_path='results/SearchWriterInfer/searchwritebench_refine.jsonl')

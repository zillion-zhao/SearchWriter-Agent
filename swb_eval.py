import json
from config import *
from utils import *


class SWBEval:
    def __init__(self) -> None:
        self.prompt_path = './benchmarks/SearchWriteBench-prompts/'

        self.ce_cn_prompt = load_prompt_from_file(self.prompt_path + '0ce_cn.txt')
        self.ce_en_prompt = load_prompt_from_file(self.prompt_path + '0ce_en.txt')
        self.cc_cn_prompt = load_prompt_from_file(self.prompt_path + '0cc_cn.txt')
        self.cc_en_prompt = load_prompt_from_file(self.prompt_path + '0cc_en.txt')

        self.ed_cn_prompt = load_prompt_from_file(self.prompt_path + '1ed_cn.txt')
        self.ed_en_prompt = load_prompt_from_file(self.prompt_path + '1ed_en.txt')
        self.tc_cn_prompt = load_prompt_from_file(self.prompt_path + '2tc_cn.txt')
        self.tc_en_prompt = load_prompt_from_file(self.prompt_path + '2tc_en.txt')
        self.rc_cn_prompt = load_prompt_from_file(self.prompt_path + '3rc_cn.txt')
        self.rc_en_prompt = load_prompt_from_file(self.prompt_path + '3rc_en.txt')

        self.silicon_llm_client = OpenAI(api_key=SILICON_KEY, base_url="https://api.siliconflow.cn/v1")
        self.uiui_llm_client = OpenAI(api_key=UIUI_KEY, base_url="https://sg.uiuiapi.com/v1")
        self.gemini_llm_client = Client(api_key=GEMINI_KEY)
        self.search_client = GoogleSearch()

    def claim_extraction(self, paper):
        lang = detect_language(paper)
        if lang == 'cn':
            prompt = self.ce_cn_prompt
        else:
            prompt = self.ce_en_prompt
        inst = prompt.replace('<GENERATED_TEXT>', paper)
        claims = openai_generate(self.silicon_llm_client, inst, stage='other', model='deepseek-ai/DeepSeek-V3.2')
        return claims

    def claim_confirmation(self, paper, claims):
        lang = detect_language(paper)
        if lang == 'cn':
            prompt = self.cc_cn_prompt
        else:
            prompt = self.cc_en_prompt
        res = []
        for i in range(len(claims)):
            inst = prompt.replace('<CLAIM>', claims[i])
            r = openai_generate(self.silicon_llm_client, inst, stage='other', model='deepseek-ai/DeepSeek-V3.2')
            res.append(r)
        return res

    def information_density(self, paper):
        lang = detect_language(paper)
        if lang == 'cn':
            ed_prompt = self.ed_cn_prompt
            tc_prompt = self.tc_cn_prompt
            rc_prompt = self.rc_cn_prompt
        else:
            ed_prompt = self.ed_en_prompt
            tc_prompt = self.tc_en_prompt
            rc_prompt = self.rc_en_prompt
        ed_prompt = ed_prompt.replace('<GENERATED_TEXT>', paper)
        tc_prompt = tc_prompt.replace('<GENERATED_TEXT>', paper)
        rc_prompt = rc_prompt.replace('<GENERATED_TEXT>', paper)
        ed = openai_generate(self.silicon_llm_client, ed_prompt, stage='other', model='deepseek-ai/DeepSeek-V3.2')
        tc = openai_generate(self.silicon_llm_client, tc_prompt, stage='other', model='deepseek-ai/DeepSeek-V3.2')
        rc = openai_generate(self.silicon_llm_client, rc_prompt, stage='other', model='deepseek-ai/DeepSeek-V3.2')
        return ed, tc, rc

    def infer(self, paper_path, save_path, start_idx=0):
        # 读取paper文件
        papers = []
        with open(paper_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            papers.append(json.loads(line)['response'])
        # 逐条推理并保存
        for i in range(start_idx, len(papers)):
            print(i)
            paper = papers[i]
            ed, tc, rc = self.information_density(paper)
            claims = self.claim_extraction(paper)
            with open(save_path + '-ed.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(ed, ensure_ascii=False) + '\n')
            with open(save_path + '-tc.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(tc, ensure_ascii=False) + '\n')
            with open(save_path + '-rc.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(rc, ensure_ascii=False) + '\n')
            with open(save_path + '-claims.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(claims, ensure_ascii=False) + '\n')

    def load_ec(self, file_path):
        res = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    # 去除行首尾空格并解析 JSON
                    value = eval(line.strip()).replace('```json\n', '').replace('\n```', '')
                    if len(value) == 0:
                        data = {}
                    else:
                        data = json.loads(value)
                    res.append(data)
                except Exception:
                    continue
        return res

    def load_tc(self, file_path):
        res = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    # 去除行首尾空格并解析 JSON
                    value = eval(line.strip())
                    if len(value) == 0:
                        data = {"years": []}
                    else:
                        data = json.loads(value)
                    # 获取 years 列表
                    years = data.get("years", [])
                    # print(f"第 {line_number} 行的数据: {years}")
                    res.append(years)
            return res
        except FileNotFoundError:
            print(f"错误：找不到文件 {file_path}")
        except json.JSONDecodeError as e:
            print(f"解析 JSON 时出错: {e}")

    def load_rc(self, file_path):
        res = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    # 去除行首尾空格并解析 JSON
                    value = eval(line.strip()).replace('```json\n', '').replace('\n```', '')
                    if len(value) == 0:
                        data = {}
                    else:
                        data = json.loads(value)
                    res.append(data)
                except Exception:
                    continue
        return res

    def calc_ec(self):
        def count(ec):
            c = 0
            for i in range(len(ec)):
                if 'named_entities_list' in ec[i].keys() and 'events_list' in ec[i].keys() and 'time_list' in ec[i].keys() and 'statistics_list' in ec[i].keys():
                    c += (len(ec[i]['named_entities_list']) + len(ec[i]['events_list']) + len(ec[i]['time_list']) + len(ec[i]['statistics_list']))
            return c  / len(ec)
        r1_ec = self.load_ec('./searchwritebench-scores/DeepSeek-R1-ed.jsonl')
        v3_ec = self.load_ec('./searchwritebench-scores/DeepSeek-V3-ed.jsonl')
        flash_lite_ec = self.load_ec('./searchwritebench-scores/gemini-2.5-flash-lite-ed.jsonl')
        flash_ec = self.load_ec('./searchwritebench-scores/gemini-2.5-flash-ed.jsonl')
        pro_ec = self.load_ec('./searchwritebench-scores/gemini-2.5-pro-ed.jsonl')
        gpt35_ec = self.load_ec('./searchwritebench-scores/gpt-3.5-turbo-ed.jsonl')
        gpt4o_ec = self.load_ec('./searchwritebench-scores/gpt-4o-ed.jsonl')
        lw_ec = self.load_ec('./searchwritebench-scores/LongWriter-ed.jsonl')
        lwz_ec = self.load_ec('./searchwritebench-scores/LongWriter-Zero-ed.jsonl')
        o3_ec = self.load_ec('./searchwritebench-scores/o3-mini-ed.jsonl')
        qwen7_ec = self.load_ec('./searchwritebench-scores/Qwen2.5-7B-Instruct-ed.jsonl')
        qwen32_ec = self.load_ec('./searchwritebench-scores/Qwen2.5-32B-Instruct-ed.jsonl')
        qwen72_ec = self.load_ec('./searchwritebench-scores/Qwen2.5-72B-Instruct-ed.jsonl')
        qwq_ec = self.load_ec('./searchwritebench-scores/QwQ-32B-ed.jsonl')
        sw_ec = self.load_ec('./searchwritebench-scores/SearchWriter-ed.jsonl')

        print('gpt-3.5-turbo', len(gpt35_ec), count(gpt35_ec))
        print('gpt-4o', len(gpt4o_ec), count(gpt4o_ec))
        print('gemini-2.5-flash-lite', len(flash_lite_ec), count(flash_lite_ec))
        print('gemini-2.5-flash', len(flash_ec), count(flash_ec))
        print('gemini-2.5-pro', len(pro_ec), count(pro_ec))
        print('QWen2.5-7B-Instruct', len(qwen7_ec), count(qwen7_ec))
        print('QWen2.5-32B-Instruct', len(qwen32_ec), count(qwen32_ec))
        print('QWen2.5-72B-Instruct', len(qwen72_ec), count(qwen72_ec))
        print('DeepSeek-V3', len(v3_ec), count(v3_ec))
        print('o3-mini', len(o3_ec), count(o3_ec))
        print('QwQ-32B', len(qwq_ec), count(qwq_ec))
        print('DeepSeek-R1', len(r1_ec), count(r1_ec))
        print('LongWriter', len(lw_ec), count(lw_ec))
        print('LongWriter-Zero', len(lwz_ec), count(lwz_ec))
        print('SearchWriter-Agent', len(sw_ec), count(sw_ec))

    def calc_tc(self):
        def count(tc):
            c = 0
            for i in range(len(tc)):
                c += len(set(tc[i]))
            return c  / len(tc)
        r1_tc = self.load_tc('./searchwritebench-scores/DeepSeek-R1-tc.jsonl')
        v3_tc = self.load_tc('./searchwritebench-scores/DeepSeek-V3-tc.jsonl')
        flash_lite_tc = self.load_tc('./searchwritebench-scores/gemini-2.5-flash-lite-tc.jsonl')
        flash_tc = self.load_tc('./searchwritebench-scores/gemini-2.5-flash-tc.jsonl')
        pro_tc = self.load_tc('./searchwritebench-scores/gemini-2.5-pro-tc.jsonl')
        gpt35_tc = self.load_tc('./searchwritebench-scores/gpt-3.5-turbo-tc.jsonl')
        gpt4o_tc = self.load_tc('./searchwritebench-scores/gpt-4o-tc.jsonl')
        lw_tc = self.load_tc('./searchwritebench-scores/LongWriter-tc.jsonl')
        lwz_tc = self.load_tc('./searchwritebench-scores/LongWriter-Zero-tc.jsonl')
        o3_tc = self.load_tc('./searchwritebench-scores/o3-mini-tc.jsonl')
        qwen7_tc = self.load_tc('./searchwritebench-scores/Qwen2.5-7B-Instruct-tc.jsonl')
        qwen32_tc = self.load_tc('./searchwritebench-scores/Qwen2.5-32B-Instruct-tc.jsonl')
        qwen72_tc = self.load_tc('./searchwritebench-scores/Qwen2.5-72B-Instruct-tc.jsonl')
        qwq_tc = self.load_tc('./searchwritebench-scores/QwQ-32B-tc.jsonl')
        sw_tc = self.load_tc('./searchwritebench-scores/SearchWriter-tc.jsonl')

        print('gpt-3.5-turbo', count(gpt35_tc))
        print('gpt-4o', count(gpt4o_tc))
        print('gemini-2.5-flash-lite', count(flash_lite_tc))
        print('gemini-2.5-flash', count(flash_tc))
        print('gemini-2.5-pro', count(pro_tc))
        print('QWen2.5-7B-Instruct', count(qwen7_tc))
        print('QWen2.5-32B-Instruct', count(qwen32_tc))
        print('QWen2.5-72B-Instruct', count(qwen72_tc))
        print('DeepSeek-V3', count(v3_tc))
        print('o3-mini', count(o3_tc))
        print('QwQ-32B', count(qwq_tc))
        print('DeepSeek-R1', count(r1_tc))
        print('LongWriter', count(lw_tc))
        print('LongWriter-Zero', count(lwz_tc))
        print('SearchWriter-Agent', count(sw_tc))

    def calc_rc(self):
        def count(rc):
            c = 0
            for i in range(len(rc)):
                if 'inline_attributions' in rc[i].keys() and 'formal_markers' in rc[i].keys() and 'bibliography_count' in rc[i].keys() and 'functional_citation_count' in rc[i].keys():
                    c += (len(rc[i]['inline_attributions']))
            return c  / len(rc)
        r1_rc = self.load_rc('./searchwritebench-scores/DeepSeek-R1-rc.jsonl')
        v3_rc = self.load_rc('./searchwritebench-scores/DeepSeek-V3-rc.jsonl')
        flash_lite_rc = self.load_rc('./searchwritebench-scores/gemini-2.5-flash-lite-rc.jsonl')
        flash_rc = self.load_rc('./searchwritebench-scores/gemini-2.5-flash-rc.jsonl')
        pro_rc = self.load_rc('./searchwritebench-scores/gemini-2.5-pro-rc.jsonl')
        gpt35_rc = self.load_rc('./searchwritebench-scores/gpt-3.5-turbo-rc.jsonl')
        gpt4o_rc = self.load_rc('./searchwritebench-scores/gpt-4o-rc.jsonl')
        lw_rc = self.load_rc('./searchwritebench-scores/LongWriter-rc.jsonl')
        lwz_rc = self.load_rc('./searchwritebench-scores/LongWriter-Zero-rc.jsonl')
        o3_rc = self.load_rc('./searchwritebench-scores/o3-mini-rc.jsonl')
        qwen7_rc = self.load_rc('./searchwritebench-scores/Qwen2.5-7B-Instruct-rc.jsonl')
        qwen32_rc = self.load_rc('./searchwritebench-scores/Qwen2.5-32B-Instruct-rc.jsonl')
        qwen72_rc = self.load_rc('./searchwritebench-scores/Qwen2.5-72B-Instruct-rc.jsonl')
        qwq_rc = self.load_rc('./searchwritebench-scores/QwQ-32B-rc.jsonl')
        sw_rc = self.load_rc('./searchwritebench-scores/SearchWriter-rc.jsonl')

        print('gpt-3.5-turbo', len(gpt35_rc), count(gpt35_rc))
        print('gpt-4o', len(gpt4o_rc), count(gpt4o_rc))
        print('gemini-2.5-flash-lite', len(flash_lite_rc), count(flash_lite_rc))
        print('gemini-2.5-flash', len(flash_rc), count(flash_rc))
        print('gemini-2.5-pro', len(pro_rc), count(pro_rc))
        print('QWen2.5-7B-Instruct', len(qwen7_rc), count(qwen7_rc))
        print('QWen2.5-32B-Instruct', len(qwen32_rc), count(qwen32_rc))
        print('QWen2.5-72B-Instruct', len(qwen72_rc), count(qwen72_rc))
        print('DeepSeek-V3', len(v3_rc), count(v3_rc))
        print('o3-mini', len(o3_rc), count(o3_rc))
        print('QwQ-32B', len(qwq_rc), count(qwq_rc))
        print('DeepSeek-R1', len(r1_rc), count(r1_rc))
        print('LongWriter', len(lw_rc), count(lw_rc))
        print('LongWriter-Zero', len(lwz_rc), count(lwz_rc))
        print('SearchWriter-Agent', len(sw_rc), count(sw_rc))


if __name__ =='__main__':
    evaluator = SWBEval()
    # evaluator.infer('./results/searchwritebench-LongWriter-Zero.jsonl', './searchwritebench-scores/LongWriter-Zero')
    # evaluator.calc_ec()
    # evaluator.calc_tc()
    evaluator.calc_rc()

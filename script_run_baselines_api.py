from config import *
from utils import *


def main():
    # 读取所有benchmark中的元数据和query数据 (~2k)
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
    print(len(hellobench_chat_queries), len(hellobench_htg_queries), len(hellobench_qa_queries), len(hellobench_sum_queries), len(hellobench_comp_queries))

    # (3) 读取SearchWriteBench
    searchwritebench_queries = dl.load_search_write_task()
    print(len(searchwritebench_queries))

    # 调用API来为每个bench进行inference
    silicon_llm_client = OpenAI(api_key=SILICON_KEY, base_url="https://api.siliconflow.cn/v1")
    uiui_llm_client = OpenAI(api_key=UIUI_KEY, base_url="https://sg.uiuiapi.com/v1")
    gemini_llm_client = Client(api_key=GEMINI_KEY)

    # gemini-2.5-flash-lite (ok)
    # gemini-2.5-flash (ok)
    # gemini-2.5-pro (ok)
    # model_name = 'gemini-2.5-pro'
    # gemini_batch_generate(client=gemini_llm_client,
    #                       query_set=writingbench_queries,
    #                       model=model_name,
    #                       save_path='results/writingbench-' + model_name + '.jsonl')

    # gemini_batch_generate(client=gemini_llm_client,
    #                       query_set=hellobench_chat_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-chat-' + model_name + '.jsonl')
    # gemini_batch_generate(client=gemini_llm_client,
    #                       query_set=hellobench_htg_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-htg-' + model_name + '.jsonl')
    # gemini_batch_generate(client=gemini_llm_client,
    #                       query_set=hellobench_qa_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-qa-' + model_name + '.jsonl')
    # gemini_batch_generate(client=gemini_llm_client,
    #                       query_set=hellobench_sum_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-sum-' + model_name + '.jsonl')
    # gemini_batch_generate(client=gemini_llm_client,
    #                       query_set=hellobench_comp_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-comp-' + model_name + '.jsonl')

    # gemini_batch_generate(client=gemini_llm_client,
    #                       query_set=searchwritebench_queries,
    #                       model=model_name,
    #                       save_path='results/searchwritebench-' + model_name + '.jsonl')

    # gpt-3.5-turbo (ok)
    # gpt-4o (ok)
    # o3-mini (ok)
    # model_name = 'o3-mini'
    # openai_batch_generate(client=uiui_llm_client,
    #                       query_set=writingbench_queries,
    #                       model=model_name,
    #                       save_path='results/writingbench-' + model_name + '.jsonl')

    # openai_batch_generate(client=uiui_llm_client,
    #                       query_set=hellobench_chat_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-chat-' + model_name + '.jsonl')
    # openai_batch_generate(client=uiui_llm_client,
    #                       query_set=hellobench_htg_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-htg-' + model_name + '.jsonl')
    # openai_batch_generate(client=uiui_llm_client,
    #                       query_set=hellobench_qa_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-qa-' + model_name + '.jsonl')
    # openai_batch_generate(client=uiui_llm_client,
    #                       query_set=hellobench_sum_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-sum-' + model_name + '.jsonl')
    # openai_batch_generate(client=uiui_llm_client,
    #                       query_set=hellobench_comp_queries,
    #                       model=model_name,
    #                       save_path='results/hellobench-comp-' + model_name + '.jsonl')

    # openai_batch_generate(client=uiui_llm_client,
    #                       query_set=searchwritebench_queries,
    #                       model=model_name,
    #                       save_path='results/searchwritebench-' + model_name + '.jsonl')

    # QWen/Qwen2.5-7B-Instruct (ok)
    # QWen/Qwen2.5-32B-Instruct (ok)
    # QWen/Qwen2.5-72B-Instruct (ok)
    # Qwen/QwQ-32B (ok)
    # deepseek-ai/DeepSeek-V3 (ok)
    # deepseek-ai/DeepSeek-R1 (ok)
    model_name = 'deepseek-ai/DeepSeek-R1'
    # openai_batch_generate(client=silicon_llm_client,
    #                       query_set=writingbench_queries,
    #                       model=model_name,
    #                       save_path='results/writingbench-' + model_name.split('/')[-1] + '.jsonl')

    openai_batch_generate(client=silicon_llm_client,
                          query_set=hellobench_chat_queries,
                          model=model_name,
                          save_path='results/hellobench-chat-' + model_name.split('/')[-1] + '.jsonl')
    openai_batch_generate(client=silicon_llm_client,
                          query_set=hellobench_htg_queries,
                          model=model_name,
                          save_path='results/hellobench-htg-' + model_name.split('/')[-1] + '.jsonl')
    openai_batch_generate(client=silicon_llm_client,
                          query_set=hellobench_qa_queries,
                          model=model_name,
                          save_path='results/hellobench-qa-' + model_name.split('/')[-1] + '.jsonl')
    openai_batch_generate(client=silicon_llm_client,
                          query_set=hellobench_sum_queries,
                          model=model_name,
                          save_path='results/hellobench-sum-' + model_name.split('/')[-1] + '.jsonl')
    openai_batch_generate(client=silicon_llm_client,
                          query_set=hellobench_comp_queries,
                          model=model_name,
                          save_path='results/hellobench-comp-' + model_name.split('/')[-1] + '.jsonl')

    openai_batch_generate(client=silicon_llm_client,
                          query_set=searchwritebench_queries,
                          model=model_name,
                          save_path='results/searchwritebench-' + model_name.split('/')[-1] + '.jsonl')


if __name__ == '__main__':
    main()

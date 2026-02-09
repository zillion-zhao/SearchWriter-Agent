import json
from utils import *

write_path = 'results/SearchWriterInfer/writingbench_write.jsonl'
refine_path = 'results/SearchWriterInfer/writingbench_refine.jsonl'
idx = 7
write_save_md = 'write-res.md'
refine_save_md = 'refine-res.md'

# ----------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------

with open(write_path, 'r', encoding='utf-8') as f:
    line = f.readlines()[idx]
text = json.loads(line)['written_text_with_search']
with open(write_save_md, 'w', encoding='utf-8') as f:
    f.write(text)

with open(refine_path, 'r', encoding='utf-8') as f:
    line = f.readlines()[idx]
text = json.loads(line)['refine_text']
with open(refine_save_md, 'w', encoding='utf-8') as f:
    f.write(text)

print(writingbench_queries[idx])

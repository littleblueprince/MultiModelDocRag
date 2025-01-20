import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
import openai

model = "qwen2.5-72b-instruct"
api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# model = "qwen2-vl-instruct"
# api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
# base_url = "http://36.213.0.171:9997/v1"

client = openai.Client(api_key=api_key, base_url=base_url)


def get_desc(title: str, table_name: str, table: str, ) -> str:
    messages = [
        {
            "role": "user",
            "content": f"Generate a brief and accurate description of the given table.**title**:{title} **content**:{table_name}:{table}",
        }
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return completion.choices[0].message.content


path = './tab_questions.jsonl'
data = read_jsonl(path)
store_path = './tab_descriptions.jsonl'
f, processed = get_output_file(store_path, force=False)
for item in tqdm(data):
    try:
        qid = item['qid']
        if qid in processed:
            continue
        question = item['question']
        answer = item['answer']
        metadata = item['metadata']
        golden_tab = item['golden_tab']
        golden_tab_title = golden_tab['metadata']['title']
        golden_tab_table_name = golden_tab['metadata']['table_name']
        golden_tab_table = golden_tab['metadata']['table']
        golden_tab_desc = get_desc(title=golden_tab_title, table_name=golden_tab_table_name, table=golden_tab_table)

        f.write(json.dumps({
            "qid": qid,
            "golden_tab_desc": golden_tab_desc,
        }, ensure_ascii=False) + '\n')
        f.flush()
    except Exception as e:
        continue
f.close()

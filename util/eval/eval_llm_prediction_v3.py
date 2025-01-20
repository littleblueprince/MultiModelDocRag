# -*- coding: utf-8 -*-
# @Time    : 2024/12/12 18:30
# @Author  : blue
# @Description : 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"
from openai import OpenAI
from util.api import *

base_url = "http://36.137.79.97:30250/v1/"
model = "qwen2-72b-instruct"
api_key = "test"
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
result = []

# baseline
# path = '/data1/zch/MultiModelDocRag/pipeline/baseline_qwen2_vl_7b_v1.jsonl'
# store_path = "./eval_result_v3.jsonl"

path = '/data1/zch/MultiModelDocRag/generate_augment/dag/baseline_v1.jsonl'
store_path = "./eval_result_v2.jsonl"

data = read_jsonl(path)[:875]

intersection_count = 0
subset_count = 0
for item in tqdm(data):
    corpus=item['corpus']
    supporting_context=item['supporting_context']
    supporting_context_ids=[s['doc_id'] for s in supporting_context]
    corpus_ids=[s[0] for s in corpus]
    if set(corpus_ids) >= set(supporting_context_ids):
        subset_count +=1
    if set(corpus_ids) & set(supporting_context_ids):
        intersection_count +=1

print(f'all subset rate: {subset_count / len(data)}')
print(f'all intersection rate: {intersection_count / len(data)}')

f, processed_results = get_output_file(store_path, force=False)

for item in tqdm(data):
    qid = item['qid']
    if qid in processed_results:
        continue
    question = item['question']
    answer = item['answer']
    prediction = item['prediction']
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "你的任务是根据question和ground_truth来判断prediction的结果是否回答正确,"
                           "如果prediction中出现ground_truth本身或者相似词则回答正确输出True，否则输出False。请不要进行任何推理，只输出Ture、False"
                           f"\n\n**question**：{question} \n**ground_truth**：{answer} \n\n**prediction**：{prediction}",
            }
        ],
        model=model,
        max_tokens=150,  # 设置最大生成的 token 数量
        temperature=0.7,  # 控制生成内容的随机性
        top_p=0.9,  # 控制生成内容的多样性
        n=1,  # 返回生成的候选响应数量
        stop=["\n"],  # 生成内容的终止标识
        presence_penalty=0.6,  # 控制内容多样性，减少内容重复性
        frequency_penalty=0.5,  # 减少重复词汇生成的频率
        stream=False,  # 启用流式响应
    )
    f.write(json.dumps({
        "qid": item['qid'],
        "question": item['question'],
        "ground_truth": item['answer'],
        "predict": item['prediction'],
        "eval_res": chat_completion.choices[0].message.content,
    }) + "\n")
    f.flush()
f.close()

eval_data = read_jsonl(store_path)
count = 0
for item in eval_data:
    if item['eval_res'] == 'True':
        count += 1
print(f'count: {count} \ntotal: {len(eval_data)}')

print(f'all hit rate: {count / len(eval_data)}')

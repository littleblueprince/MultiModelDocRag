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

query_data = read_jsonl('/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl')
query_data_dict = dict()
for item in query_data:
    query_data_dict[item['question']] = item
query_keys = list(query_data_dict.keys())

# baseline
# path = '/data1/zch/MultiModelDocRag/pipeline/baseline_qwen2_vl_7b_v1.jsonl'
# store_path = "./eval_result_v1.jsonl"

path = '/data1/zch/MultiModelDocRag/pipeline/baseline_qwen2_vl_7b_v10.jsonl'
store_path = "./eval_result_v10.jsonl"

data = read_jsonl(path)
for item in tqdm(data):
    prompt = item['prompt']
    label = item['label']
    predict = item['predict']
    for key in query_keys:
        if key in prompt:
            query_data_dict[key]['predict'] = predict
            query_data_dict[key]['ground_truth'] = label
            break

f, processed_results = get_output_file(store_path, force=False)

for item in tqdm(query_data_dict.values()):
    qid = item['qid']
    if qid in processed_results:
        continue
    question = item['question']
    ground_truth = item['ground_truth']
    predict = item['predict']
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "你的任务是根据question和ground_truth来判断prediction的结果是否回答正确,"
                           "回答正确输出True，回答不正确输出False。请不要进行任何推理，只输出Ture、False"
                           f"\n\n**question**：{question} \n**ground_truth**：{ground_truth} \n\n**prediction**：{predict}",
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
        "ground_truth": item['ground_truth'],
        "predict": item['predict'],
        "eval_res": chat_completion.choices[0].message.content,
        "type": item['metadata']['type'],
        "supporting_context": item['supporting_context'],
    }) + "\n")
    f.flush()
f.close()

eval_data = read_jsonl(store_path)
count = 0
type_count = dict()
for item in eval_data:
    if item['type'] not in type_count:
        type_count[item['type']] = {
            "hit": 0,
            "total": 0,
        }
    type_count[item['type']]['total'] += 1
    if item['eval_res'] == 'True':
        count += 1
        type_count[item['type']]['hit'] += 1

print(f'count: {count} \ntotal: {len(eval_data)}')

print(f'all hit rate: {count / len(eval_data)}')

for key in type_count:
    value = type_count[key]
    value['hit_rate'] = value['hit'] / value['total']
pretty_print_json(type_count)

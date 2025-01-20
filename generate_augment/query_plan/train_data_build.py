# -*- coding: utf-8 -*-
# @Time    : 2024/12/27 00:28
# @Author  : blue
# @Description :
import json
import os

os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
from util.chat_template import *
from openai import OpenAI

# model = "qwen2.5-instruct"
# base_url = "http://36.213.0.171:9997/v1/"

api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen2.5-72b-instruct"

# base_url = "https://api.deepseek.com/v1"
# model = "deepseek-chat"
# api_key = 'sk-550e97b86ab9414f9832694805f361ca'

path = '/data1/zch/datasets/multimodalqa/MMQA_train.jsonl'
# path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
store_path = './train_data_' + model + '_v3.jsonl'
# store_path = './dev_data_' + model + '_v1.jsonl'

data = read_jsonl(path)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
types = set()
for item in tqdm(data):
    types.add(item['metadata']['type'])
f, processed_results = get_output_file(store_path, force=False)


path1 = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/train_data_qwen2.5-72b-instruct_v1.jsonl'
path2 = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/train_data_v2.jsonl'

data1 = read_jsonl(path1)
data2 = read_jsonl(path2)
data1_ids = [item['qid'] for item in data1]
data2_ids = [item['qid'] for item in data2]
data_ids = data1_ids + data2_ids
data=[item for item in data if (item['qid'] not in data_ids and item['qid'] not in processed_results)]
for item in tqdm(data):
    qid = item['qid']
    type = item['metadata']['type']
    if qid in processed_results:
        continue
    question = item['question']
    answer = item['answers'][0]['answer']
    supporting_context = item['supporting_context']
    messages = get_train_query_plan_prompt(question=question, answer=answer, type=type)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    f.write(json.dumps({
        "qid": qid,
        "question": question,
        "answer": answer,
        "supporting_context": supporting_context,
        "dag": chat_completion.choices[0].message.content,
        "prompt": messages[0]['content'],

    }) + "\n")
    f.flush()
f.close()


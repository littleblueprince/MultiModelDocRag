# -*- coding: utf-8 -*-
# @Time    : 2024/12/27 00:28
# @Author  : blue
# @Description : 
import os

os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
from util.chat_template import *
from openai import OpenAI

# model = "qwen2.5-instruct"
# base_url = "http://36.213.0.171:9997/v1/"

base_url = "http://36.137.79.97:30250/v1"
model = "qwen2-72b-instruct"

path = '/data1/zch/datasets/multimodalqa/MMQA_train.jsonl'
store_path = './train_data_v2.jsonl'

data = read_jsonl(path)

api_key = 'test'
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
types = set()
for item in tqdm(data):
    types.add(item['metadata']['type'])
f, processed_results = get_output_file(store_path, force=False)
for item in tqdm(data):
    qid = item['qid']
    type = item['metadata']['type']
    supporting_context = item['supporting_context']
    if qid in processed_results or type not in ['TableQ', 'TextQ', 'ImageQ', 'ImageListQ'] or len(
            supporting_context) != 1:
        continue
    question = item['question']
    answer = item['answers'][0]['answer']

    messages = get_train_query_plan_prompt_1_hop(question=question, answer=answer)
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

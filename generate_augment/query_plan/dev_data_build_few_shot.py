# -*- coding: utf-8 -*-
# @Time    : 2024/12/27 00:28
# @Author  : blue
# @Description : 
import os

import requests

os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
from util.chat_template import *
from openai import OpenAI
from pymilvus import connections, Collection


def retrieval_dag_examples(
        query_vector: list,
        collection_name: str,
        db_name="mmqa",
        top_k=3,
        host="10.176.40.88",
        port="19530",
        user='root',
        password='password',
) -> list:
    # 根据query_vec去milvus数据库搜索corpus返回
    connections.connect(
        host=host,
        port=port,
        db_name=db_name,
        user=user,
        password=password,
    )
    collection = Collection(name=collection_name)
    collection.load()
    results = collection.search(
        data=[query_vector],
        anns_field='vector',
        param={
            "M": 64,
            "efConstruction": 1000
        },
        output_fields=["metadata"],
        limit=top_k,
    )
    return [
        {
            "question": results[0][0].fields['metadata']['question'],
            "dag": results[0][0].fields['metadata']['dag'],
        },
        {
            "question": results[0][1].fields['metadata']['question'],
            "dag": results[0][1].fields['metadata']['dag'],
        },
        {
            "question": results[0][2].fields['metadata']['question'],
            "dag": results[0][2].fields['metadata']['dag'],
        },
    ]


emb_base_url = "http://10.176.40.88:9997/v1/embeddings"
emb_api_key = 'test'
embedding_model = "bge_m3_cuda0"
emb_headers = {
    'Authorization': f'Bearer {emb_api_key}',
    'Content-Type': 'application/json',
}

model = "qwen2.5-instruct"
base_url = "http://36.213.0.171:9997/v1/"
api_key = 'test'

# base_url = "https://api.deepseek.com/v1"
# model = "deepseek-chat"
# api_key = 'sk-550e97b86ab9414f9832694805f361ca'

# base_url = "http://36.137.79.97:30250/v1"
# model = "qwen2-72b-instruct"

path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
store_path = './dev_data_' + model + '_v1.jsonl'

data = read_jsonl(path)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

f, processed_results = get_output_file(store_path, force=False)
for item in tqdm(data):
    qid = item['qid']

    if qid in processed_results:
        continue
    question = item['question']
    answer = item['answers'][0]['answer']
    supporting_context = item['supporting_context']
    metadata = item['metadata']
    query_vector_res = requests.post(emb_base_url, headers=emb_headers, json={
        "model": embedding_model,
        "input": question,
    }).json()
    query_vector = query_vector_res['data'][0]['embedding']
    examples = retrieval_dag_examples(query_vector, collection_name='dag_v2')

    messages = get_dev_query_plan_prompt_few_shot_v1(question=question, examples=examples)
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
        "metadata": metadata,
        "prompt": messages[0]['content'],
    }) + "\n")
    f.flush()
f.close()

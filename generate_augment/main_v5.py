# -*- coding: utf-8 -*-
# @Time    : 2024/12/28 15:17
# @Author  : blue
# @Description :
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

import requests
from util.retrieval_api import *
from util.api import *


def get_embedding_sync(text: str, url: str = "http://36.213.0.171:8001/encode/"):
    data = {
        "text": text,
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            json_response = response.json()
            return {
                "embedding": json_response["embedding"][0]
            }
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        return {"error": str(e)}


path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/prediction_qwen2_vl_72b_dev_v4.jsonl'
store_path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/prediction_qwen2_vl_72b_dev_v6.jsonl'
data = read_jsonl(path)
res = []
for item in tqdm(data):
    for node in item['graph_logs']:
        query = node['query']
        action = node['action']
        if "rewrite_query" in node:
            query = node['rewrite_query']
        query_vector = get_embedding_sync(text=query)['embedding']
        doc_ids = [str(id) for id in node['corpus']]
        corpus = retrieval_corpus(
            query_vector=query_vector,
            text_filtered_ids=doc_ids,
            tabs_filtered_ids=doc_ids,
            imgs_filtered_ids=doc_ids,
            text_collection_name='text_v1',
            tabs_collection_name='tab_v1',
            imgs_collection_name='img_v2',
        )
        node['corpus'] = [[c['metadata']['id'],c['score']] for c in corpus[:3] ]

    res.append(item)

store_jsonl(store_path, res)

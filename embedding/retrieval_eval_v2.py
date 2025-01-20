# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 15:36
# @Author  : blue
# @Description :  评估embedding模型检索效果
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *

store_path = './eval_v1.json'
data = read_json(store_path)

query_data = read_jsonl('/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl')
qids = [
    item['qid']
    for item in query_data
    if any(context['doc_part'] == 'image' for context in item['supporting_context'])
]

# data = [item for item in data if item['qid']  in qids]
data = [item for item in data if item['qid'] not in qids]
# data=data[:500]

topk1 = 3

count_topk1_issubset = 0
count_topk1_intersection = 0
for item in data:
    supporting_context = item['supporting_context']
    top_k_corpus = item['top_k_corpus']
    supporting_context_set = set(supporting_context)
    top_k1_corpus_set = set(top_k_corpus[:topk1])
    if top_k1_corpus_set & supporting_context_set:
        count_topk1_intersection += 1
    count_topk1_issubset += supporting_context_set.issubset(top_k1_corpus_set)

print(f'dataset len: {len(data)}')
print(f'count_topk{topk1}_issubset : {count_topk1_issubset}')
print(f'count_topk{topk1}_intersection : {count_topk1_intersection}')

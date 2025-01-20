# -*- coding: utf-8 -*-
# @Time    : 2024/12/21 16:02
# @Author  : blue
# @Description :
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *

store_path = './eval_v1.json'
data = read_json(store_path)

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

"""
No Ft:
dataset len: 940
average corpus len: 21.46276595744681
count_topk3_issubset : 266
count_topk3_intersection : 754
count_topk5_issubset : 322
count_topk5_intersection : 793
"""
"""
训练1epoch 5e-5
dataset len: 940
count_topk3_issubset : 394
count_topk3_intersection : 798
"""

"""
50
16
41
"""

# -*- coding: utf-8 -*-
# @Time    : 2025/01/15 16:13
# @Author  : blue
# @Description : 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *


path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
data = read_jsonl(path)
for item in data:
    if  len(item['supporting_context']) ==1 and item['supporting_context'][0]['doc_part'] =="image":
        pretty_print_json(item)
        input()

# path = '/data1/zch/MultiModelDocRag/rerank/train_dataset_v1.jsonl'
# data = read_jsonl(path)
# for item in data:
#     pretty_print_json(item)
#     input()


# path = '/data1/zch/datasets/multimodalqa/MMQA_train.jsonl'
# data = read_jsonl(path)
# dataset_path = '/data1/zch/MultiModelDocRag/generate_augment/dataset.json'
# dataset = read_json(dataset_path)
# res = []
# for item in tqdm(data):
#     if len(item['supporting_context']) == 1:
#         question = item['question']
#         image_doc_ids = item['metadata']['image_doc_ids']
#         text_doc_ids = item['metadata']['text_doc_ids']
#         table_id = item['metadata']['table_id']
#         ids_all = image_doc_ids + text_doc_ids
#         ids_all.append(table_id)
#         supporting_context_id = item['supporting_context'][0]['doc_id']
#         for idx in ids_all:
#             if idx == supporting_context_id:
#                 res.append(
#                     {
#                         "question": question,
#                         "candidate": dataset[idx],
#                         "label": 1,
#                     }
#                 )
#             else:
#                 res.append(
#                     {
#                         "question": question,
#                         "candidate": dataset[idx],
#                         "label": 0,
#                     }
#                 )
#     else:
#         pass
#
# store_jsonl('/data1/zch/MultiModelDocRag/rerank/train_dataset_v1.jsonl', res)

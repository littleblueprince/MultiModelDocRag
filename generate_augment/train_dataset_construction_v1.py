# -*- coding: utf-8 -*-
# @Time    : 2024/12/12 21:03
# @Author  : blue
# @Description :
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
from util.chat_template import *
import random

to_del_img_ids = [
    '17ae0616ac745e70781203267f3a382d',
    '225c3db49d60b5ef30ed0bfc649ebf78',
    'b413cc1dc4969dcbe4cb6a55c0f2e359',
    'bf201cbbd058ef51aef89b1be4158c2a',
    'e81b2acfd792b171389c8f47a0e14504',
    'ef457a7b3ab437cd78ab9f82dc083048'
]

path = '/data1/zch/datasets/multimodalqa/MMQA_train.jsonl'
# path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
dataset_path = "/data1/zch/MultiModelDocRag/generate_augment/dataset.json"

# data=pre_process_dataset()
# store_json(dataset_path, data)

store_path = "/data1/zch/MultiModelDocRag/generate_augment/qwen2_vl_train_v2.json"
# store_path = "/data1/zch/MultiModelDocRag/generate_augment/qwen2_vl_dev_v1.json"
data = read_jsonl(path)
dataset_dict = read_json(dataset_path)

res = []
for item in tqdm(data):
    question = item['question']
    ground_truth = item['answers'][0]['answer']
    supporting_context = item['supporting_context']
    corpus = []
    metadata = item['metadata']
    image_doc_ids = metadata['image_doc_ids']
    image_doc_ids = [idx for idx in image_doc_ids if idx not in to_del_img_ids]
    text_doc_ids = metadata['text_doc_ids']
    table_id = metadata['table_id']
    if len(supporting_context) <= 2:
        negs_ids = []
        negs_ids.extend(image_doc_ids)
        negs_ids.extend(text_doc_ids)
        negs_ids.append(table_id)
        negs_ids = list(set(negs_ids) - set(corpus))
        negs = random.sample(negs_ids, 3 - len(corpus))
        for idx in negs:
            corpus.append(dataset_dict[idx])
        for context in supporting_context:
            doc_id = context['doc_id']
            corpus.append(dataset_dict[doc_id])
        random.shuffle(corpus)
        prompt = get_train_QA_prompt(question=question, corpus=corpus, ground_truth=ground_truth)
        res.append(prompt)
    elif len(supporting_context) == 3:
        for context in supporting_context:
            doc_id = context['doc_id']
            corpus.append(dataset_dict[doc_id])
        random.shuffle(corpus)
        prompt = get_train_QA_prompt(question=question, corpus=corpus, ground_truth=ground_truth)
        res.append(prompt)
    else:
        pass

store_json(store_path, res)

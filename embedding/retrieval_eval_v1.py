# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 15:36
# @Author  : blue
# @Description :  评估embedding模型检索效果
import asyncio
import os

import aiohttp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='/data1/zch/models/bge-base-en-v1.5',
    use_fast=False
)


def truncate_and_restore(text, MAX_LENGTH=512) -> str:
    encoded = tokenizer(
        text,
        padding=True,  # 自动填充，确保长度一致
        truncation=True,  # 启用截断
        max_length=MAX_LENGTH,  # 最大长度
        return_tensors='pt'
    )
    input_ids = encoded['input_ids']
    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return decoded_text


async def get_embedding_async(text=None, file_path=None, url: str = "http://36.213.0.171:8001/encode/"):
    data = {}
    if text:
        data['text'] = text
    if file_path:
        data['file_path'] = file_path
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    json_response = await response.json()
                    return json_response["embedding"][0]
                else:
                    return {"error": f"Error {response.status}: {await response.text()}"}
    except aiohttp.ClientError as e:
        return {"error": str(e)}


async def get_multiple_embeddings_async(items):
    tasks = []
    for item in items:
        if "file_path" in item:
            tasks.append(get_embedding_async(text=item['text'], file_path=img_dir + item['file_path']))
        else:
            tasks.append(get_embedding_async(text=item['text']))
    results = await asyncio.gather(*tasks)
    return results


def find_top_k_similar_lists(lists, k):
    lists_array = np.array(lists)
    first_list = lists_array[0]
    distances = [cosine(first_list, other) for other in lists_array[1:]]
    similarities = [1 - dist for dist in distances]
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return top_k_indices + 1


path = "/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl"
# path = "/data1/zch/datasets/multimodalqa/MMQA_train.jsonl"
dataset = read_jsonl(path)

top_k = 100
db_name = "mmqa"

store_path = './eval_v1.json'
# store_path = './milvus_corpus_full.json'

hits = []
count = 0
# dataset_dict = dict()
# texts_path = '/data1/zch/datasets/multimodalqa/MMQA_texts.jsonl'
# texts_dataset = read_jsonl(texts_path)
# print(f'processing texts_dataset')
# for item in tqdm(texts_dataset):
#     title = item['title']
#     id = item['id']
#     text = item['text']
#     dataset_dict[id] = {
#         "id": id,
#         "text": truncate_and_restore(f'**title**:{title}**text**:{text}'),
#     }
# imgs_path = '/data1/zch/datasets/multimodalqa/MMQA_images.jsonl'
# imgs_dataset = read_jsonl(imgs_path)
#
# print(f'processing imgs_dataset')
# for item in tqdm(imgs_dataset):
#     title = item['title']
#     id = item['id']
#     path = item['path']
#     dataset_dict[id] = {
#         "id": id,
#         "text": title,
#         "file_path": path,
#     }
# tables_path = '/data1/zch/datasets/multimodalqa/MMQA_tables_md_cleaned.jsonl'
# tables_dataset = read_jsonl(tables_path)
# print(f'processing tables_dataset')
# for item in tqdm(tables_dataset):
#     title = item['title']
#     id = item['id']
#     table = item['table']
#     table_name = item['table_name']
#     dataset_dict[id] = {
#         "id":id,
#         "text": truncate_and_restore(
#             f'**{title}:** ## {table_name}: {table}'),
#     }
#
# store_json('./dataset_dict.json',dataset_dict)

dataset_dict = read_json('/data1/zch/MultiModelDocRag/embedding/dataset_dict.json')
img_dir = '/mnt/sdb/zch/LLaMA-Factory/data/qwen2_vl_ft/mllm_data_v1/'
print(f'Total {len(dataset)} questions')

# dataset=dataset[:500]

for i, item in enumerate(tqdm(dataset)):
    count += 1
    qid = item['qid']
    answers = item['answers']
    question = item['question']
    metadata = item['metadata']
    supporting_context = item['supporting_context']

    image_doc_ids = metadata['image_doc_ids']
    text_doc_ids = metadata['text_doc_ids']
    table_id = metadata['table_id']

    objects = []
    objects.append({
        "text": truncate_and_restore(question),
    })
    for idx in image_doc_ids:
        objects.append(dataset_dict[idx])
    for idx in text_doc_ids:
        objects.append(dataset_dict[idx])
    objects.append(dataset_dict[table_id])
    objects_embeddings = asyncio.run(get_multiple_embeddings_async(objects))
    top_k_indices = find_top_k_similar_lists(objects_embeddings, k=3)
    # hits.append({
    #     "qid": qid,
    #     "answers": answers,
    #     "question": question,
    #     "supporting_context": supporting_context,
    #     "metadata": metadata,
    #     "milvus_corpus": milvus_corpus
    # })
    hits.append({
        "qid": qid,
        "supporting_context": [context["doc_id"] for context in supporting_context],
        "top_k_corpus": [objects[i]['id'] for i in top_k_indices]
    })

store_json(store_path, hits)

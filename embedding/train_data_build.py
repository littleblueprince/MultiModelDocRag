# -*- coding: utf-8 -*-
# @Time    : 2024/12/18 11:16
# @Author  : blue
# @Description : 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"
from util.api import *
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


query_path = '/data1/zch/datasets/multimodalqa/MMQA_train.jsonl'
query_data = read_jsonl(query_path)

imgs_path = '/data1/zch/datasets/multimodalqa/MMQA_images.jsonl'
imgs_data = read_jsonl(imgs_path)
imgs_dict = dict()
for item in imgs_data:
    imgs_dict[item['id']] = item

tabs_path = '/data1/zch/datasets/multimodalqa/MMQA_tables_md_cleaned.jsonl'
tabs_data = read_jsonl(tabs_path)
tabs_dict = dict()
for item in tabs_data:
    tabs_dict[item['id']] = item

texts_path = '/data1/zch/datasets/multimodalqa/MMQA_texts.jsonl'
texts_data = read_jsonl(texts_path)
texts_dict = dict()
for item in texts_data:
    texts_dict[item['id']] = item

result = []
# todo: 添加硬负样本挖掘
# /data1/zch/FlagEmbedding/mycode/milvus_corpus_part_v4.json

for item in tqdm(query_data):
    qid = item['qid']
    question = item['question']
    answers = item['answers']
    metadata = item['metadata']
    # 候选项id
    image_doc_ids = metadata['image_doc_ids']
    text_doc_ids = metadata['text_doc_ids']
    table_id = metadata['table_id']

    supporting_context = item['supporting_context']  # doc_id doc_part
    supporting_ids = [context['doc_id'] for context in supporting_context]
    for image_doc_id in image_doc_ids:
        result.append(
            {
                "question": question,
                "text": imgs_dict[image_doc_id]['title'],
                "img": imgs_dict[image_doc_id]['path'],
                "label": 1 if image_doc_id in supporting_ids else 0,
            }
        )

    # for image_doc_id in image_doc_ids:
    #     result.append(
    #         {
    #             "question": question,
    #             "candidate": {
    #                 "text": imgs_dict[image_doc_id]['title'],
    #                 "img": imgs_dict[image_doc_id]['path'],
    #             },
    #             "task": 't2it',
    #             "label": 1 if image_doc_id in supporting_ids else 0,
    #         }
    #     )
    # for text_doc_id in text_doc_ids:
    #     title = texts_dict[text_doc_id]['title']
    #     text = texts_dict[text_doc_id]['text']
    #     result.append(
    #         {
    #             "question": question,
    #             "candidate": {
    #                 "text": truncate_and_restore(f'**title**:{title}**text**:{text}'),
    #             },
    #             "task": 't2t',
    #             "label": 1 if text_doc_id in supporting_ids else 0,
    #         }
    #     )
    # result.append(
    #     {
    #         "question": question,
    #         "candidate": {
    #             "text": truncate_and_restore(
    #                 f'**{tabs_dict[table_id]["title"]}:** ## {tabs_dict[table_id]["table_name"]}: {tabs_dict[table_id]["table"]}'),
    #         },
    #         "task": 't2t',
    #         "label": 1 if table_id in supporting_ids else 0,
    #     }
    # )

store_path = '/data1/zch/MultiModelDocRag/embedding/train_v4.jsonl'
store_jsonl(store_path, result)

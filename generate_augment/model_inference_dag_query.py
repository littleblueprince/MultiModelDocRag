# # -*- coding: utf-8 -*-
# # @Time    : 2024/12/11 20:59
# # @Author  : blue
# # @Description :
import os

os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
from util.chat_template import *
from util.retrieval_api import *
from visual_bge.modeling import Visualized_BGE
from openai import OpenAI
import torch

model_name = "Qwen2-VL-72B-Instruct-0"

path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
store_path = './data/dag_query_72b_v1.jsonl'
data = read_jsonl(path)

# base_url = "http://36.137.79.97:30250/v1/"
base_url = "http://36.213.0.171:9997/v1/"

api_key = 'test'
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

model = Visualized_BGE(
    model_name_bge="bge-base-en-v1.5",
    model_weight="/data1/zch/models/bge-visualized/Visualized_base_en_v1.5.pth",
    from_pretrained='/data1/zch/models/bge-base-en-v1.5'
)
model.eval()

f, processed_results = get_output_file(store_path, force=False)
topk = 3
for item in tqdm(data):
    qid = item['qid']
    if qid in processed_results:
        continue
    question = item['question']
    answers = item['answers']
    supporting_context = item['supporting_context']
    metadata = item['metadata']
    text_filtered_ids = []
    tabs_filtered_ids = []
    imgs_filtered_ids = []

    for idx in metadata['image_doc_ids']:
        imgs_filtered_ids.append(idx)
    for idx in metadata['text_doc_ids']:
        text_filtered_ids.append(idx)
    tabs_filtered_ids.append(metadata['table_id'])

    with torch.no_grad():
        query_emb = model.encode(text=question).detach().cpu()
    milvus_corpus = retrieval_corpus(
        query_emb,
        text_filtered_ids,
        tabs_filtered_ids,
        imgs_filtered_ids,
        text_collection_name="v1",
        tabs_collection_name="v3",
        imgs_collection_name="v2",
    )
    milvus_corpus = milvus_corpus[:topk]
    messages = get_temp_messages_v1(question=question, corpus=milvus_corpus)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    f.write(json.dumps({
        "qid": qid,
        "question": question,
        "answers": answers,
        "supporting_context": supporting_context,
        "dag": chat_completion.choices[0].message.content,
    }) + "\n")
    f.flush()
f.close()

# -*- coding: utf-8 -*-
# @Time    : 2024/12/13 13:41
# @Author  : blue
# @Description : 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from util.api import *
from util.chat_template import *
from util.retrieval_api import *
from visual_bge.modeling import Visualized_BGE
from openai import OpenAI

model_names = [
    "qwen2-vl-instruct-0",
    "qwen2-vl-instruct-1",
    "qwen2-vl-instruct-2",
    "qwen2-vl-instruct-3",
    "qwen2-vl-instruct-4",
    "qwen2-vl-instruct-5",
    "qwen2-vl-instruct-6",
    "qwen2-vl-instruct-7",
]

path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
store_path = './data/golden_baseline.jsonl'
data = read_jsonl(path)
base_url = "http://36.213.0.171:9997/v1/"

api_key = 'test'
topk = 3

models = {}
for model_name in model_names:
    models[model_name] = Visualized_BGE(
        model_name_bge="bge-base-en-v1.5",
        model_weight="/data1/zch/models/bge-visualized/Visualized_base_en_v1.5.pth",
        from_pretrained='/data1/zch/models/bge-base-en-v1.5'
    )
    models[model_name].eval()

clients = {model_name: OpenAI(api_key=api_key, base_url=base_url) for model_name in model_names}

f, processed_results = get_output_file(store_path, force=False)


def process_item(item, model_name):
    qid = item['qid']
    print(f'processing qid: {qid}')
    if qid in processed_results:
        return None

    question = item['question']
    answers = item['answers']
    supporting_context = item['supporting_context']
    metadata = item['metadata']

    # text_filtered_ids = metadata['text_doc_ids']
    # tabs_filtered_ids = [metadata['table_id']]
    # imgs_filtered_ids = metadata['image_doc_ids']

    text_filtered_ids = []
    tabs_filtered_ids = []
    imgs_filtered_ids = []
    for context in supporting_context:
        if context['doc_part'] == "image":
            imgs_filtered_ids.append(context['doc_id'])
        elif context['doc_part'] == "table":
            tabs_filtered_ids.append(context['doc_id'])
        else:
            text_filtered_ids.append(context['doc_id'])
    with torch.no_grad():
        query_emb = models[model_name].encode(text=question).detach().cpu()

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

    messages = get_temp_messages_v2(question=question, corpus=milvus_corpus)
    chat_completion = clients[model_name].chat.completions.create(
        messages=messages,
        model=model_name,
    )

    return {
        "qid": qid,
        "question": question,
        "answers": answers,
        "supporting_context": supporting_context,
        "prediction": chat_completion.choices[0].message.content,
    }


results = []
with ThreadPoolExecutor(max_workers=len(model_names)) as executor:
    future_to_item = {
        executor.submit(process_item, item, model_names[i % len(model_names)]): item
        for i, item in enumerate(data)
    }
    for future in tqdm(as_completed(future_to_item), total=len(data)):
        try:
            result = future.result()
            if result:
                results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()
        except Exception as e:
            print(f"Error processing item: {e}")

f.close()

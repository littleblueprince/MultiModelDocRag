# -*- coding: utf-8 -*-
# @Time    : 2024/12/26 19:41
# @Author  : blue
# @Description : 最简单的rag baseline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
from util.retrieval_api import retrieval_corpus
import torch
from openai import OpenAI
from visual_bge.modeling import Visualized_BGE
from util.chat_template import get_normal_QA_prompt

path = '/data1/zch/MultiModelDocRag/generate_augment/dag/dag_query_72b_dev_v1.jsonl'
data = read_jsonl(path)

model = Visualized_BGE(
    model_name_bge="bge-base-en-v1.5",
    model_weight="/data1/zch/models/bge-visualized/Visualized_base_en_v1.5.pth",
    from_pretrained='/data1/zch/models/bge-base-en-v1.5'
)
model.eval()


def get_embedding(text: input) -> list:
    with torch.no_grad():
        query_emb = model.encode(text=text).detach().cpu()
    return query_emb


base_url = "http://36.213.0.171:9997/v1/"
api_key = 'test'
client = OpenAI(api_key=api_key, base_url=base_url)
model_name = 'Qwen2-VL-7B-Instruct'


def llm_predict(messages: list, model_name='Qwen2-VL-72B-Instruct') -> str:
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    return chat_completion.choices[0].message.content


store_path = '/data1/zch/MultiModelDocRag/generate_augment/dag/baseline_v2.jsonl'

f, processed = get_output_file(store_path, force=False)
topk = 3
for item in tqdm(data):
    qid = item['qid']
    if qid in processed:
        continue
    question = item['question']
    answer = item['answer']
    graph = item['graph']
    metadata = item['metadata']
    supporting_context = item['supporting_context']

    text_filtered_ids = metadata['text_doc_ids']
    tabs_filtered_ids = [metadata['table_id']]
    imgs_filtered_ids = metadata['image_doc_ids']
    graph_dict = dict()
    query_emb = get_embedding(question)
    milvus_corpus = retrieval_corpus(
        query_emb,
        text_filtered_ids,
        tabs_filtered_ids,
        imgs_filtered_ids,
        text_collection_name="text_v1",
        tabs_collection_name="tab_v1",
        imgs_collection_name="img_v1",
    )
    corpus = milvus_corpus[:topk]
    corpus_ids = [(c['metadata']['id'], c['score']) for c in corpus]
    messages = get_normal_QA_prompt(question=question, corpus=corpus)
    prediction = llm_predict(messages=messages, model_name=model_name)
    f.write(json.dumps(
        {
            "qid": qid,
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "supporting_context": supporting_context,
            "corpus": corpus_ids,
            "metadata": metadata,
        }
    ) + "\n")
    f.flush()
f.close()

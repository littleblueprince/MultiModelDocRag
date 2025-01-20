# -*- coding: utf-8 -*-
# @Time    : 2024/12/13 15:05
# @Author  : blue
# @Description : 基于查询计划递归查询回答
import asyncio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
from openai import OpenAI
from util.chat_template import get_normal_QA_prompt
import aiohttp
from scipy.spatial.distance import cosine
import requests

path = '/data1/zch/MultiModelDocRag/generate_augment/dag/dag_query_72b_dev_v4.jsonl'
data = read_jsonl(path)
img_dir = '/mnt/sdb/zch/LLaMA-Factory/data/qwen2_vl_ft/mllm_data_v1/'


async def get_multiple_embeddings_async(items):
    tasks = []
    for item in items:
        if "file_path" in item:
            tasks.append(get_embedding_async(text=item['text'], file_path=img_dir + item['file_path']))
        else:
            tasks.append(get_embedding_async(text=item['text']))
    results = await asyncio.gather(*tasks)
    return results


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


def get_embedding(text=None, file_path=None, url: str = "http://36.213.0.171:8001/encode/"):
    if not text and not file_path:
        raise ValueError("Either 'text' or 'file_path' must be provided.")
    data = {}
    if text:
        data['text'] = text
    if file_path:
        data['file_path'] = file_path
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            json_response = response.json()
            if "embedding" not in json_response:
                return {"error": "Response does not contain 'embedding' field."}
            return json_response["embedding"][0]
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        return {"error": str(e)}


def find_top_k_similar_candidates(query_rep, candidates_reps, k: int):
    query_rep_array = np.array(query_rep, dtype=float)
    candidates_rep_arrays = np.array(candidates_reps, dtype=float)
    distances = [cosine(query_rep_array, other) for other in candidates_rep_arrays]
    similarities = [1 - dist for dist in distances]  # 相似度 = 1 - 距离
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return top_k_indices, [similarities[i] for i in top_k_indices]


base_url = "http://36.213.0.171:9997/v1/"
api_key = 'test'
client = OpenAI(api_key=api_key, base_url=base_url)
model_name = 'Qwen2-VL-72B-Instruct'


def llm_predict(messages: list, model_name='Qwen2-VL-72B-Instruct') -> str:
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    return chat_completion.choices[0].message.content


def query_rewrite(original_question: str, sub_question: str, contexts: str,
                  model_name='Qwen2-VL-72B-Instruct') -> str:
    instruction = """
    **task:**
    Rewrite the current sub-question based on the provided context, clarify the query intent, and generate a logically coherent question sentence.
    
    **Original Full Question:**
    {original_question}

    **Context:**
    {contexts}

    **Current Sub-question:**
    {sub_question}
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": instruction.format(
                    contexts=contexts,
                    original_question=original_question,
                    sub_question=sub_question,
                )
            }
        ],
        model=model_name,
    )
    return chat_completion.choices[0].message.content


store_path = '/data1/zch/MultiModelDocRag/generate_augment/dag/dev/prediction_qwen2_vl_7b_dev_v1.jsonl'
dataset_dict = read_json('/data1/zch/MultiModelDocRag/embedding/eval/dataset_dict.json')

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

    image_doc_ids = metadata['image_doc_ids']
    text_doc_ids = metadata['text_doc_ids']
    table_id = metadata['table_id']
    objects = []
    for idx in image_doc_ids:
        objects.append(dataset_dict[idx])
    for idx in text_doc_ids:
        objects.append(dataset_dict[idx])
    objects.append(dataset_dict[table_id])
    objects_embeddings = asyncio.run(get_multiple_embeddings_async(objects))
    graph_dict = dict()

    for node in graph:
        node['contexts'] = []
        node['answer'] = ""
        graph_dict[node['node_name']] = node
    graph_logs = []
    for node in graph:
        node_name = node['node_name']
        query = node['query']
        type = node['type']
        dependencies = node['dependencies']
        if not dependencies:  # 不存在需要解决的依赖
            if node['type'] == 'retrieval':
                query_emb = get_embedding(query)
                top_k_indices, scores = find_top_k_similar_candidates(query_emb, objects_embeddings, k=3)
                corpus = [objects[i] for i in top_k_indices]
            else:
                corpus = []
            node['contexts'].extend(corpus)
            messages = get_normal_QA_prompt(question=query, corpus=corpus)
            node['answer'] = llm_predict(messages=messages, model_name=model_name)
        else:  # 存在依赖未解决
            for dependency in dependencies:
                if dependency in graph_dict:
                    node['contexts'].append(
                        f"**{graph_dict[dependency]['node_name']}**:{graph_dict[dependency]['query']}:{graph_dict[dependency]['answer']}\n")
            query = query_rewrite(original_question=question, sub_question=query, contexts=node['contexts'],
                                  model_name=model_name)
            node['contexts'] = []
            node['rewritten_query'] = query
            # if node['type'] == 'retrieval':
            query_emb = get_embedding(query)
            top_k_indices, scores = find_top_k_similar_candidates(query_emb, objects_embeddings, k=3)
            corpus = [objects[i] for i in top_k_indices]
            # else:
            #     corpus = []
            node['contexts'].extend(corpus)
            messages = get_normal_QA_prompt(question=query, corpus=corpus)
            node['answer'] = llm_predict(messages=messages, model_name=model_name)

        graph_dict[node_name] = node
        node['contexts'] = [context['id'] for context in node['contexts']]
        graph_logs.append(node)
    f.write(json.dumps(
        {
            "qid": qid,
            "question": question,
            "answer": answer,
            "graph_logs": graph_logs,
            "supporting_context": supporting_context,
            "metadata": metadata,
        }
    ) + "\n")
    f.flush()
f.close()

# import instructor
# from pydantic import BaseModel
# from openai import OpenAI
# from typing import List
#
#
# class GraphNode(BaseModel):
#     name: str
#     query: str
#     type: str
#     dependencies: list[str]
#
#
# class Dag(BaseModel):
#     graph_nodes: List[GraphNode]
#
#
# # base_url="http://36.213.0.171:9997/v1"
# base_url = "http://36.137.79.97:30250/v1"
# client = instructor.from_openai(OpenAI(base_url=base_url, api_key='test'))
# # model = "Qwen2-VL-72B-Instruct-0"
# model = "qwen2-72b-instruct"
#
# path = '/data1/zch/MultiModelDocRag/generate_augment/dag/dag_query_72b_dev_v1.jsonl'
# data = read_jsonl(path)
#
# store_path = '/data1/zch/MultiModelDocRag/generate_augment/dag/dag_query_72b_dev_v2.jsonl'
# f, processed_results = get_output_file(store_path, force=False)
#
# for idx, item in enumerate(tqdm(data)):
#     qid = item['qid']
#     if qid in processed_results:
#         continue
#     question = item['question']
#     dag = item['dag']
#     res = client.chat.completions.create(
#         model=model,
#         response_model=Dag,
#         messages=[{"role": "user", "content": dag}],
#     )
#     graph = []
#     for node in res.graph_nodes:
#         graph.append(
#             {
#                 "node_name": node.name,
#                 "query": node.query,
#                 "type": node.type,
#                 "dependencies": node.dependencies,
#             }
#         )
#     f.write(json.dumps({"qid": qid, "question": question, "graph": graph}) + "\n")
#     f.flush()
# f.close()

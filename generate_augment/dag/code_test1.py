# -*- coding: utf-8 -*-
# @Time    : 2024/12/13 15:05
# @Author  : blue
# @Description :
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *

import instructor
from pydantic import BaseModel
from openai import OpenAI
from typing import List
import json
import re

def parse_query_plan(input_string):
    blocks = re.findall(r"\{.*?}", input_string, re.DOTALL)

    parsed_blocks = []
    for block in blocks:
        try:
            parsed_blocks.append(json.loads(block))
        except json.JSONDecodeError as e:
            print(f"Error parsing block: {block}\nError: {e}")

    return parsed_blocks


class GraphNode(BaseModel):
    name: str
    content: str
    operation: str = "retrieval"
    dependencies: list[str]


class Dag(BaseModel):
    graph_nodes: List[GraphNode]


# base_url="http://36.213.0.171:9997/v1"
base_url = "http://36.137.79.97:30250/v1"
client = instructor.from_openai(OpenAI(base_url=base_url, api_key='test'))
# model = "Qwen2-VL-72B-Instruct-0"
model = "qwen2-72b-instruct"

path = '/data1/zch/MultiModelDocRag/generate_augment/dag/dev_dag_query_qwen2vl_72b_few_shots.jsonl'
# path = '/data1/zch/MultiModelDocRag/generate_augment/dag/dag_query_qwen2.5_72b_dev_v1.jsonl'
data = read_jsonl(path)

dataset_path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
dataset = read_jsonl(dataset_path)

dataset_dict = dict()
for item in dataset:
    dataset_dict[item['qid']] = item

store_path = '/data1/zch/MultiModelDocRag/generate_augment/dag/dag_query_72b_dev_v1.jsonl'
f, processed_results = get_output_file(store_path, force=False)

for idx, item in enumerate(tqdm(data)):
    qid = item['qid']
    if qid in processed_results:
        continue
    # answer = item['answers'][0]['answer']
    answer = item['answer']
    question = item['question']
    dag = item['dag']
    # res = client.chat.completions.create(
    #     model=model,
    #     response_model=Dag,
    #     messages=[{"role": "user", "content": dag}],
    # )
    plan_nodes = parse_query_plan(dag)
    graph = []
    # for node in res.graph_nodes:
    #     graph.append(
    #         {
    #             "node_name": node.name,
    #             "query": node.content,
    #             "type": node.operation,
    #             "dependencies": node.dependencies,
    #         }
    #     )
    for node in plan_nodes:
        graph.append(
            {
                "node_name": node.get('name',''),
                "query": node.get('query',''),
                "type": node.get('type',''),
                "dependencies": node.get('dependencies',''),
            }
        )
    f.write(json.dumps(
        {
            "qid": qid,
            "answer": answer,
            "question": question,
            "graph": graph,
            "metadata": dataset_dict[qid]['metadata'],
            "supporting_context": dataset_dict[qid]['supporting_context'],
        }
    ) + "\n")
    f.flush()
f.close()



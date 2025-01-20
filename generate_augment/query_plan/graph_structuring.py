# -*- coding: utf-8 -*-
# @Time    : 2024/12/27 21:09
# @Author  : blue
# @Description : 
import os
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"
import instructor
from pydantic import BaseModel
from openai import OpenAI
from util.api import *

import json


class GraphNode(BaseModel):
    name: str
    query: str
    dependencies: list[str]


class Dag(BaseModel):
    graph_nodes: List[GraphNode]


# model = "qwen2.5-instruct"
# base_url = "http://36.213.0.171:9997/v1/"

base_url = "http://36.137.79.97:30250/v1"
model = "qwen2-72b-instruct"

client = instructor.from_openai(OpenAI(base_url=base_url, api_key='test'))

# original_data = read_jsonl('/data1/zch/MultiModelDocRag/generate_augment/query_plan/dev_data_qwen2.5-instruct_v1.jsonl')
original_data = read_jsonl('/data1/zch/MultiModelDocRag/generate_augment/query_plan/train_data_qwen2.5-instruct_v1.jsonl')
store_path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/formated_train_qwen2.5-instruct_v1.jsonl'
f, processed = get_output_file(store_path, force=False)
res = []
for item in tqdm(original_data):
    if item['qid'] in processed:
        continue
    try:
        json_objects = re.findall(r'{.*?}', item['dag'], re.DOTALL)
        parsed_objects = [json.loads(obj) for obj in json_objects]
        item['graph'] = parsed_objects
        f.write(json.dumps(item) + "\n")
        f.flush()
    except json.JSONDecodeError:
        res = client.chat.completions.create(
            model=model,
            response_model=Dag,
            messages=[{"role": "user", "content": item['dag']}],
        )
        graph = []
        for node in res.graph_nodes:
            graph.append(
                {
                    "name": node.name,
                    "query": node.query,
                    "dependencies": node.dependencies,
                }
            )
        item['graph'] = graph
        f.write(json.dumps(item) + "\n")
        f.flush()
f.close()

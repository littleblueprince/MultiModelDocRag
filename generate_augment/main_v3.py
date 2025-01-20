# -*- coding: utf-8 -*-
# @Time    : 2024/12/28 15:17
# @Author  : blue
# @Description :
import os

import aiohttp
import instructor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

import requests
from util.retrieval_api import *
from util.chat_template import *
from pydantic import BaseModel
from openai import OpenAI
from util.api import *
from typing import List
import json


class GraphNode(BaseModel):
    name: str
    query: str
    action: str
    dependencies: list[str]


class Dag(BaseModel):
    graph_nodes: List[GraphNode]


def get_texts_embeddings(texts: list,
                         emb_base_url='http://10.176.40.88:9997/v1/embeddings',
                         embedding_model="bge_m3_cuda0",
                         emb_api_key='test'):
    emb_headers = {
        'Authorization': f'Bearer {emb_api_key}',
        'Content-Type': 'application/json',
    }
    vectors = requests.post(emb_base_url, headers=emb_headers, json={
        "model": embedding_model,
        "input": texts,
    }).json()
    return [vec['embedding'] for vec in vectors['data']]


def get_embedding_sync(text: str, url: str = "http://36.213.0.171:8001/encode/"):
    data = {
        "text": text,
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            json_response = response.json()
            return {
                "embedding": json_response["embedding"][0]
            }
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        return {"error": str(e)}


class DagReasoningSystem:
    def __init__(self,
                 qid: str,
                 original_question: str,
                 ground_truth: str,
                 metadata: list,
                 supporting_context: list,
                 plan: list,
                 ):
        # self.base_url = "https://api.deepseek.com/v1"
        # self.model = "deepseek-chat"
        # self.api_key = 'sk-550e97b86ab9414f9832694805f361ca'
        # self.base_url = "http://36.213.0.171:9997/v1"
        # self.model = "Qwen2-VL-72B-Instruct"
        # self.api_key = 'test'
        # self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        # self.model = "qwen2-vl-72b-instruct"
        # self.api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
        self.api_key = 'sk-BLTI0o3z4R8GF8JNOHUuTmqtx6oLTt1SpeCXezoHwPPf4v5P'
        self.base_url = "https://yunwu.ai/v1"
        self.model = "gpt-4o-mini-2024-07-18"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.qid = qid
        self.original_question = original_question
        self.ground_truth = ground_truth
        self.metadata = metadata
        self.text_doc_ids = metadata['text_doc_ids']
        self.image_doc_ids = metadata['image_doc_ids']
        self.table_doc_ids = [metadata['table_id']]
        self.text_collection = 'text_v1'
        # self.image_collection = 'img_v1'
        self.image_collection = 'img_v2'
        self.table_collection = 'tab_v1'
        self.supporting_context = supporting_context
        self.plan_examples_collection_name = 'dag_v2'
        self.topk = 3
        self.plan = []
        self.solved = dict()
        self.graph_logs = []
        self.plan = plan

    def node_execute(self, node: dict) -> dict:
        node_name = node['name']
        query = node['query']
        action = node['action']
        dependencies = node['dependencies']
        if dependencies:
            history_contexts = ''
            for dependency in dependencies:
                if dependency in self.solved:
                    history_contexts += self.solved[dependency]
            rewrite_query = self.query_rewrite(
                original_question=self.original_question,
                sub_question=query,
                contexts=history_contexts,
            )

            query_vector = get_embedding_sync(text=rewrite_query)['embedding']
            if action == "text_retrieval":
                corpus = text_retrieval_corpus(
                    query_vector=query_vector,
                    text_filtered_ids=self.text_doc_ids,
                    tabs_filtered_ids=self.table_doc_ids,
                    text_collection_name=self.text_collection,
                    tabs_collection_name=self.table_collection,
                )
            elif action == "table_retrieval":
                corpus = table_retrieval_corpus(
                    query_vector=query_vector,
                    tabs_filtered_ids=self.table_doc_ids,
                    tabs_collection_name=self.table_collection,
                )
            elif action == "image_retrieval":
                corpus = image_retrieval_corpus(
                    query_vector=query_vector,
                    imgs_filtered_ids=self.image_doc_ids,
                    imgs_collection_name=self.image_collection,
                )
            else:  # general_retrieval
                corpus = retrieval_corpus(
                    query_vector=query_vector,
                    text_filtered_ids=self.text_doc_ids,
                    tabs_filtered_ids=self.table_doc_ids,
                    imgs_filtered_ids=self.image_doc_ids,
                    text_collection_name=self.text_collection,
                    tabs_collection_name=self.table_collection,
                    imgs_collection_name=self.image_collection,
                )
            corpus = corpus[:self.topk]
            messages = get_normal_QA_prompt_v2(original_question=self.original_question,
                                               sub_question=rewrite_query,
                                               corpus=corpus,
                                               )
            output = self.llm_predict(messages=messages)
            self.solved[node_name] = f'**Sub Question:**{rewrite_query}\n**Answer:**{output}'
            return {
                "answer": output,
                "corpus": [c['metadata']['id'] for c in corpus],
                "rewrite_query": rewrite_query,
            }
        else:
            query_vector = get_embedding_sync(text=query)['embedding']
            if action == "text_retrieval":
                corpus = text_retrieval_corpus(
                    query_vector=query_vector,
                    text_filtered_ids=self.text_doc_ids,
                    tabs_filtered_ids=self.table_doc_ids,
                    text_collection_name=self.text_collection,
                    tabs_collection_name=self.table_collection,
                )
            elif action == "image_retrieval":
                corpus = image_retrieval_corpus(
                    query_vector=query_vector,
                    imgs_filtered_ids=self.image_doc_ids,
                    imgs_collection_name=self.image_collection,
                )
            else:  # general_retrieval
                corpus = retrieval_corpus(
                    query_vector=query_vector,
                    text_filtered_ids=self.text_doc_ids,
                    tabs_filtered_ids=self.table_doc_ids,
                    imgs_filtered_ids=self.image_doc_ids,
                    text_collection_name=self.text_collection,
                    tabs_collection_name=self.table_collection,
                    imgs_collection_name=self.image_collection,
                )
            corpus = corpus[:self.topk]
            messages = get_normal_QA_prompt_v2(original_question=self.original_question,
                                               sub_question=query,
                                               corpus=corpus,
                                               )
            output = self.llm_predict(messages=messages)
            self.solved[node_name] = f'**Sub Question:**{query}\n**Answer:**{output}'
            return {
                "answer": output,
                "corpus": [c['metadata']['id'] for c in corpus],
            }

    def llm_predict(self, messages: list) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
        )
        output = chat_completion.choices[0].message.content
        return output

    def query_rewrite(self, original_question: str, sub_question: str, contexts: str) -> str:
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
        chat_completion = self.client.chat.completions.create(
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
            model=self.model,
        )
        return chat_completion.choices[0].message.content

    def format_plan_str_to_graph(self, plan_str: str) -> list:
        try:
            json_objects = re.findall(r'{.*?}', plan_str, re.DOTALL)
            graph = [json.loads(obj) for obj in json_objects]
            return graph
        except json.JSONDecodeError:
            base_url = "http://36.137.79.97:30250/v1"
            model = "qwen2-72b-instruct"
            api_key = 'test'
            client = instructor.from_openai(OpenAI(
                api_key=api_key,
                base_url=base_url,
            ))
            res = client.chat.completions.create(
                model=model,
                response_model=Dag,
                messages=[{"role": "user", "content": plan_str}],
            )
            graph = []
            for node in res.graph_nodes:
                graph.append(
                    {
                        "name": node.name,
                        "query": node.query,
                        "action": node.action,
                        "dependencies": node.dependencies,
                    }
                )
            return graph

    def save_log_partly(self) -> dict:
        return {
            "qid": self.qid,
            "original_question": self.original_question,
            "ground_truth": self.ground_truth,
            "graph_logs": self.graph_logs,
            "supporting_context": self.supporting_context,
            "type": self.metadata['type'],
        }

    def save_log_fully(self) -> dict:
        return {
            "qid": self.qid,
            "original_question": self.original_question,
            "ground_truth": self.ground_truth,
            "graph_logs": self.graph_logs,
            "supporting_context": self.supporting_context,
            "metadata": self.metadata,
        }

    # Validate results
    def validate_results(self, query_result):
        if not query_result or not query_result.get("result"):
            return False
        return True

    # Dynamic adjustment
    def adjust_plan(self, query_name):
        if query_name == "Q1":
            self.plan.append(
                {"name": "Q1.1", "query": "What is Dzenis Beganovic's transfer history?", "type": "retrieval",
                 "dependencies": []})
        elif query_name == "Q2":
            self.plan.append(
                {"name": "Q2.1", "query": "What was Tuzla City's season performance in 2020-2021?", "type": "retrieval",
                 "dependencies": ["Q1"]})

    # Main execution loop
    def execute_plan(self):
        for node in self.plan:
            output = self.node_execute(node)
            node['answer'] = output['answer']
            node['corpus'] = output['corpus']
            if "rewrite_query" in output:
                node['rewrite_query'] = output['rewrite_query']
            self.graph_logs.append(node)
        return self.save_log_partly(), self.save_log_fully()


path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
dataset = read_jsonl(path)

plan_dataset = read_jsonl('/data1/zch/MultiModelDocRag/generate_augment/query_plan/dag_plan_qwen25_7b_dev_v1.jsonl')
for item in dataset:
    for d in plan_dataset:
        if item['question'] in d['prompt']:
            item['plan'] = json.loads(d['predict'])
            break

store_path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/prediction_qwen2_vl_72b_dev_v4.jsonl'
f, processed = get_output_file(store_path, force=False)
for item in tqdm(dataset):
    if item["qid"] in processed:
        continue
    try:
        reasoning_system = DagReasoningSystem(
            qid=item["qid"],
            original_question=item["question"],
            ground_truth=item["answers"][0]['answer'],
            metadata=item["metadata"],
            supporting_context=item["supporting_context"],
            plan=item["plan"],
        )
        part_log, full_log = reasoning_system.execute_plan()
        f.write(json.dumps(
            part_log
        ) + "\n")
        f.flush()
    except:
        continue
f.close()

# -*- coding: utf-8 -*-
# @Time    : 2024/12/28 15:17
# @Author  : blue
# @Description :
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

import requests
from util.retrieval_api import *
from openai import OpenAI
from util.api import *
import json

# class GraphNode(BaseModel):
#     name: str
#     query: str
#     action: str
#     dependencies: list[str]
#
#
# class Dag(BaseModel):
#     graph_nodes: List[GraphNode]


"""
获取问答的prompt，
如果是纯文本 ？
如果是图片 ？
如果是表格 ？
"""


def get_QA_prompt(question: str, corpus: list) -> list:
    instruction = '**Instructions:** Based on the provided reference information, first describe the content. Then carefully extract and present all relevant data in a coherent statement. Begin the final answer with the special tag <answer>. If the reference does not provide sufficient information, state: "Reference does not provide the required information." Do not use external knowledge.'
    user_inputs = []
    user_inputs.append({"type": "text", "text": instruction})
    user_inputs.append({"type": "text", "text": "**Reference Documents:**  \n"})
    for doc in corpus:
        text = doc['content']
        if 'path' in doc['metadata']:
            file_path = doc['metadata']['path']
            user_inputs.append({"type": "text", "text": text})
            user_inputs.append({
                "type": "image_url",
                "image_url":
                    {
                        "url": f"data:image/jpeg;base64,{get_base64_img(file_path, img_dir='/data1/zch/datasets/multimodalqa/final_dataset_images/')}"
                    }
            }
            )
        else:
            user_inputs.append({"type": "text", "text": text})

    user_inputs.append({"type": "text", "text": f"**Question:**  \n{question}"})
    messages = [
        {
            "role": "user",
            "content": user_inputs,
        }
    ]
    return messages


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
                 global_question: str,
                 ground_truth: str,
                 metadata: list,
                 supporting_context: list,
                 ):
        self.plan_base_url = "http://36.213.0.171:9997/v1"
        self.plan_model = "qwen2.5-instruct"
        self.plan_api_key = 'test'
        self.plan_client = OpenAI(
            api_key=self.plan_api_key,
            base_url=self.plan_base_url,
        )

        # self.text_ans_base_url = "http://36.213.0.171:9997/v1"
        # self.text_ans_model = "qwen2.5-instruct-Jz780nQ2"
        # self.text_ans_api_key = 'test'
        # self.text_ans_client = OpenAI(
        #     api_key=self.text_ans_api_key,
        #     base_url=self.text_ans_base_url,
        # )

        # self.ans_base_url = "http://36.213.0.171:9997/v1"
        # self.ans_model = "qwen2-vl-instruct"
        # self.ans_api_key = 'test'

        self.ans_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.ans_model = "qwen2-vl-72b-instruct"
        self.ans_api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
        self.ans_client = OpenAI(
            api_key=self.ans_api_key,
            base_url=self.ans_base_url,
        )

        self.val_base_url = "https://api.deepseek.com/v1"
        self.val_model = "deepseek-chat"
        self.val_api_key = 'sk-550e97b86ab9414f9832694805f361ca'
        self.val_client = OpenAI(
            api_key=self.val_api_key,
            base_url=self.val_base_url,
        )

        self.qid = qid
        self.global_question = global_question
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
        self.dependence_existence_flag = True
        self.topk = 3
        self.solved = dict()
        self.graph_logs = []
        self.known_content = ""
        self.failure_operation_records = ""

    def get_plan_nodes(self, question, known_content='', failure_operation_records='') -> list:
        instruction = 'Please plan a problem retrieval and reasoning solution based on the following questions, known content, and query failure operation records.'
        input_prompt = '**Question:**{question}\n**Known Content:**{known_content}\n**Failure Operation Records:**{failure_operation_records}'
        self.dependence_existence_flag = False
        chat_completion = self.plan_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": instruction + input_prompt.format(question=question, known_content=known_content,
                                                                 failure_operation_records=failure_operation_records),
                }
            ],
            model=self.plan_model,
            temperature=0.7,
            top_p=0.8,
            stream=True,
        )
        # 逐步读取并解析流式响应
        output_str = ""
        nodes = []
        for chunk in chat_completion:
            delta = chunk.choices[0].delta
            content = delta.content
            if content:
                output_str += content
                # print(content, end="")  # end="" 确保内容逐步打印在同一行
                if '</NO_DEPENDENCY>' in output_str:
                    temp_output_str = output_str.split('<NO_DEPENDENCY>')[1].split('</NO_DEPENDENCY>')[0].replace('\\"',
                                                                                                                  '"')
                    nodes.append(json.loads(temp_output_str))
                    output_str = output_str.split('</NO_DEPENDENCY>')[1]
                elif '<HAS_DEPENDENCY>' in output_str:
                    # print('仍然存在被依赖问题需要解决')
                    self.dependence_existence_flag = True
                    break

        return nodes

    def node_execute(self, node: dict) -> dict:
        node_name = node['name']
        query = node['query']
        action = node['action']
        dependencies = node['dependencies']
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
        messages = get_QA_prompt(question=query, corpus=corpus, )
        output = self.llm_predict(messages=messages)
        if '<answer>' in output:
            output = output.split('<answer>')[1]
        elif 'answer' in output:
            output = output.split('answer')[1]
        return {
            "answer": output,
            "corpus": [c['metadata']['id'] for c in corpus],
        }

    def llm_predict(self, messages: list) -> str:
        chat_completion = self.ans_client.chat.completions.create(
            messages=messages,
            model=self.ans_model,
        )
        output = chat_completion.choices[0].message.content
        return output

    def save_log_partly(self) -> dict:
        return {
            "qid": self.qid,
            "global_question": self.global_question,
            "ground_truth": self.ground_truth,
            "graph_logs": self.graph_logs,
            "known_content": self.known_content,
            "failure_operation_records": self.failure_operation_records,
            "supporting_context": self.supporting_context,
            "type": self.metadata['type'],
        }

    def save_log_fully(self) -> dict:
        return {
            "qid": self.qid,
            "global_question": self.global_question,
            "ground_truth": self.ground_truth,
            "graph_logs": self.graph_logs,
            "supporting_context": self.supporting_context,
            "metadata": self.metadata,
        }

    # Validate results
    def validate_results(self, input_string: str) -> bool:
        instruction = "Does the input contain a phrase like 'The reference does not provide the required information'? Output yes if it does, and no if it does not. Output only 'yes' or 'no'. **Input string:**{input_string}"
        chat_completion = self.val_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": instruction.format(input_string=input_string),
                }
            ],
            model=self.val_model,
        )
        output = chat_completion.choices[0].message.content
        if 'yes' == output or 'yes' in output:
            return True
        else:
            return False

    # Main execution loop
    def execute_plan_and_solve(self):
        count = 0
        while self.dependence_existence_flag:
            query_nodes = self.get_plan_nodes(
                question=self.global_question,
                known_content=self.known_content,
                failure_operation_records=self.failure_operation_records,
            )
            for node in query_nodes:
                node_res = self.node_execute(node)
                node['name'] = 'loop-' + str(count) + '-' + node['name']
                node['answer'] = node_res['answer']
                node['corpus'] = node_res['corpus']
                val_res = self.validate_results(input_string=node['answer'])
                if val_res:
                    # 查询失败
                    fail_record = 'sub query: {query} search failed in query mode: {action}'
                    self.dependence_existence_flag = True
                    self.failure_operation_records += fail_record.format(query=node['answer'], action=node['action'])
                else:
                    self.known_content += f'{node["answer"]}\n'
                self.graph_logs.append(node)
        return self.save_log_partly(), self.save_log_fully()


path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
dataset = read_jsonl(path)[5:]
store_path = './prediction_dev_v1.jsonl'
f, processed = get_output_file(store_path, force=False)
for item in tqdm(dataset):
    if item["qid"] in processed:
        continue
    try:
        reasoning_system = DagReasoningSystem(
            qid=item["qid"],
            global_question=item["question"],
            ground_truth=item["answers"][0]['answer'],
            metadata=item["metadata"],
            supporting_context=item["supporting_context"],
        )
        part_log, full_log = reasoning_system.execute_plan_and_solve()
        f.write(json.dumps(
            part_log
        ) + "\n")
        f.flush()
    except Exception as e:
        print(str(e))
f.close()

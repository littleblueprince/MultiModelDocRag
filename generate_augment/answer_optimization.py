# # -*- coding: utf-8 -*-
# # @Time    : 2025/01/02 15:56
# # @Author  : blue
# # @Description :
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["HF_HOME"] = "/data1/zch/tmp"
# os.environ["TMPDIR"] = "/data1/zch/tmp"
#
# from util.api import *
#
# from openai import OpenAI
#
# # base_url = "http://36.137.79.97:30250/v1/"
# # model = "qwen2-72b-instruct"
# # api_key = 'test'
#
# api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
# base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# model = "qwen2.5-72b-instruct"
#
# # api_key = 'sk-BLTI0o3z4R8GF8JNOHUuTmqtx6oLTt1SpeCXezoHwPPf4v5P'
# # base_url="https://yunwu.ai/v1"
# # model="gpt-4o-mini-2024-07-18"
#
# client = OpenAI(
#     api_key=api_key,
#     base_url=base_url,
# )
#
# path = '/data1/zch/MultiModelDocRag/util/eval/eval_result_v6.jsonl'
# store_path = '/data1/zch/FlagEmbedding/mycode/prediction_dev_v6.jsonl'
# data = read_jsonl(path)
# res = []
# for item in tqdm(data):
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"Provide only the essential answer to the following input. For yes or no questions, respond with 'yes' or 'no'. For other questions, return only the key entity or factual information without any additional explanation. **Question:** {item['question']} **Prediction:** {item['prediction']}",
#             }
#         ],
#         model=model,
#         temperature=0,
#     )
#     res.append(
#         {
#             "id": item['qid'],
#             "question": item['question'],
#             "answers": [{"answer": item['ground_truth']}],
#             "prediction": chat_completion.choices[0].message.content,
#         }
#     )
# store_jsonl(store_path, res)



import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *

from openai import OpenAI


api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen2.5-72b-instruct"


client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/prediction_qwen2_vl_72b_dev_v4.jsonl'
store_path = '/data1/zch/FlagEmbedding/mycode/prediction_dev_v6.jsonl'
data = read_jsonl(path)
res = []
for item in tqdm(data):
    original_question = item['original_question']
    graph_logs = item['graph_logs']
    history = ''
    for node in graph_logs:
        if "rewrite_query" in node:
            sub_query = node['rewrite_query']
        else:
            sub_query = node['query']
        sub_answer = node['answer']
        history += f"{sub_query}\n{sub_answer}\n"
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f'Your task is to extract a concise answer output based on the provided conversation history, without the need for additional explanation. For yes or no questions, please answer "yes" or "no"Your task is to extract a concise answer output based on the provided conversation history, without the need for additional explanation. For yes or no questions, please answer "yes" or "noff". For other questions, only key entities or factual information are returned without any other explanation. **Global Questions:**{original_question}**Reference Information:**{history}',
            }
        ],
        model=model,
        temperature=0,
    )
    res.append(
        {
            "id": item['qid'],
            "question": item['original_question'],
            "answers": [{"answer": item['ground_truth']}],
            "prediction": chat_completion.choices[0].message.content,
        }
    )
store_jsonl(store_path, res)

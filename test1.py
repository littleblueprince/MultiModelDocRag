# # -*- coding: utf-8 -*-
# # @Time    : 2024/12/05 19:50
# # @Author  : blue
# # @Description :
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *

path='/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
data=read_jsonl(path)
for item in data:
    if "mage" in item['metadata']['type']:
        count = 0
        for s in item['supporting_context']:
            if s['doc_part'] == 'image':
                count+=1
        if count>1:
            print(item['qid'])

# data1=read_jsonl('/data1/zch/MultiModelDocRag/generate_augment/query_plan/prediction_qwen2_vl_72b_dev_v3.jsonl')
# data2=read_jsonl('/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl')
# pre_dict=dict()
# for item in data1:
#     pre_dict[item['qid']]=item['graph_logs'][-1]['answer']
# res=[]
# for item in data2:
#     if item['qid'] in pre_dict:
#         res.append(
#             {
#                 "id": item['qid'],
#                 "question": item['question'],
#                 "answers": item['answers'],
#                 "prediction": pre_dict[item['qid']]
#             }
#         )
# store_jsonl('/data1/zch/FlagEmbedding/mycode/prediction_dev_v1.jsonl',res)

# data1 = read_jsonl('/data1/zch/MultiModelDocRag/generate_augment/query_plan/train_data_qwen2.5-72b-instruct_v1.jsonl')
# data2 = read_jsonl('/data1/zch/MultiModelDocRag/generate_augment/query_plan/train_data_v2.jsonl')
# data3 = read_jsonl('/data1/zch/MultiModelDocRag/generate_augment/query_plan/train_data_qwen2.5-72b-instruct_v3.jsonl')
#
# res = []
# for item in (data1 + data2 + data3):
#     res.append(
#         {
#             "instruction": "Your task is to specify a question retrieval reasoning plan for the following question.",
#             "input": f"**Question:**{item['question']}",
#             "output": item['dag'],
#         }
#     )
# store_path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/plan_train_v1.jsonl'
# store_jsonl(store_path, res)

# data = read_jsonl('/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl')
# res = []
# for item in data:
#     res.append(
#         {
#             "instruction": "Your task is to specify a question retrieval reasoning plan for the following question.",
#             "input": f"**Question:**{item['question']}",
#             "output": "",
#         }
#     )
# store_path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/plan_dev_v1.jsonl'
# store_jsonl(store_path, res)

# path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/prediction_qwen2_vl_72b_dev_v3.jsonl'
# store_path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/dag_plan_qwen2-5_72b_dev_v1.jsonl'
#
# dataset=read_jsonl('/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl')
# dataset_dict=dict()
# for item in dataset:
#     dataset_dict[item['qid']]=item['metadata']
#
# res =[]
# for item in read_jsonl(path):
#     dag=[
#         {
#             "name": g['name'],
#             "query": g['query'],
#             "action": g['action'],
#             "dependencies": g['dependencies'],
#         }
#         for g in item['graph_logs']
#     ]
#     res.append(
#         {
#             "qid":item['qid'],
#             "question":item['original_question'],
#             "ground_truth":item['ground_truth'],
#             "dag":dag,
#             "supporting_context":item['supporting_context'],
#             "metadata":dataset_dict[item['qid']],
#         }
#     )
#
#
# store_jsonl(store_path,res)

# query_path = '/data1/zch/datasets/multimodalqa/MMQA_train.jsonl'
# # dataset_path = '/data1/zch/MultiModelDocRag/generate_augment/dataset.json'
# imgs_path = '/data1/zch/datasets/multimodalqa/MMQA_images.jsonl'
# # path = '/data1/zch/FlagEmbedding/mycode/milvus_corpus_part_train_v1.json'
# # path='/data1/zch/FlagEmbedding/research/visual_bge/visual_bge/data'
# query_data = read_jsonl(query_path)
# # dataset_dict = read_json(dataset_path)
# imgs_data = read_jsonl(imgs_path)
# imgs_dict = dict()
# for item in imgs_data:
#     imgs_dict[item['id']] = item
#
# result = []
# for item in tqdm(query_data):
#     qid = item['qid']
#     question = item['question']
#     answers = item['answers']
#     metadata = item['metadata']
#     supporting_context = item['supporting_context']  # doc_id doc_part
#     supporting_ids = [context['doc_id'] for context in supporting_context]
#     image_doc_ids = metadata['image_doc_ids']
#     for image_doc_id in image_doc_ids:
#         result.append(
#             {
#                 "question": question,
#                 "text_candidate": imgs_dict[image_doc_id]['title'],
#                 "img_candidate": imgs_dict[image_doc_id]['path'],
#                 "label": 1 if image_doc_id in supporting_ids else 0,
#             }
#         )
#
# store_path = '/data1/zch/MultiModelDocRag/embedding/train_v1.jsonl'
# store_jsonl(store_path, result)
# #
# # # path=''
# # # data=read_json()
# # #
# # # print(11)
# #
# #
# # from transformers import AutoModel
# #
# # # model_name = '/data1/zch/models/bge-large-en-v1.5'
# # # model_name = '/data1/zch/models/bge-base-en-v1.5'
# # # model_name = '/data1/zch/models/bge-m3'
# # model_name = '/data1/zch/models/Qwen2-VL-2B-Instruct'
# #
# # model = AutoModel.from_pretrained(model_name)
# # print(model)
# #
# #
# # import base64
# #
# # import openai
# #
# # image_path = "/data1/zch/datasets/multimodalqa/final_dataset_images/82cde12db73e69dabf66bd84a8455b1e.jpg"
# # with open(image_path, "rb") as img_file:
# #     img_data = img_file.read()
# #
# # b64_img = base64.b64encode(img_data).decode("utf-8")
# #
# # messages = [
# #     {
# #         "role": "user",
# #         "content": [
# #             {"type": "text", "text": "你是谁"},
# #             {
# #                 "type": "image_url",
# #                 "image_url": {
# #                     "url": f"data:image/jpeg;base64,{b64_img}",
# #                     #"url": "http",
# #                 },
# #             },
# #         ],
# #     }
# # ]
# #
# # client = openai.Client(api_key="api_key", base_url="http://36.213.0.171:9997/v1")
# #
# # completion = client.chat.completions.create(
# #     model="qwen2-vl-instruct",
# #     messages=messages,
# # )
# #
# # print(completion)

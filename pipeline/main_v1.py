# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 12:59
# @Author  : blue
# @Description : 调用api接口实现迭代式rag流程
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
from util.chat_template import *
from util.output_parse import *
import openai
from visual_bge.modeling import Visualized_BGE
from util.retrieval_api import retrieval_corpus
import torch
import instructor
from openai import OpenAI
from pydantic import BaseModel


# file_path = '/data1/zch/MultiModelDocRag/pipeline/data/dataset.json'
# data = read_json(file_path)


class Node1Output(BaseModel):
    answerability: bool
    direct_answer: str
    sub_query: str
    hypothetical_answer: str


qwen2_base_url = "http://36.137.79.97:30250/v1"
qwen2_model = "qwen2-72b-instruct"
qwen2_client = instructor.from_openai(OpenAI(api_key="api_key", base_url=qwen2_base_url))

path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
data = read_jsonl(path)

model = Visualized_BGE(
    model_name_bge="bge-base-en-v1.5",
    model_weight="/data1/zch/models/bge-visualized/Visualized_base_en_v1.5.pth",
    from_pretrained='/data1/zch/models/bge-base-en-v1.5'
)
model.eval()

client = openai.Client(api_key="api_key", base_url="http://36.213.0.171:9997/v1")


def process_item(item, topk=3, file_path="/data1/zch/MultiModelDocRag/pipeline/data/logs/{qid}_log.jsonl") -> dict:
    qid = item['qid']
    answers = item['answers']
    question = item['question']
    metadata = item['metadata']
    supporting_context = item['supporting_context']
    type = item['metadata']['type']
    text_filtered_ids = []
    tabs_filtered_ids = []
    imgs_filtered_ids = []

    for idx in metadata['image_doc_ids']:
        imgs_filtered_ids.append(idx)
    for idx in metadata['text_doc_ids']:
        text_filtered_ids.append(idx)
    tabs_filtered_ids.append(metadata['table_id'])

    # 初始化日志文件路径
    file_path = file_path.format(qid=qid)

    # 中间步骤记录
    loop = 5
    contexts = []  # 用来保存所有迭代的上下文

    # 写入初始日志
    write_incremental_log(
        file_path,
        {
            "step": "start",
            "qid": qid,
            "original_question": question,
            "supporting_context": supporting_context,
        }
    )

    while loop > 0:
        write_incremental_log(
            file_path,
            {
                "step": f"loop_{5 - loop + 1}",
                "original_question": question,
                "contexts": contexts
            }
        )
        # Step 1: 生成 node1
        node1_messages = get_node1_messages_v1(question=question, contexts=contexts)
        # node1_completion = client.chat.completions.create(
        #     model="qwen2-vl-instruct",
        #     messages=node1_messages,
        # )
        node1_structure_output = qwen2_client.chat.completions.create(
            model=qwen2_model,
            response_model=Node1Output,
            messages=node1_messages,
            # messages=[
            #     {
            #         "role": "user",
            #         "content": node1_completion.choices[0].message.content
            #     }
            # ],
        )

        # 记录 node1 输出

        write_incremental_log(file_path, {
            "answerability": node1_structure_output.answerability,
            "direct_answer": node1_structure_output.direct_answer,
            "sub_query": node1_structure_output.sub_query,
            "hypothetical_answer": node1_structure_output.hypothetical_answer,
        })

        if node1_structure_output.answerability:
            # 如果答案已经找到，直接返回
            final_answer = node1_structure_output.direct_answer
            write_incremental_log(file_path, {"direct_final_answer": final_answer})
            return {
                "qid": qid,
                "question": question,
                "ground_truth": answers[0]['answer'],
                "prediction": final_answer,
                "type": type,
            }
        else:
            sub_question = node1_structure_output.sub_query
            evidence1 = node1_structure_output.hypothetical_answer
            loop -= 1

        # Step 2: 查询相似文档
        write_incremental_log(file_path, {"sub_question": sub_question})

        with torch.no_grad():
            query_emb = model.encode(text=sub_question).detach().cpu()

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
        milvus_corpus_logs = [
            {
                "id": item['metadata']['id'],
                "score": item['score'],
            }
            for item in milvus_corpus
        ]
        write_incremental_log(file_path, {"retrieved_corpus": milvus_corpus_logs})
        node2_messages = get_node2_messages_v1(sub_question=sub_question, corpus=milvus_corpus)
        node2_completion = client.chat.completions.create(
            model="qwen2-vl-instruct",
            messages=node2_messages,
        )
        evidence2 = node2_completion.choices[0].message.content
        write_incremental_log(file_path,
                              {
                                  "evidence1": evidence1,
                                  "evidence2": evidence2,
                              }
                              )

        # Step 3: 融合知识
        node3_messages = get_node3_messages_v1(sub_question=sub_question, evidence1=evidence1, evidence2=evidence2)
        node3_completion = client.chat.completions.create(
            model="qwen2-vl-instruct",
            messages=node3_messages,
        )

        # 保存每次生成的内容副本，而不是直接修改contexts
        current_context = node3_completion.choices[0].message.content
        contexts.append(current_context)  # 将每轮生成的内容作为独立记录

        # 写入增量日志
        write_incremental_log(
            file_path,
            {
                "node3_output": current_context
            }
        )

    final_messages = get_final_messages_v1(question=question, contexts=contexts)
    final_completion = client.chat.completions.create(
        model="qwen2-vl-instruct",
        messages=final_messages,
    )
    final_answer = final_completion.choices[0].message.content

    write_incremental_log(file_path, {"out_loop_final_answer": final_answer})

    return {
        "qid": qid,
        "question": question,
        "ground_truth": answers[0]['answer'],
        "prediction": final_answer,
        "type": type,
    }


def write_incremental_log(file_path, log_data):
    """增量写入 JSONL 文件"""
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False)
            f.write("\n")  # 每个日志条目单独写一行
    except Exception as e:
        print(f"Error writing log: {e}")


store_path = "/data1/zch/MultiModelDocRag/pipeline/data/predictions.json"
predictions = []
for item in data:
    res = process_item(item)
    pretty_print_json(res)
    predictions.append(res)

store_json(store_path, predictions)


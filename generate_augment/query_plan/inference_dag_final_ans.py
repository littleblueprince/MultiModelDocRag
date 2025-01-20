# # -*- coding: utf-8 -*-
# # @Time    : 2024/12/11 20:59
# # @Author  : blue
# # @Description :
import os

os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.chat_template import *
from util.api import *
from util.retrieval_api import retrieval_corpus
import torch
from openai import OpenAI
from visual_bge.modeling import Visualized_BGE

path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/formated_dev_qwen2.5-instruct_v1.jsonl'
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


store_path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/prediction_qwen2_vl_72b_dev_v1.jsonl'

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

    for node in graph:
        node['contexts'] = []
        node['answer'] = ""
        graph_dict[node['name']] = node
    graph_logs = []
    for node in graph:
        node_name = node['name']
        query = node['query']
        dependencies = node['dependencies']
        if not dependencies:  # 不存在需要解决的依赖
            query_emb = get_embedding(query)
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

            node['contexts'].extend(corpus)
            messages = get_normal_QA_prompt_v1(question=query, corpus=corpus)
            node['answer'] = llm_predict(messages=messages, model_name=model_name)
        else:  # 存在依赖未解决
            for dependency in dependencies:
                if dependency in graph_dict:
                    node['contexts'].append(f"{graph_dict[dependency]['query']}:{graph_dict[dependency]['answer']}\n")
            query = query_rewrite(original_question=question, sub_question=query, contexts=node['contexts'],
                                  model_name=model_name)
            node['contexts'] = []
            node['rewritten_query'] = query
            query_emb = get_embedding(query)
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
            node['contexts'].extend(corpus)
            messages = get_normal_QA_prompt_v1(question=query, corpus=corpus)
            node['answer'] = llm_predict(messages=messages, model_name=model_name)

        graph_dict[node_name] = node
        node['contexts'] = [context['metadata']['id'] for context in node['contexts']]
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
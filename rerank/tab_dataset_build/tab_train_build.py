import os

"""
先输出表格描述，再输出针对表格和问题的思考，再输出是否需要的特殊token，再输出重点描述信息
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *


def get_output(desc: str, think: str, helpful: bool, key_points: str):
    desc_str = f'<description>{desc}</description>'
    think_str = f'<relevance_analysis>{think}</relevance_analysis>'
    if helpful:
        helpful_str = f'<decision><helpful></decision>'
    else:
        helpful_str = f'<decision><not_helpful></decision>'
    key_points_str = f'<key_information>{key_points}</key_information>'
    return desc_str + think_str + helpful_str + key_points_str


def get_tab_QA_messages(
        question: str,
        table_content: str,
        output: str,
) -> dict:
    instruction = """You will process an input question, a table. Please output the following information in the specified format:

    1. **Table Description** (<description>):
       Provide a concise description of the table based on the input content.

    2. **Relevance Analysis** (<relevance_analysis>):
       Analyze the relevance of the table to the given question and determine if the table is helpful in answering the question.

    3. **Decision** (<decision>):
       Based on the relationship between the table and the question, output either <helpful> (if the table is helpful) or <not_helpful> (if the table is not helpful).

    4. **Key Information** (<key_information>):
       If <decision> is <helpful>, extract the key information from the table and describe it concisely to ensure accurate question answering.
       If <decision> is <not_helpful>, output: This table cannot help answer the question.
    """
    input_template = """
    **Table Content:**
    {table_content}

    **Question:**
    {question}
    """
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {
            "role": "user",
            "content": input_template.format(question=question, table_content=table_content),
        },
        {
            "role": "assistant",
            "content": output,
        }
    ]
    return {
        "messages": messages,
    }


query_data = read_jsonl('/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl')
# query_data = read_jsonl('/data1/zch/MultiModelDocRag/rerank/tab_dataset_build/tab_questions.jsonl')
tab_data = read_jsonl('/data1/zch/datasets/multimodalqa/MMQA_tables_md_cleaned.jsonl')
# desc_data = read_jsonl('/data1/zch/MultiModelDocRag/rerank/tab_dataset_build/tab_descriptions.jsonl')
# think_data = read_jsonl('/data1/zch/MultiModelDocRag/rerank/tab_dataset_build/tab_thinks.jsonl')
# key_points_data = read_jsonl('/data1/zch/MultiModelDocRag/rerank/tab_dataset_build/tab_key_points.jsonl')

# store_path = '/data1/zch/MultiModelDocRag/rerank/tab_dataset_build/tabQA_train.json'
store_path = '/data1/zch/MultiModelDocRag/rerank/tab_dataset_build/tabQA_dev.json'

tab_dict = dict()
for item in tab_data:
    tab_dict[
        item['id']] = f'**Table title:**{item["title"]}\n**Table name:**{item["table_name"]}**Table:**{item["table"]}'

result = []
for item in query_data:
    try:
        qid = item['qid']
        answer = item['answers'][0]['answer']
        type = item['metadata']['type']
        if type =='TableQ':
            question = item['question']
            tab_id = item['supporting_context'][0]['doc_id']
            result.append(
                get_tab_QA_messages(
                    question=question,
                    table_content=tab_dict[tab_id],
                    output=str(answer)
            ))
    except Exception as e:
        print(e)
        continue

store_json(store_path, result)


# desc_dict = dict()
# for item in desc_data:
#     desc_dict[
#         item['qid']] = item['golden_tab_desc']
#
# think_dict = dict()
# for item in think_data:
#     qid = item['qid']
#     think_dict[qid] = item['golden_tab_think']
#
# key_points_dict = dict()
# for item in key_points_data:
#     key_points_dict[item['qid']] = item['key_points_desc']

# result = []
# for item in query_data:
#     try:
#         qid = item['qid']
#         question = item['question']
#         tab_id = item['golden_tab']['metadata']['id']
#         result.append(
#             get_tab_QA_messages(
#                 question=question,
#                 table_content=tab_dict[tab_id],
#                 output=get_output(
#                     desc=desc_dict[qid],
#                     think=think_dict[qid],
#                     helpful=True,
#                     key_points=key_points_dict[qid]),
#             )
#         )
#     except Exception as e:
#         print(e)
#         continue
#
# store_json(store_path, result)

import os

"""
先输出图片描述，再输出图片的思考，再输出是否需要的特殊token，再输出重点描述信息
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *


def get_img_QA_train_prompt(
        question: str,
        img_title: str,
        img_name: str,
        output: str,
) -> dict:
    instruction = """You will process an input question, a related image, and the image's caption. Please output the following information in the specified format:

    1. **Image Description** (<description>):
       Provide a concise description of the image based on the input image and caption.

    2. **Relevance Analysis** (<relevance_analysis>):
       Analyze the relevance of the image to the given question and determine if the image is helpful in answering the question.

    3. **Decision** (<decision>):
       Based on the relationship between the image and the question, output either <helpful> (if the image is helpful) or <not_helpful> (if the image is not helpful).

    4. **Key Information** (<key_information>):
       If <decision> is <helpful>, extract the key information from the image and describe it concisely to ensure accurate question answering.
       If <decision> is <not_helpful>, output: This image cannot help answer the question.
    """
    QA_prompt_template = """**Image Content:**
{docs}

**Question:**
{question}
"""
    images = []
    docs = ""
    images.append('/mnt/sdb/zch/LLaMA-Factory/data/qwen2_vl_ft/mllm_data_v1/' + img_name)
    docs += f'**Image Title**:{img_title} **Image**:<image>'
    messages = [{
        "role": "system",
        "content": instruction,
    }, {
        "role": "user",
        "content": QA_prompt_template.format(question=question, docs=docs),
    }, {
        "role": "assistant",
        "content": output,
    }]
    return {
        "messages": messages,
        "images": images,
    }


def get_output(desc: str, think: str, helpful: bool, key_points: str):
    desc_str = f'<description>{desc}</description>'
    think_str = f'<relevance_analysis>{think}</relevance_analysis>'
    if helpful:
        helpful_str = f'<decision><helpful></decision>'
    else:
        helpful_str = f'<decision><not_helpful></decision>'
    key_points_str = f'<key_information>{key_points}</key_information>'
    return desc_str + think_str + helpful_str + key_points_str


query_data = read_jsonl('/data1/zch/MultiModelDocRag/rerank/img_dataset_build/img_questions.jsonl')
desc_dict = read_json('/data1/zch/MultiModelDocRag/rerank/img_dataset_build/img_descriptions.json')
think_data = read_jsonl('/data1/zch/MultiModelDocRag/rerank/img_dataset_build/img_thinks.jsonl')
key_points_data = read_jsonl('/data1/zch/MultiModelDocRag/rerank/img_dataset_build/img_key_points.jsonl')

store_path = '/data1/zch/MultiModelDocRag/rerank/img_dataset_build/imgQA_train.json'

think_dict = dict()
for item in think_data:
    qid = item['qid']
    q_think_dict = item['think_dict']
    for key, value in q_think_dict.items():
        think_dict[(qid, key)] = value

key_points_dict = dict()
for item in key_points_data:
    key_points_dict[item['qid']] = item['key_points_desc']

result = []
for item in query_data:
    qid = item['qid']
    question = item['question']
    golden_img = item['golden_img']
    golden_img_title = golden_img['metadata']['title']
    golden_img_id = golden_img['metadata']['id']
    golden_img_path = golden_img['metadata']['path']
    result.append(
        get_img_QA_train_prompt(
            question=question,
            img_title=golden_img_title,
            img_name=golden_img_path,
            output=get_output(
                desc=desc_dict[golden_img_path],
                think=think_dict[(qid, golden_img_id)],
                helpful=True,
                key_points=key_points_dict[qid]),
        )
    )
    hard_negative_img = item['hard_negative_img']
    for hn in hard_negative_img:
        hn_id = hn['metadata']['id']
        hn_title = hn['metadata']['title']
        hn_path = hn['metadata']['path']
        result.append(
            get_img_QA_train_prompt(
                question=question,
                img_title=hn_title,
                img_name=hn_path,
                output=get_output(
                    desc=desc_dict[hn_path],
                    think=think_dict[(qid, hn_id)],
                    helpful=False,
                    key_points="This image cannot help answer the question."),
            )
        )

store_json(store_path, result)

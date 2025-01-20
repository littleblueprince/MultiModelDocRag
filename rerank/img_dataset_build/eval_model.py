import os

"""
发送请求测试评估
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
import random

import openai


def get_img_QA_messages(
        question: str,
        img_title: str,
        img_name: str,
) -> list:
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
    image_type, b64_img = get_base64_img(image_name=img_name)
    messages = [
        {
            "role": "system",
            "content": instruction,
        }, {
            "role": "user",
            "content": [
                {"type": "text", "text": f"**Image Content:**\n**Image Title**:{img_title} **Image**:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{b64_img}",
                    },
                },
                {"type": "text", "text": f"\n**Question:**\n{question}"},
            ],
        },
    ]
    return messages


model = "qwen2-vl-instruct"
api_key = "api_key"
base_url = "http://36.213.0.171:9997/v1"
client = openai.Client(api_key=api_key, base_url=base_url)


query_data = read_jsonl('/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl')
img_data = read_jsonl('/data1/zch/datasets/multimodalqa/MMQA_images.jsonl')

img_dict = dict()
for img in img_data:
    img_dict[img['id']] = img

result = []
for item in query_data:
    qid = item['qid']
    question = item['question']
    type = item['metadata']['type']
    if type == 'ImageQ':
        image_doc_ids = item['metadata']['image_doc_ids']
        gt_id = item['supporting_context'][0]['doc_id']
        image_doc_ids = [idx for idx in image_doc_ids if idx != gt_id]
        negative_doc_id = random.sample(image_doc_ids, min(len(image_doc_ids), 1))[0]
        messages = get_img_QA_messages(
            question=question,
            img_title=img_dict[gt_id]['title'],
            img_name=img_dict[gt_id]['path'],
        )
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        print(completion.choices[0].message.content)
        input()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
import openai

# model = "qwen2-vl-72b-instruct"
# api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
# base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

model = "qwen2-vl-instruct"
api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
base_url = "http://36.213.0.171:9997/v1"

client = openai.Client(api_key=api_key, base_url=base_url)


def get_desc(img_path: str) -> str:
    image_type, b64_img = get_base64_img(image_name=img_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": "Generate a brief and accurate description of the given image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{b64_img}",
                    },
                },
            ],
        }
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return completion.choices[0].message.content


path = './img_questions.jsonl'
data = read_jsonl(path)
store_path = './img_descriptions.json'
desc_dict = dict()
for item in tqdm(data):
    qid = item['qid']
    question = item['question']
    answer = item['answer']
    metadata = item['metadata']
    golden_img = item['golden_img']
    hard_negative_img = item['hard_negative_img']
    golden_img_name = golden_img['metadata']['path']
    if golden_img_name not in desc_dict.keys():
        desc_dict[golden_img_name] = get_desc(golden_img_name)
    for i in hard_negative_img:
        hn_img_name = i['metadata']['path']
        if hn_img_name not in desc_dict.keys():
            desc_dict[hn_img_name] = get_desc(hn_img_name)

store_json(store_path, desc_dict)

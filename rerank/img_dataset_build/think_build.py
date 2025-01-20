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
api_key = 'test'
base_url = "http://36.213.0.171:9997/v1"

client = openai.Client(api_key=api_key, base_url=base_url)


def get_think(question: str, standard_answer: str, img_text: str, img_path: str) -> str:
    image_type, b64_img = get_base64_img(image_name=img_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Given the question, standard answer, and image, explain how the image helps answer the question. Focus on key elements in the image and how they relate to the question and standard answer.\n**Question:**{question}\n**Standard Answer:**{standard_answer}\n**Image Caption:**{img_text}\n**Image:**",
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
store_path = './img_thinks.jsonl'

result = []
for item in tqdm(data):
    try:
        think_dict = dict()
        qid = item['qid']
        question = item['question']
        answer = item['answer']
        metadata = item['metadata']
        golden_img = item['golden_img']
        hard_negative_img = item['hard_negative_img']
        golden_img_id = golden_img['metadata']['id']
        golden_img_name = golden_img['metadata']['path']
        golden_img_text = golden_img['content']
        if golden_img_name not in think_dict:
            think_dict[golden_img_id] = get_think(question,
                                                  f"This picture can answer the question. The correct answer is {answer}",
                                                  golden_img_text, golden_img_name)
        for i in hard_negative_img:
            hn_img_id = i['metadata']['id']
            hn_img_name = i['metadata']['path']
            hn_img_text = i['content']
            if hn_img_name not in think_dict:
                think_dict[hn_img_id] = get_think(question, "This picture doesn't answer the question", hn_img_text,
                                                  hn_img_name)
        result.append(
            {
                "qid": qid,
                "think_dict": think_dict,
            }
        )
    except Exception as e:
        print(e)
store_jsonl(store_path, result)

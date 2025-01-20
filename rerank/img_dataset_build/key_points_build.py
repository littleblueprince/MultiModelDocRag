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


def get_key_points_desc(question: str, img_text: str, img_path: str) -> str:
    image_type, b64_img = get_base64_img(image_name=img_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Given the question and image, describe the key parts of the image that support the answer. Focus on the most relevant details in the image that help answer the question.**Question:**{question}\n**Image Caption:**{img_text}\n**Image:**",
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
store_path = './img_key_points.jsonl'
f, processed = get_output_file(store_path, force=False)

result = []
for item in tqdm(data):
    try:
        qid = item['qid']
        if qid in processed:
            continue
        question = item['question']
        answer = item['answer']
        metadata = item['metadata']
        golden_img = item['golden_img']
        hard_negative_img = item['hard_negative_img']
        golden_img_id = golden_img['metadata']['id']
        golden_img_name = golden_img['metadata']['path']
        golden_img_text = golden_img['content']
        key_points_desc = get_key_points_desc(question, golden_img_text, golden_img_name)
        f.write(json.dumps({
            "qid": qid,
            "key_points_desc": key_points_desc,
        }, ensure_ascii=False) + '\n')
        f.flush()
    except Exception as e:
        continue
f.close()

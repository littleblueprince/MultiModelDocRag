import io
import openai
import base64
from PIL import Image

image_path = "/data1/zch/datasets/multimodalqa/final_dataset_images/7b9807a9e8461d29b9ae53de1d9e238f.jpg"
with open(image_path, "rb") as img_file:
    img_data = img_file.read()

b64_img = base64.b64encode(img_data).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text",
             "text": "What eyewear or accessory, if any, is Stevie Wonder wearing over his eyes in the image associated with his \"Part-Time Lover\" performance at the 28th Annual Grammy Awards?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    # "url": "http",
                },
            },
        ],
    }
]
# base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# model = "qwen2-vl-72b-instruct"
# api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'

api_key = 'sk-BLTI0o3z4R8GF8JNOHUuTmqtx6oLTt1SpeCXezoHwPPf4v5P'
base_url = "https://yunwu.ai/v1"
model = "gpt-4o-mini-2024-07-18"
#  model = "gpt-4o"

client = openai.Client(api_key=api_key, base_url=base_url)

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0,
)

print(completion.choices[0].message.content)

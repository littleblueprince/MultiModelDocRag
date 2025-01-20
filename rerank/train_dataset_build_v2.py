# -*- coding: utf-8 -*-
# @Time    : 2025/01/15 16:13
# @Author  : blue
# @Description :尝试合成图片理解数据
"""
先输出图片描述，再输出图片的思考，再输出是否需要的特殊token，再输出重点描述信息


"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

import io
import openai
import base64
from PIL import Image
from util.api import *

dir = '/data1/zch/datasets/multimodalqa/final_dataset_images/'
path = '3a54edd73657b7508a9e60575b002f2b.jpg'
image_path = dir + path

# display_image(image_path)


b64_img = get_base64_img(image_name=path, img_dir=dir)

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
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    # "url": "http",
                },
            },
        ],
    }
]

# model = "qwen2-vl-72b-instruct"
# api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
# base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

model = "qwen2-vl-instruct"
api_key = 'sk-3d3b8b4c32594e8fb4c69e0c9897819e'
base_url = "http://36.213.0.171:9997/v1"

client = openai.Client(api_key=api_key, base_url=base_url)

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0,
)

print(completion.choices[0].message.content)

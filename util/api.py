# -*- coding: utf-8 -*-
# @Time    : 2024/12/06 13:03
# @Author  : blue
# @Description : 封装一些工具函数
import os

os.environ["TMPDIR"] = "/data1/zch/tmp"
import base64
import json, re

import numpy as np


def read_jsonl(file_path: str) -> list[any]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def store_jsonl(file_path: str, data: list[any]) -> None:
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')


def read_json(file_path: str) -> list[any] | dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks


def store_json(file_path: str, data: list[any] | dict) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def pretty_print_json(data, key_color="green", value_color="reset"):
    # 定义颜色
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "reset": "\033[0m",
        "blue": "\033[94m",
    }

    # 获取键和值的颜色代码
    key_color_code = colors.get(key_color, colors["red"])
    value_color_code = colors.get(value_color, colors["reset"])

    # 将 JSON 数据格式化为字符串
    json_str = json.dumps(data, indent=4, ensure_ascii=False)

    # 正则表达式匹配键和值部分
    # 匹配模式：匹配键 "key": (保留引号，保证匹配到整个键)
    pattern = r'(".*?")(\s*:\s*)(.*?)(,?\n?)'

    # 使用正则替换，将键和值分别应用颜色
    def replace_match(match):
        key = f"{key_color_code}{match.group(1)}{colors['reset']}"
        separator = match.group(2)
        value = f"{value_color_code}{match.group(3)}{colors['reset']}"
        return f"{key}{separator}{value}{match.group(4)}"

    # 格式化后的 JSON 字符串
    formatted_str = re.sub(pattern, replace_match, json_str)

    print(formatted_str)


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def get_vec1_and_vec2_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) == 0 or len(vec2) == 0:
        raise ValueError("Input vectors must not be empty.")
    vec1_emb = np.array(vec1, dtype=float)
    vec2_emb = np.array(vec2, dtype=float)
    norm_vec1 = np.linalg.norm(vec1_emb)
    norm_vec2 = np.linalg.norm(vec2_emb)
    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ValueError("Input vectors must not be zero vectors.")
    vec1_emb = vec1_emb / norm_vec1
    vec2_emb = vec2_emb / norm_vec2
    similarity = np.dot(vec1_emb, vec2_emb)
    return float(similarity)


def pre_process_dataset(
        texts_path='/data1/zch/datasets/multimodalqa/MMQA_texts.jsonl',
        tabs_path='/data1/zch/datasets/multimodalqa/MMQA_tables_md.jsonl',
        imgs_path='/data1/zch/datasets/multimodalqa/MMQA_images.jsonl',
        img_dir='/data1/zch/datasets/multimodalqa/final_dataset_images/',
) -> dict:
    # 将各个模态的数据读取然后组织为一个dict,key是id
    texts_data = read_jsonl(texts_path)
    tabs_data = read_jsonl(tabs_path)
    imgs_data = read_jsonl(imgs_path)
    dataset_dict = dict()
    for item in texts_data:
        title = item['title']
        text = item['text']
        id = item['id']
        if id in dataset_dict:
            raise ValueError
        dataset_dict[id] = {
            "type": "text",
            "title": title,
            "content": text,
        }
    for item in tabs_data:
        title = item['title']
        table = item['table']
        id = item['id']
        if id in dataset_dict:
            raise ValueError
        dataset_dict[id] = {
            "type": "tab",
            "title": title,
            "content": table,
        }
    for item in imgs_data:
        title = item['title']
        path = item['path']
        id = item['id']
        with open(img_dir + path, "rb") as img_file:
            img_data = img_file.read()
        b64_img = base64.b64encode(img_data).decode("utf-8")
        if id in dataset_dict:
            raise ValueError
        dataset_dict[id] = {
            "type": "img",
            "title": title,
            "content": b64_img,
        }
    return dataset_dict

# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 20:31
# @Author  : blue
# @Description : 插入文本
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["HF_HOME"] = "/mnt/zch/tmp"
os.environ["TMPDIR"] = "/mnt/zch/tmp"

import json
from pymilvus import (
    MilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
)
from more_itertools import chunked
from tqdm import tqdm
import requests


def get_text_embedding(text: str, url: str = "http://36.213.0.171:8002/encode-text/"):
    data = {"text": text}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["embedding"][0]
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_img_embedding(text: str, url: str = "http://36.213.0.171:8002/encode-image/"):
    data = {"text": text}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["embedding"][0]
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def read_jsonl(file_path: str) -> list[any]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


class Milvus:
    def __init__(self, host, db_name, collection_name):
        self.host = host or "localhost"
        self.port = "19530"
        self.uri = f"http://{self.host}:{self.port}"
        self.db_name = db_name
        self.collection_name = collection_name

    def connect(self):
        self.client = MilvusClient(uri=self.uri, db_name=self.db_name, user='root', password='password')

    def exist_collection(self):
        return self.client.has_collection(self.collection_name)

    def create_schema(self):
        # 创建密集向量索引
        fields = []
        fields.append(
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True)
        )
        fields.append(FieldSchema("content", DataType.VARCHAR, max_length=65535))
        fields.append(FieldSchema("vector", DataType.FLOAT_VECTOR, dim=768))
        fields.append(FieldSchema("metadata", DataType.JSON))
        self.schema = CollectionSchema(fields)

    def create_collection(self, collection_name=None, schema=None):
        if not schema:
            self.create_schema()
        self.client.create_collection(
            collection_name=collection_name or self.collection_name,
            schema=schema or self.schema,
        )
        index_param = self.client.prepare_index_params()
        index_param.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="HNSW",
            index_name="source_index",
            params={"M": 64, "efConstruction": 1000},
        )
        self.client.create_index(self.collection_name, index_param)

    def drop_collection(self, collection_name=None):
        self.client.drop_collection(
            collection_name=collection_name or self.collection_name
        )

    def insert(self, datas: dict | list[dict], collection_name=None):
        res = self.client.insert(
            db_name=self.db_name,
            collection_name=collection_name or self.collection_name,
            data=datas,
        )
        return res


if __name__ == "__main__":

    inserted_ids_file = "inserted_ids.txt"
    # 加载已插入的ID
    if os.path.exists(inserted_ids_file):
        with open(inserted_ids_file, 'r') as f:
            inserted_ids = set(line.strip() for line in f)
    else:
        inserted_ids = set()

    db_name = "mmqa"
    collection_name = "v1"
    milvus = Milvus(host="10.176.40.88", db_name=db_name, collection_name=collection_name)
    milvus.connect()
    if not milvus.exist_collection():
        milvus.create_collection()
    texts_path = '/data1/zch/datasets/multimodalqa/MMQA_texts.jsonl'
    texts_dataset = read_jsonl(texts_path)
    batched_data = chunked(texts_dataset, 512)
    for batch in tqdm(batched_data):
        insert_datas = []
        batch = [item for item in batch if item['id'] not in inserted_ids]
        if not batch:
            continue
        for item in batch:
            title = item['title']
            url = item['url']
            text = item['text']
            id = item['id']
            vector = get_text_embedding(f'**title**:{title}**text**:{text}')
            insert_datas.append(
                {
                    "content": f'**title**:{title}**text**:{text}',
                    "vector": vector,
                    "metadata": item
                }
            )
        milvus.insert(insert_datas, collection_name=collection_name)
        # 记录已插入的ID
        with open(inserted_ids_file, 'a') as f:
            for item in batch:
                f.write(f"{item['id']}\n")
    milvus.client.load_collection(collection_name=collection_name)
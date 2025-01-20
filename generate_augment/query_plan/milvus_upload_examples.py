# -*- coding: utf-8 -*-
# @Time    : 2024/12/27 15:36
# @Author  : blue
# @Description :
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"
import json
from pymilvus import (
    MilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
)

import aiohttp
import asyncio
import os


async def get_embedding_async(texts: list[str], base_url: str = "http://10.176.40.88:9997/v1/embeddings",
                              embedding_model='bge_m3_cuda0', api_key='test') -> list:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": embedding_model,
        "input": texts,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=data) as response:
            if response.status == 200:
                embedding_completions = await response.json()
                return [item['embedding'] for item in embedding_completions['data']]
            else:
                error_text = await response.text()
                print(f"Request failed with status code {response.status}")
                print(error_text)
                return []


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
        fields.append(FieldSchema("vector", DataType.FLOAT_VECTOR, dim=1024))
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


async def process_and_insert_batch(batch, milvus, collection_name):
    questions = [b['question'] for b in batch]
    # 处理每个批次
    results = await get_embedding_async(questions)

    # 插入到 Milvus
    inserted_results = []
    for i, vec in enumerate(results):
        inserted_results.append(
            {
                "content": batch[i]["question"],
                "vector": vec,
                "metadata": batch[i],
            }
        )

    milvus.insert(inserted_results, collection_name=collection_name)
    print(f"Inserted {len(inserted_results)} items into Milvus.")
    return len(inserted_results)


async def process_all_items(dataset, milvus, collection_name, batch_size=32):
    total_inserted = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        total_inserted += await process_and_insert_batch(batch, milvus, collection_name)
    return total_inserted


async def main():
    db_name = "mmqa"
    collection_name = "dag_v2"

    # 连接到 Milvus
    milvus = Milvus(host="10.176.40.88", db_name=db_name, collection_name=collection_name)
    milvus.connect()

    # 如果没有该 collection，就创建它
    if not milvus.exist_collection():
        milvus.create_collection()

    dataset = read_jsonl('/data1/zch/MultiModelDocRag/generate_augment/query_plan/train_data_v1.jsonl')

    print(f'{len(dataset)} images need to be inserted')

    total_inserted = await process_all_items(dataset, milvus, collection_name, batch_size=1024)
    print(f"Total {total_inserted} items inserted.")
    milvus.client.load_collection(collection_name=collection_name)


if __name__ == "__main__":
    asyncio.run(main())

# from util.api import *
#
# path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/train_data_v2.jsonl'
#
# store_path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/train_data_v3.jsonl'
# res = []
# for item in read_jsonl(path):
#     try:
#         item['dag'] = item['dag'].replace('`', '').replace('json', '')
#         item['dag'] = json.loads(item['dag'])
#         if item['supporting_context'][0]['doc_part'] == "image":
#             item['dag']['action'] = 'image_retrieval'
#         else:
#             item['dag']['action'] = 'text_retrieval'
#         item['dag'] = json.dumps(item['dag'])
#         res.append(item)
#     except Exception as e:
#         print(e)
#         continue
#
# store_jsonl(store_path, res)

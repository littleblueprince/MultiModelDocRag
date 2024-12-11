# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 20:31
# @Author  : blue
# @Description : 插入table
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

import aiohttp
import asyncio
import os


async def get_embedding_async(text=None, file_path=None, url: str = "http://36.213.0.171:8001/encode/"):
    data = {
        "text": text,
        "file_path": file_path,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    json_response = await response.json()
                    return json_response["embedding"][0]
                else:
                    return {"error": f"Error {response.status}: {await response.text()}"}
    except aiohttp.ClientError as e:
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


async def async_process_item(item):
    id = item['id']
    title = item['title']
    table_name = item['table_name']
    table = item['table']
    text = f'**{title}:** ## {table_name}: {table}'
    vector = await get_embedding_async(text=text)
    return {
        "content": text,
        "vector": vector,
        "metadata": item,
    }


# 将 Milvus 插入和写入 inserted_ids 操作放入到每个批次中
async def process_and_insert_batch(batch, milvus, collection_name, inserted_ids_file):
    # 处理每个批次
    results = await asyncio.gather(*[async_process_item(item) for item in batch])

    # 插入到 Milvus
    milvus.insert(results, collection_name=collection_name)
    print(f"Inserted {len(results)} items into Milvus.")

    # 更新 inserted_ids
    with open(inserted_ids_file, 'a') as f:
        for item in batch:
            f.write(f"{item['id']}\n")

    return len(results)


async def process_all_items(imgs_dataset, milvus, collection_name, inserted_ids_file, batch_size=32):
    total_inserted = 0
    for i in range(0, len(imgs_dataset), batch_size):
        batch = imgs_dataset[i:i + batch_size]
        total_inserted += await process_and_insert_batch(batch, milvus, collection_name, inserted_ids_file)
    return total_inserted


async def main():
    inserted_ids_file = "inserted_tabs_ids_v2.txt"
    # 加载已插入的ID
    if os.path.exists(inserted_ids_file):
        with open(inserted_ids_file, 'r') as f:
            inserted_ids = set(line.strip() for line in f)
    else:
        inserted_ids = set()

    db_name = "mmqa"
    collection_name = "v3"
    milvus = Milvus(host="10.176.40.88", db_name=db_name, collection_name=collection_name)
    milvus.connect()
    if not milvus.exist_collection():
        milvus.create_collection()
    tables_path = '/data1/zch/datasets/multimodalqa/MMQA_tables_md_cleaned.jsonl'
    tables_dataset = read_jsonl(tables_path)
    tables_dataset_filtered = [item for item in tables_dataset if item['id'] not in inserted_ids]
    print(f'{len(tables_dataset_filtered)} images need to be inserted')

    # 处理所有图片，并将它们插入到 Milvus
    total_inserted = await process_all_items(tables_dataset_filtered, milvus, collection_name, inserted_ids_file,
                                             batch_size=32)
    print(f"Total {total_inserted} items inserted.")
    milvus.client.load_collection(collection_name=collection_name)


if __name__ == "__main__":
    asyncio.run(main())

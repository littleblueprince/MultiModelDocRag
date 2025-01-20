# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 14:10
# @Author  : blue
# @Description :
import numpy as np
from pymilvus import connections, Collection


def cosine_similarity(query_emb: list[float], vectors: list[list[float]]) -> list[float]:
    query_emb = np.array(query_emb)
    vectors = np.array(vectors)
    query_emb = query_emb / np.linalg.norm(query_emb)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = np.dot(vectors, query_emb.T)
    return similarities.tolist()


def retrieval_corpus(
        query_vector: list,
        text_filtered_ids: list[str],
        tabs_filtered_ids: list[str],
        imgs_filtered_ids: list[str],
        text_collection_name: str,
        tabs_collection_name: str,
        imgs_collection_name: str,
        db_name="mmqa",
        top_k=50,
        host="10.176.40.88",
        port="19530",
        user='root',
        password='password',
) -> list:
    # 根据query_vec去milvus数据库搜索corpus返回
    connections.connect(
        host=host,
        port=port,
        db_name=db_name,
        user=user,
        password=password,
    )
    text_collection = Collection(name=text_collection_name)
    tabs_collection = Collection(name=tabs_collection_name)
    imgs_collection = Collection(name=imgs_collection_name)
    text_collection.load()
    tabs_collection.load()
    imgs_collection.load()
    text_filter_ids_str = ",".join([f'"{item}"' for item in text_filtered_ids])
    tabs_filter_ids_str = ",".join([f'"{item}"' for item in tabs_filtered_ids])
    imgs_filter_ids_str = ",".join([f'"{item}"' for item in imgs_filtered_ids])
    if text_filtered_ids:
        text_results = text_collection.query(
            expr=f'metadata["id"] in [{text_filter_ids_str}]',
            output_fields=["content", "metadata", "vector"],
            limit=top_k,
        )
    else:
        text_results = []
    if tabs_filtered_ids:
        tabs_results = tabs_collection.query(
            expr=f'metadata["id"] in [{tabs_filter_ids_str}]',
            output_fields=["content", "metadata", "vector"],
            limit=top_k,
        )
    else:
        tabs_results = []
    if imgs_filtered_ids:
        imgs_results = imgs_collection.query(
            expr=f'metadata["id"] in [{imgs_filter_ids_str}]',
            output_fields=["content", "metadata", "vector"],
            limit=top_k,
        )
    else:
        imgs_results = []
    results = text_results + tabs_results + imgs_results
    vectors = []
    for hit in results:
        vectors.append(hit['vector'])
    similarities = cosine_similarity(query_vector, vectors)
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    sorted_similarities = [similarities[i] for i in sorted_indices]
    sorted_results = [results[i] for i in sorted_indices]
    milvus_corpus = []
    for sorted_result, score in zip(sorted_results, sorted_similarities):
        milvus_corpus.append(
            {
                "metadata": sorted_result['metadata'],
                "content": sorted_result['content'],
                "score": score,
            }
        )
    return milvus_corpus


def retrieval_dag_examples_with_expr(
        query_vector: list,
        expr: str,
        collection_name: str,
        db_name="mmqa",
        top_k=3,
        host="10.176.40.88",
        port="19530",
        user='root',
        password='password',
) -> list:
    # 根据query_vec去milvus数据库搜索corpus返回
    connections.connect(
        host=host,
        port=port,
        db_name=db_name,
        user=user,
        password=password,
    )
    collection = Collection(name=collection_name)
    collection.load()
    results = collection.search(
        data=[query_vector],
        expr=expr,
        anns_field='vector',
        param={
            "M": 64,
            "efConstruction": 1000
        },
        output_fields=["vector", "metadata"],
        limit=top_k,
    )
    return [
        {
            "question": results[0][0].fields['metadata']['question'],
            "dag": results[0][0].fields['metadata']['dag'],
        },
        {
            "question": results[0][1].fields['metadata']['question'],
            "dag": results[0][1].fields['metadata']['dag'],
        },
        {
            "question": results[0][2].fields['metadata']['question'],
            "dag": results[0][2].fields['metadata']['dag'],
        },
    ]


def retrieval_dag_examples(
        query_vector: list,
        collection_name: str,
        db_name="mmqa",
        top_k=3,
        host="10.176.40.88",
        port="19530",
        user='root',
        password='password',
) -> list:
    # 根据query_vec去milvus数据库搜索corpus返回
    connections.connect(
        host=host,
        port=port,
        db_name=db_name,
        user=user,
        password=password,
    )
    collection = Collection(name=collection_name)
    collection.load()
    results = collection.search(
        data=[query_vector],
        anns_field='vector',
        param={
            "M": 64,
            "efConstruction": 1000
        },
        output_fields=["vector", "metadata"],
        limit=top_k,
    )
    return [
        {
            "question": results[0][0].fields['metadata']['question'],
            "dag": results[0][0].fields['metadata']['dag'],
        },
        {
            "question": results[0][1].fields['metadata']['question'],
            "dag": results[0][1].fields['metadata']['dag'],
        },
        {
            "question": results[0][2].fields['metadata']['question'],
            "dag": results[0][2].fields['metadata']['dag'],
        },
    ]


def text_retrieval_corpus(
        query_vector: list,
        text_filtered_ids: list[str],
        tabs_filtered_ids: list[str],
        text_collection_name: str,
        tabs_collection_name: str,
        db_name="mmqa",
        top_k=50,
        host="10.176.40.88",
        port="19530",
        user='root',
        password='password',
) -> list:
    # 根据query_vec去milvus数据库搜索corpus返回
    connections.connect(
        host=host,
        port=port,
        db_name=db_name,
        user=user,
        password=password,
    )
    text_collection = Collection(name=text_collection_name)
    tabs_collection = Collection(name=tabs_collection_name)
    text_collection.load()
    tabs_collection.load()
    text_filter_ids_str = ",".join([f'"{item}"' for item in text_filtered_ids])
    tabs_filter_ids_str = ",".join([f'"{item}"' for item in tabs_filtered_ids])
    if text_filtered_ids:
        text_results = text_collection.query(
            expr=f'metadata["id"] in [{text_filter_ids_str}]',
            output_fields=["content", "metadata", "vector"],
            limit=top_k,
        )
    else:
        text_results = []
    if tabs_filtered_ids:
        tabs_results = tabs_collection.query(
            expr=f'metadata["id"] in [{tabs_filter_ids_str}]',
            output_fields=["content", "metadata", "vector"],
            limit=top_k,
        )
    else:
        tabs_results = []
    results = text_results + tabs_results
    vectors = []
    for hit in results:
        vectors.append(hit['vector'])
    similarities = cosine_similarity(query_vector, vectors)
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    sorted_similarities = [similarities[i] for i in sorted_indices]
    sorted_results = [results[i] for i in sorted_indices]
    milvus_corpus = []
    for sorted_result, score in zip(sorted_results, sorted_similarities):
        milvus_corpus.append(
            {
                "metadata": sorted_result['metadata'],
                "content": sorted_result['content'],
                "score": score,
            }
        )
    return milvus_corpus


def table_retrieval_corpus(
        query_vector: list,
        tabs_filtered_ids: list[str],
        tabs_collection_name: str,
        db_name="mmqa",
        top_k=50,
        host="10.176.40.88",
        port="19530",
        user='root',
        password='password',
) -> list:
    # 根据query_vec去milvus数据库搜索corpus返回
    connections.connect(
        host=host,
        port=port,
        db_name=db_name,
        user=user,
        password=password,
    )

    tabs_collection = Collection(name=tabs_collection_name)
    tabs_collection.load()
    tabs_filter_ids_str = ",".join([f'"{item}"' for item in tabs_filtered_ids])
    if tabs_filtered_ids:
        tabs_results = tabs_collection.query(
            expr=f'metadata["id"] in [{tabs_filter_ids_str}]',
            output_fields=["content", "metadata", "vector"],
            limit=top_k,
        )
    else:
        tabs_results = []
    results = tabs_results
    vectors = []
    for hit in results:
        vectors.append(hit['vector'])
    similarities = cosine_similarity(query_vector, vectors)
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    sorted_similarities = [similarities[i] for i in sorted_indices]
    sorted_results = [results[i] for i in sorted_indices]
    milvus_corpus = []
    for sorted_result, score in zip(sorted_results, sorted_similarities):
        milvus_corpus.append(
            {
                "metadata": sorted_result['metadata'],
                "content": sorted_result['content'],
                "score": score,
            }
        )
    return milvus_corpus


def image_retrieval_corpus(
        query_vector: list,
        imgs_filtered_ids: list[str],
        imgs_collection_name: str,
        db_name="mmqa",
        top_k=50,
        host="10.176.40.88",
        port="19530",
        user='root',
        password='password',
) -> list:
    # 根据query_vec去milvus数据库搜索corpus返回
    connections.connect(
        host=host,
        port=port,
        db_name=db_name,
        user=user,
        password=password,
    )
    imgs_collection = Collection(name=imgs_collection_name)
    imgs_collection.load()
    imgs_filter_ids_str = ",".join([f'"{item}"' for item in imgs_filtered_ids])
    if imgs_filtered_ids:
        imgs_results = imgs_collection.query(
            expr=f'metadata["id"] in [{imgs_filter_ids_str}]',
            output_fields=["content", "metadata", "vector"],
            limit=top_k,
        )
    else:
        imgs_results = []
    results = imgs_results
    vectors = []
    for hit in results:
        vectors.append(hit['vector'])
    similarities = cosine_similarity(query_vector, vectors)
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    sorted_similarities = [similarities[i] for i in sorted_indices]
    sorted_results = [results[i] for i in sorted_indices]
    milvus_corpus = []
    for sorted_result, score in zip(sorted_results, sorted_similarities):
        milvus_corpus.append(
            {
                "metadata": sorted_result['metadata'],
                "content": sorted_result['content'].split('img_path')[0] + "img_content: ",
                "score": score,
            }
        )
    return milvus_corpus

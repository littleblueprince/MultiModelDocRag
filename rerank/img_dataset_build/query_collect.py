"""
先输出图片描述，再输出图片的思考，再输出是否需要的特殊token，再输出重点描述信息


"""
import os

import requests

from util.retrieval_api import image_retrieval_corpus

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *


def get_embedding_sync(text: str, url: str = "http://36.213.0.171:8001/encode/"):
    data = {
        "text": text,
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            json_response = response.json()
            return {
                "embedding": json_response["embedding"][0]
            }
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        return {"error": str(e)}


path = '/data1/zch/datasets/multimodalqa/MMQA_train.jsonl'
data = read_jsonl(path)

store_path = './img_questions.jsonl'

res = []
for item in tqdm(data):
    qid = item['qid']
    question = item['question']
    answer = item['answers'][0]['answer']
    metadata = item['metadata']
    supporting_context = item['supporting_context']
    type = item['metadata']['type']
    if type == 'ImageQ':
        gt_id = item['supporting_context'][0]['doc_id']
        image_doc_ids = metadata['image_doc_ids']
        query_vector = get_embedding_sync(text=question)['embedding']
        corpus = image_retrieval_corpus(
            query_vector=query_vector,
            imgs_filtered_ids=image_doc_ids,
            imgs_collection_name='img_v2',
        )
        gt = None
        for img_data in corpus:
            img_metadata = img_data['metadata']
            img_idx = img_metadata['id']
            if img_idx == gt_id:
                gt = img_data
        corpus = [img_data for img_data in corpus if img_data['metadata']['id'] != gt_id]
        hn = corpus[:2]
        res.append(
            {
                'qid': qid,
                'question': question,
                'answer': answer,
                'golden_img': gt,
                'hard_negative_img': hn,
                'metadata': metadata,
            }
        )

store_jsonl(store_path, res)

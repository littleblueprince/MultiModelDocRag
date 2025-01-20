# -*- coding: utf-8 -*-
# @Time    : 2024/12/24 23:37
# @Author  : blue
# @Description : 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"
from util.api import *
from openai import OpenAI
import instructor
from pydantic import BaseModel

instruction = """
**Task:** Carefully evaluate whether a given retrieval plan is consistent with the original question and its correct answer. Verify if the steps logically align with deriving the correct answer from the question. Ensure the output is concise, formatted as a "Reasonable" boolean value and the node count.

**Input:**
1. **Original Question**: {question}
2. **Correct Answer**: {answer}
3. **Retrieval Plan**: {dag}

**Output:**
- Format the evaluation result as `Reasonable:True` if the retrieval plan is consistent with the original question and logically leads to the correct answer. Otherwise, format as `Reasonable:False`.
- Include the total number of nodes (steps) in the retrieval plan in the format `Node Count: [Number]`.

**Example Output:**
```
Reasonable:True
Node Count: 3
```
"""


class FilterPlan(BaseModel):
    reasonable: bool
    node_count: int


path = '/data1/zch/MultiModelDocRag/generate_augment/dag/dag_query_qwen2.5_72b_train_v1.jsonl'
data = read_jsonl(path)

store_path = './filter_train_dag_v1.jsonl'
f, processed_results = get_output_file(store_path, force=False)

api_key = 'test'
base_url = "http://36.137.79.97:30250/v1"
model = "qwen2-72b-instruct"
client = instructor.from_openai(OpenAI(
    api_key=api_key,
    base_url=base_url,
))

for item in tqdm(data):
    qid = item['qid']
    question = item['question']
    answer = item['answers'][0]['answer']
    supporting_context = item['supporting_context']
    dag = item['dag']
    if qid in processed_results:
        continue
    prompt = instruction.format(question=question, answer=answer, dag=dag)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_model=FilterPlan,
    )
    f.write(
        json.dumps({
            "qid": qid,
            "question": question,
            "answer": answer,
            "supporting_context": supporting_context,
            "dag": dag,
            "reasonable": response.reasonable,
            "node_count": response.node_count,
        }) + '\n'
    )
    f.flush()

f.close()

# -*- coding: utf-8 -*-
# @Time    : 2024/12/12 18:30
# @Author  : blue
# @Description :
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"
from openai import OpenAI
from util.api import *

# data=read_jsonl('/data1/zch/MultiModelDocRag/util/eval/eval_result_v3.jsonl')
# res= []
# for item in data:
#     if item['eval_res']=='0':
#         res.append(item['qid'])
#
# print(res)
# input()


# path1='/data1/zch/MultiModelDocRag/util/eval/eval_result_v3.jsonl'
# path2='/data1/zch/MultiModelDocRag/util/eval/eval_result_v4.jsonl'
# data1=read_jsonl(path1)
# data2=read_jsonl(path2)
# for item1,item2 in zip(data1,data2):
#     if item1['eval_res']!=item2['eval_res']:
#         pretty_print_json(item1)
#         pretty_print_json(item2)
#         input()

base_url = "http://36.137.79.97:30250/v1/"
model = "qwen2-72b-instruct"
api_key = "test"
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
result = []

topk = -1
topk = 51
# version = 3
version = 4
# version = 6
# version = 7
# version = 8

path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/prediction_qwen2_vl_72b_dev_v' + str(version) + '.jsonl'
data = read_jsonl(path)[:topk]
store_path = './eval_result_v' + str(version) + '.jsonl'
f, processed_results = get_output_file(store_path, force=False)

prompt_template = """以下是问答评估规则：

    规则：
    1. 如果预测答案和标准答案完全一致或等价，则评分为 1。
    2. 如果预测答案部分正确，但有缺失或包含不必要的错误信息，则评分为 0.5。
    3. 如果预测答案完全错误或与问题无关，则评分为 0。

    请根据以下输入，判断并输出评分（仅输出 1、0.5 或 0）：
    - Question: {question}
    - Ground Truth: {ground_truth}
    - Prediction: {prediction}
    """

for item in tqdm(data):
    qid = item['qid']
    if qid in processed_results or not item['graph_logs']:
        continue
    question = item['original_question']
    # question = item['question']
    ground_truth = item['ground_truth']
    # ground_truth = item['answer']
    # prediction = item['prediction']
    prediction = item['graph_logs'][-1]['answer']
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt_template.format(
                    question=question,
                    ground_truth=ground_truth,
                    prediction=prediction,
                )
            }
        ],
        model=model,
        max_tokens=150,  # 设置最大生成的 token 数量
        temperature=0.7,  # 控制生成内容的随机性
        top_p=0.9,  # 控制生成内容的多样性
        n=1,  # 返回生成的候选响应数量
        stop=["\n"],  # 生成内容的终止标识
        presence_penalty=0.6,  # 控制内容多样性，减少内容重复性
        frequency_penalty=0.5,  # 减少重复词汇生成的频率
        stream=False,  # 启用流式响应
    )
    f.write(json.dumps({
        "qid": qid,
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "eval_res": chat_completion.choices[0].message.content,
    }) + "\n")
    f.flush()
f.close()

eval_data = read_jsonl(store_path)[:topk]
count = 0

for item in eval_data:
    if item['eval_res'] == '1':
        count += 1
    elif item['eval_res'] == '0.5':
        count += 0.5
    elif item['eval_res'] == '0':
        count += 0
    else:
        print(item)

print(f'count: {count} \ntotal: {len(eval_data)}')

print(f'hit rate: {count / len(eval_data)}')

import json

from util.api import *

from openai import OpenAI

# path = '/data1/zch/datasets/multimodalqa/MMQA_train.jsonl'
path = '/data1/zch/MultiModelDocRag/generate_augment/query_plan/plan_dev_v1.jsonl'
store_path = './dev_v1.jsonl'

instruction = 'Please plan a problem retrieval and reasoning solution based on the following questions, known content, and query failure operation records.'
input_prompt = '**Question:**{question}\n**Known Content:**{known_content}\n**Failure Operation Records:**{failure_operation_records}'
data = read_jsonl(path)
print(len(data))
result = []
for item in data:
    instruction = item['instruction']
    question = item['input'].replace('**Question:**', '')
    try:
        graph = json.loads(item['output'])
        output = ''
        for node in graph:
            if not node['dependencies']:
                output += '<NO_DEPENDENCY>' + json.dumps(node) + '</NO_DEPENDENCY>'
            else:
                output += '<HAS_DEPENDENCY>' + json.dumps(node) + '</HAS_DEPENDENCY>'
        result.append(
            {
                'instruction': instruction,
                'input': input_prompt.format(question=question, known_content="", failure_operation_records=""),
                'output': output,
            }
        )
    except:
        pretty_print_json(item)

store_jsonl(store_path, result)

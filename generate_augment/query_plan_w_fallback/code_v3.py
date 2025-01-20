from util.api import *

from openai import OpenAI

base_url = "http://36.213.0.171:9997/v1/"
model = "qwen2.5-instruct"
api_key = 'test'
instruction = 'Please plan a problem retrieval and reasoning solution based on the following questions, known content, and query failure operation records.'
input_prompt = '**Question:**{question}\n**Known Content:**{known_content}\n**Failure Operation Records:**{failure_operation_records}'

path = '/data1/zch/datasets/multimodalqa/MMQA_dev.jsonl'
data = read_jsonl(path)
for item in tqdm(data):
    question = item['question']
    known_content = ""
    failure_operation_records = ""
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": instruction + input_prompt.format(question=question, known_content=known_content,
                                                             failure_operation_records=failure_operation_records),
            }
        ],
        model=model,
        temperature=0.7,
        top_p=0.8,
        stream=True,
    )

    # 逐步读取并解析流式响应
    output_str = ""
    nodes = []
    for chunk in chat_completion:
        delta = chunk.choices[0].delta
        content = delta.content
        if content:
            output_str += content
            # print(content, end="")  # end="" 确保内容逐步打印在同一行
            if '</NO_DEPENDENCY>' in output_str:
                temp_output_str = output_str.split('<NO_DEPENDENCY>')[1].split('</NO_DEPENDENCY>')[0].replace('\\"',
                                                                                                              '"')
                nodes.append(json.loads(temp_output_str))
                output_str = output_str.split('</NO_DEPENDENCY>')[1]
            elif '<HAS_DEPENDENCY>' in output_str:
                print('仍然存在被依赖问题需要解决')
                break
    pretty_print_json(nodes)
    input()

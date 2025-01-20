import random
import time

from util.api import *


def stream_string_random_chunk(text, min_chunk_size=3, max_chunk_size=10, delay=0.01):
    """
    模拟按随机大小流式输出字符串，每次输出一个随机大小的块并暂停一定的时间。
    :param text: 需要流式输出的字符串
    :param min_chunk_size: 最小块大小
    :param max_chunk_size: 最大块大小
    :param delay: 每次输出之间的延迟（秒）
    """
    i = 0
    while i < len(text):
        chunk_size = random.randint(min_chunk_size, max_chunk_size)  # 随机生成块大小
        chunk = text[i:i + chunk_size]
        yield chunk
        i += chunk_size  # 更新下一个块的起始位置
        time.sleep(delay)  # 模拟延迟输出


input_str = '<NO_DEPENDENCY>{\\\"name\\\": \\\"Q1\\\", \\\"query\\\": \\\"Identify the team that has won the Europa League the most times.\\\", \\\"action\\\": \\\"text_retrieval\\\", \\\"dependencies\\\": []}</NO_DEPENDENCY><NO_DEPENDENCY>{\\\"name\\\": \\\"Q2\\\", \\\"query\\\": \\\"List the winners of the La Liga play-offs between 1987 and 1999.\\\", \\\"action\\\": \\\"text_retrieval\\\", \\\"dependencies\\\": []}</NO_DEPENDENCY><HAS_DEPENDENCY>{\\\"name\\\": \\\"Q3\\\", \\\"query\\\": \\\"Determine the most recent season among the winners in the 1987-1999 La Liga play-offs for both Racing Santander and the team identified in Q1.\\\", \\\"action\\\": \\\"general_retrieval\\\", \\\"dependencies\\\": [\\\"Q1\\\", \\\"Q2\\\"]}</HAS_DEPENDENCY>'

# 逐步读取并解析流式响应
output_str = ""
nodes = []
for chunk in stream_string_random_chunk(input_str):
    # delta = chunk.choices[0].delta
    # content = delta.content
    content = chunk
    if content:
        output_str += content
        # print(content, end="")  # end="" 确保内容逐步打印在同一行
        if '</NO_DEPENDENCY>' in output_str:
            temp_output_str = output_str.split('<NO_DEPENDENCY>')[1].split('</NO_DEPENDENCY>')[0].replace('\\"', '"')
            nodes.append(json.loads(temp_output_str))
            output_str = output_str.split('</NO_DEPENDENCY>')[1]
        elif '<HAS_DEPENDENCY>' in output_str:
            print('仍然存在被依赖问题需要解决')
            break
pretty_print_json(nodes)

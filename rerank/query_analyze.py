# -*- coding: utf-8 -*-
# @Time    : 2025/01/15 16:13
# @Author  : blue
# @Description : 分析问题种类

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *

path = '/data1/zch/datasets/multimodalqa/MMQA_train.jsonl'
data = read_jsonl(path)

count_dict = dict()
for item in data:
    type = item['metadata']['type']
    if type not in count_dict:
        count_dict[type] = 1
    else:
        count_dict[type] += 1

for key in count_dict:
    count_dict[key] = count_dict[key] / len(data)

pretty_print_json(count_dict)
# -*- coding: utf-8 -*-
# @Time    : 2024/12/11 14:56
# @Author  : blue
# @Description :
import os

os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

from util.api import *
import re


def remove_urls(text):
    # 正则表达式匹配以 http:// 或 https:// 开头的 URL
    cleaned_text = re.sub(r'http[s]?://\S+', '', text)
    return cleaned_text


def table_to_md_without_links(data) -> tuple:
    # 获取表头
    headers = [column["column_name"] for column in data["table"]["header"]]
    rows = data["table"]["table_rows"]
    table_name = data["table"]["table_name"]

    # 生成表头行和分隔符
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "|---" * len(headers) + "|\n"

    # 生成数据行
    for row in rows:
        md_row = []
        for cell in row:
            # 提取纯文本部分，去除链接
            text = cell["text"]
            md_row.append(text)
        md_table += "| " + " | ".join(md_row) + " |\n"

    return table_name, md_table


path = '/data1/zch/datasets/multimodalqa/MMQA_tables.jsonl'
data = read_jsonl(path)

res = []
for item in data:
    id = item['id']
    title = item['title']
    url = item['url']
    table_name, table_md = table_to_md_without_links(item)
    table_md_cleaned = remove_urls(table_md)
    res.append({
        "id": id,
        "title": title,
        "table_name": table_name,
        "url": url,
        "table": table_md_cleaned,
    })

store_path = '/data1/zch/datasets/multimodalqa/MMQA_tables_md_cleaned.jsonl'
store_jsonl(store_path, res)

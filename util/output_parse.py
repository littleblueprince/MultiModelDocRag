# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 15:43
# @Author  : blue
# @Description :
import os

os.environ["TMPDIR"] = "/data1/zch/tmp"

import re


def parse_llm_output_node1_v1(text):
    segments = text.split("**Answerability**")

    if len(segments) > 1:
        answerability_section = "**Answerability**" + segments[1]
    else:
        answerability_section = text  # 若未找到关键字，返回原文本

    def extract_fields(section):
        pattern_yes = re.compile(
            r"- Can the query be directly answered: Yes\s*"
            r"2\.\s*\*\*Response\*\*:\s*- Direct Answer: (.+)",
            re.DOTALL
        )
        pattern_no = re.compile(
            r"- Can the query be directly answered: No\s*"
            r"2\.\s*\*\*Response\*\*:\s*- Next Sub-Query: (.+?)\s*- Hypothetical Answer: (.+)",
            re.DOTALL
        )
        match_yes = pattern_yes.search(section)
        if match_yes:
            return {
                "Answerability": "Yes",
                "Direct Answer": match_yes.group(1).strip()
            }
        match_no = pattern_no.search(section)
        if match_no:
            return {
                "Answerability": "No",
                "Next Sub-Query": match_no.group(1).strip(),
                "Hypothetical Answer": match_no.group(2).strip()
            }
        return {"Raw Section": section}

    return extract_fields(answerability_section)

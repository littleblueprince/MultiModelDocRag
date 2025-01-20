# -*- coding: utf-8 -*-
# @Time    : 2024/12/17 11:16
# @Author  : blue
# @Description : 
import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"


file_name='/data1/zch/githubProjects/qwen_vl_code/eos_data_t2t.npz'
# 加载数据
loaded_data = np.load(file_name)

# 提取数据
eos_embeddings = loaded_data['eos_embeddings']
labels = loaded_data['labels']
print(len(labels))

# import torch
# from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
#
# # 模型和分词器路径
# model_dir = "/data1/zch/models/Qwen2-VL-2B-Instruct"
#
# # 加载分词器
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
#
# # 强制添加 <EOS> token
# eos_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
# print(f'eos_token:{eos_token}')
# texts = [f"This is a test sentence for hidden state extraction. {eos_token}"]
#
# # 加载模型
# model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float16)
# model.eval()
#
# # 编码输入
# inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
# input_ids = inputs["input_ids"].to(model.device)
#
# # 检查分词结果
# print("Input IDs:", input_ids)
# print("Decoded Tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))
#
# # 获取 hidden states
# with torch.no_grad():
#     outputs = model(input_ids=input_ids, output_hidden_states=True)
#     hidden_states = outputs.hidden_states[-1]  # 最后一层隐藏状态
#
# # 提取 <EOS> token 的 hidden state
# eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")
# eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)
#
# # 提取 hidden state
# if eos_positions[0].numel() > 0:  # 检查是否存在 <EOS>
#     sentence_representations = hidden_states[eos_positions]
#     print("Sentence representation shape:", sentence_representations.shape)
# else:
#     print("No <EOS> token found in the input.")

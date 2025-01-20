# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 13:47
# @Author  : blue
# @Description : 多模态嵌入embedding api服务示例
# 启动命令 : uvicorn mm_embedding_api:app --host 0.0.0.0 --port 8001 --workers 32
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 设置为空字符串，表示禁用所有 GPU

os.environ["HF_HOME"] = "/mnt/zch/tmp"
os.environ["TMPDIR"] = "/mnt/zch/tmp"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

torch.set_num_threads(4)
from visual_bge.modeling import Visualized_BGE
from transformers import AutoTokenizer

# Initialize the FastAPI app
app = FastAPI()

# Load the model (You can specify the device id here)
device_id = 0  # Specify the GPU device
MAX_LENGTH = 512
model = Visualized_BGE(
    model_name_bge='/mnt/xlj/rag paper/models/bge-base-en-v1.5',
    model_weight="/mnt/xlj/rag paper/models/Visualized_base_en_v1.5.pth",
    from_pretrained='/mnt/xlj/rag paper/models/bge-base-en-v1.5'
)
# model.to(f'cuda:{device_id}')
model.to('cpu')
model.eval()
# 加载预训练的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='/mnt/xlj/rag paper/models/bge-base-en-v1.5',
    use_fast=False
)


def truncate_and_restore(text) -> str:
    encoded = tokenizer(
        text,
        padding=True,  # 自动填充，确保长度一致
        truncation=True,  # 启用截断
        max_length=MAX_LENGTH,  # 最大长度
        return_tensors='pt'  # 返回 PyTorch tensor
    )
    input_ids = encoded['input_ids']
    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return decoded_text


class InputRequest(BaseModel):
    text: str = None
    file_path: str = None


@app.post("/encode/")
async def get_text_embedding(request: InputRequest):
    try:
        if request.text and request.file_path:
            text = truncate_and_restore(text=request.text)
            with torch.no_grad():
                embedding = model.encode(text=text, image=request.file_path).cpu().tolist()
            torch.cuda.empty_cache()
        elif request.text:
            text = truncate_and_restore(text=request.text)
            with torch.no_grad():
                embedding = model.encode(text=text).cpu().tolist()
            torch.cuda.empty_cache()
        elif request.file_path:
            with torch.no_grad():
                embedding = model.encode(image=request.file_path).cpu().tolist()
            torch.cuda.empty_cache()
        else:
            raise HTTPException(status_code=500, detail=str('no input'))
        return {
            "embedding": embedding
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(object=e))


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "API is running smoothly!"}


# Run the FastAPI app using Uvicorn (you can use `uvicorn` command as well)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

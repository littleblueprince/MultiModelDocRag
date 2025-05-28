from util.gme_inference import GmeQwen2VL
import torch


class GMEEmbeddingService:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model = GmeQwen2VL(model_path=model_path, device=device)
        self.model.base.to(device)
        self.device = device

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        with torch.no_grad():
            embeddings, _ = self.model.get_text_embeddings(texts=texts)
        return embeddings.cpu().tolist()


from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import uvicorn

embedding_service = GMEEmbeddingService(
    model_path="",  # replace with your model path
    device="cuda:0",
)

app = FastAPI(title="GME Embedding API", version="1.0")


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


@app.post("/embed/text", response_model=EmbedResponse)
def embed_text(req: EmbedRequest):
    embeddings = embedding_service.embed_text(req.texts)
    return {"embeddings": embeddings}


@app.get("/")
def root():
    return {"msg": "GME Embedding API is up."}


if __name__ == "__main__":
    uvicorn.run("gme_server:app", host="0.0.0.0", port=8010, reload=False)

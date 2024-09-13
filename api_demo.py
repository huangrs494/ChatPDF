from pydantic import BaseModel

class ChatItem(BaseModel):
    input: str
    history: list

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import argparse
import os
from loguru import logger
from rag import Rag

app = FastAPI()

pwd_path = os.path.abspath(os.path.dirname(__file__))

def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--rerank_model_name", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corpus_files", type=str, default="data/sample.pdf")
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--num_expand_context_chunk", type=int, default=1)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8082)
    parser.add_argument("--share", action='store_true', help="share model")
    args = parser.parse_args()
    logger.info(args)
    return args

def get_model(args):
    model = Rag(
        generate_model_type=args.gen_model_type,
        generate_model_name_or_path=args.gen_model_name,
        lora_model_name_or_path=args.lora_model,
        corpus_files=args.corpus_files.split(','),
        device=args.device,
        int4=args.int4,
        int8=args.int8,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_expand_context_chunk=args.num_expand_context_chunk,
        rerank_model_name_or_path=args.rerank_model_name,
    )
    logger.info(f"chatpdf model: {model}")
    return model

def predict_stream(message, history):
    history_format = []
    for human, assistant in history:
        history_format.append([human, assistant])
    model.history = history_format
    for chunk in model.predict_stream(message):
        yield chunk

@app.on_event("startup")
async def startup_event():
    args = get_args()
    global model
    model = get_model(args)
    logger.info(f"chatpdf model: {model}")

@app.post('/chat_post')
async def chat_post(item: ChatItem):
    if not item.input:
        raise HTTPException(status_code=400, detail="No input provided")

    return StreamingResponse(predict_stream(item.input, item.history), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)

from typing import List
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from .stream.ollama_stream import get_ollama_stream_instance
from .utils.instance import Request

load_dotenv(".env")

app = FastAPI()

@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    messages = request.messages
    response = await get_ollama_stream_instance(messages, protocol)
    #response.headers['x-vercel-ai-data-stream'] = 'v1'
    return response

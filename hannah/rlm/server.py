"""OpenAI-compatible local RLM server scaffold."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from hannah.rlm.model import RLMModel

app = FastAPI(title="Hannah RLM Server", version="0.1.0")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "rlm-f1-strategy-v1"
    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    temperature: float = 0.2
    max_tokens: int = 2048
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": "rlm-f1-strategy-v1", "version": "0.1.0-scaffold"}


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest) -> ChatResponse:
    model = RLMModel()
    content = model.generate(
        messages=[message.model_dump() for message in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    return ChatResponse(
        id=f"rlm-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


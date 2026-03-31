"""
LLM wrapper that exposes OpenAI-style endpoints and routes requests to FoundryLocal phi-3.5-mini.

Run:
  set FOUNDRY_URL=http://127.0.0.1:11434
  .venv\Scripts\uvicorn tools.llm_wrapper:app --host 127.0.0.1 --port 8090

Endpoints:
- POST /v1/chat/completions  (OpenAI-style payload). Default model: phi-3.5-mini
- POST /v1/embeddings        (OpenAI-style payload). Default model: phi-embedding-3 (or configured)
- GET  /health

This service performs light validation, sets defaults, and forwards to the FoundryLocal endpoint. It also normalizes responses to OpenAI-like JSON when possible.
"""
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
import httpx
import asyncio

app = FastAPI(title="LLM Wrapper -> FoundryLocal (phi-3.5-mini)")
FOUNDRY_URL = os.environ.get('FOUNDRY_URL', 'http://127.0.0.1:11434')
DEFAULT_CHAT_MODEL = os.environ.get('DEFAULT_CHAT_MODEL', 'Phi-3-mini-128k-instruct-cuda-gpu:1')
DEFAULT_EMBED_MODEL = os.environ.get('DEFAULT_EMBED_MODEL', 'Phi-3-mini-128k-instruct-cuda-gpu:1')

client = httpx.AsyncClient(timeout=120.0)


@app.get('/health')
async def health():
    # check foundry health quickly
    try:
        r = await client.get(FOUNDRY_URL.rstrip('/') + '/health', timeout=5.0)
        status = r.status_code
    except Exception:
        status = None
    return {"status": "ok", "foundry_url": FOUNDRY_URL, "foundry_status": status}


async def forward_to_foundry(path: str, payload: Any, timeout: float = 60.0):
    target = FOUNDRY_URL.rstrip('/') + path
    try:
        r = await client.post(target, json=payload, timeout=timeout)
        return r
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error contacting foundry: {e}")


@app.post('/v1/chat/completions')
async def chat_completions(req: Request):
    body = await req.json()
    # ensure model
    model = body.get('model') or DEFAULT_CHAT_MODEL
    body['model'] = model

    # Basic safety: ensure messages exist
    if 'messages' not in body or not isinstance(body['messages'], list):
        raise HTTPException(status_code=400, detail='messages (list) is required')

    # Forward to foundry
    r = await forward_to_foundry('/v1/chat/completions', body)

    # Try to return JSON directly, else wrap
    try:
        return r.json()
    except Exception:
        text = r.text
        return {
            "id": None,
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": text}}]
        }


@app.post('/v1/embeddings')
async def embeddings(req: Request):
    body = await req.json()
    model = body.get('model') or DEFAULT_EMBED_MODEL
    body['model'] = model
    if 'input' not in body:
        raise HTTPException(status_code=400, detail='input is required')

    r = await forward_to_foundry('/v1/embeddings', body)
    try:
        return r.json()
    except Exception:
        raise HTTPException(status_code=502, detail='Invalid response from FoundryLocal')


# Graceful shutdown
@app.on_event('shutdown')
async def shutdown_event():
    await client.aclose()

"""
Simple OpenAI-style proxy to a FoundryLocal HTTP endpoint.

Usage:
  export FOUNDRY_URL=http://127.0.0.1:11434
  .venv\Scripts\uvicorn tools.foundry_proxy:app --host 127.0.0.1 --port 8080

This proxy does minimal translation. It will forward incoming JSON to the configured FOUNDRY_URL.
If the target endpoint is different, update FOUNDRY_URL accordingly in the environment.

Endpoints:
- POST /v1/chat/completions  (OpenAI-compatible request body)
- POST /v1/embeddings        (forwards to FOUNDRY_URL)
- GET  /health

Note: This is a passthrough proxy and may need adaptation to your FoundryLocal HTTP API.
"""
import os
import os.path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
import httpx

app = FastAPI(title="FoundryLocal OpenAI Proxy")
FOUNDRY_URL = os.environ.get('FOUNDRY_URL')
if not FOUNDRY_URL:
    FOUNDRY_URL = 'http://127.0.0.1:11434'  # update as needed

client = httpx.AsyncClient(timeout=60.0)


@app.get('/health')
async def health():
    return {"status": "ok", "foundry_url": FOUNDRY_URL}


@app.post('/v1/chat/completions')
async def chat_completions(req: Request):
    payload = await req.json()
    # Minimal conversion: forward payload to FOUNDRY_URL
    target = FOUNDRY_URL.rstrip('/') + '/v1/chat/completions'
    try:
        r = await client.post(target, json=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # If foundrylocal returns OpenAI-style JSON, passthrough
    try:
        return r.json()
    except Exception:
        # Wrap raw text
        return {"id": None, "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": r.text}}]}


@app.post('/v1/embeddings')
async def embeddings(req: Request):
    payload = await req.json()
    target = FOUNDRY_URL.rstrip('/') + '/v1/embeddings'
    try:
        r = await client.post(target, json=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    try:
        return r.json()
    except Exception:
        raise HTTPException(status_code=502, detail='Invalid response from foundry')

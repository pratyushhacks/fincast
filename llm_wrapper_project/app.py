"""LLM Wrapper project app: exposes OpenAI-style endpoints and forwards to FoundryLocal.

See README.md for run instructions.
"""
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

from dotenv import load_dotenv
load_dotenv(dotenv_path="../llm_wrapper_project/.env")

app = FastAPI(title="LLM Wrapper Project")
FOUNDRY_URL = os.environ.get('FOUNDRY_URL')
DEFAULT_CHAT_MODEL = os.environ.get('DEFAULT_CHAT_MODEL')
DEFAULT_EMBED_MODEL = os.environ.get('DEFAULT_EMBED_MODEL')

client = httpx.AsyncClient(timeout=120.0)


@app.get('/health')
async def health():
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
    model = body.get('model') or DEFAULT_CHAT_MODEL
    body['model'] = model
    if 'messages' not in body or not isinstance(body['messages'], list):
        raise HTTPException(status_code=400, detail='messages (list) is required')

    r = await forward_to_foundry('/v1/chat/completions', body)

    if r.status_code >= 400:
        try:
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception:
            return JSONResponse(status_code=r.status_code, content={"error": r.text})

    try:
        data = r.json()
        # Normalize Foundry's delta-only response to standard OpenAI message format
        for choice in data.get("choices", []):
            if not choice.get("message") and choice.get("delta"):
                choice["message"] = {
                    "role": choice["delta"].get("role", "assistant"),
                    "content": choice["delta"].get("content", "")
                }
        return data
    except Exception:
        return {
            "id": None,
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": r.text}}]
        }


# @app.post('/v1/embeddings')
# async def embeddings(req: Request):
#     body = await req.json()
#     model = body.get('model') or DEFAULT_EMBED_MODEL
#     body['model'] = model
#     if 'input' not in body:
#         raise HTTPException(status_code=400, detail='input is required')

#     r = await forward_to_foundry('/v1/embeddings', body)
#     try:
#         return r.json()
#     except Exception:
#         raise HTTPException(status_code=502, detail='Invalid response from FoundryLocal')


@app.on_event('shutdown')
async def shutdown_event():
    await client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=int(os.getenv("PORT", "8091")), log_level="info")
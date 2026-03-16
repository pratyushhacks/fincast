LLM Wrapper Project

This is a standalone project that exposes OpenAI-style endpoints and forwards requests to FoundryLocal models (phi-4-mini-reasoning for chat; phi-3-mini-128k for embeddings).

Setup

1. Copy .env.template to .env and edit if needed.
2. Create venv and install deps:
   python -m venv .venv
   .venv\Scripts\python -m pip install --upgrade pip
   .venv\Scripts\pip install -r requirements.txt

Run

.venv\Scripts\uvicorn app:app --host 127.0.0.1 --port 8090

VSCode

Open this folder in VSCode to use the provided tasks and launch configuration.

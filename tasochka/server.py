"""FastAPI server, Ollama-compatible.

Endpoints implemented:
    POST /api/chat   — streaming NDJSON, the one index.html uses
    GET  /api/tags   — minimal model list so clients can probe

Run:
    python -m tasochka.server --checkpoint ./checkpoints/tasochka --port 11434
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .generate import generate_stream, load


class Message(BaseModel):
    role: str
    content: str


class ChatOptions(BaseModel):
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = None
    top_k: Optional[int] = 40
    max_new_chars: Optional[int] = 300


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    stream: Optional[bool] = True
    options: Optional[ChatOptions] = None


def build_prompt(messages: List[Message]) -> str:
    """Render history into the dialogue shape the char model was trained on:
    each user/assistant turn becomes a line prefixed with '- '."""
    parts: List[str] = []
    for m in messages:
        if m.role == "system":
            continue
        text = m.content.strip()
        if not text:
            continue
        if not text.startswith("-"):
            text = "- " + text
        parts.append(text)
    parts.append("-")  # prompt for assistant's reply
    return "\n".join(parts) + " "


def create_app(checkpoint_dir: Path, model_name: str = "tasochka-ai") -> FastAPI:
    model, tokenizer, cfg = load(checkpoint_dir)

    app = FastAPI(title="Tasochka AI")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/tags")
    def tags() -> Dict[str, Any]:
        return {
            "models": [
                {"name": model_name, "model": model_name, "size": 0, "details": {}}
            ]
        }

    @app.get("/")
    def root() -> Dict[str, str]:
        return {"status": "ok", "model": model_name}

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        opts = req.options or ChatOptions()
        temperature = opts.temperature if opts.temperature is not None else 0.8
        top_k = opts.top_k if opts.top_k is not None else 40
        max_new = opts.max_new_chars if opts.max_new_chars is not None else 300
        prompt = build_prompt(req.messages)
        created = int(time.time())

        async def stream():
            try:
                # break out of the inner generator into newline so reply doesn't
                # accidentally continue into the next "user" turn
                buffer = ""
                for piece in generate_stream(
                    model, tokenizer, cfg, prompt,
                    max_new_chars=max_new,
                    temperature=temperature,
                    top_k=top_k,
                ):
                    buffer += piece
                    # stop generation if the model starts a new "user" turn
                    if "\n-" in buffer and buffer.rstrip().endswith("-"):
                        # strip the trailing "\n-" we don't want to emit
                        piece = piece.rsplit("-", 1)[0]
                        if piece:
                            yield ndjson_chunk(model_name, piece)
                        break
                    yield ndjson_chunk(model_name, piece)
                    # yield to event loop so client sees streaming
                    await asyncio.sleep(0)
                yield ndjson_final(model_name, created)
            except Exception as exc:  # last-resort guard
                yield json.dumps({"error": str(exc)}) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app


def ndjson_chunk(model_name: str, content: str) -> str:
    return json.dumps({
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "message": {"role": "assistant", "content": content},
        "done": False,
    }, ensure_ascii=False) + "\n"


def ndjson_final(model_name: str, created: int) -> str:
    return json.dumps({
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "total_duration": int((time.time() - created) * 1e9),
    }, ensure_ascii=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                       default=Path(__file__).resolve().parent.parent / "checkpoints" / "tasochka")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--model-name", default="tasochka-ai")
    return parser.parse_args()


def main() -> None:
    import uvicorn
    args = parse_args()
    app = create_app(args.checkpoint, args.model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

from fastapi import APIRouter, HTTPException
import logging
import time

from agents.registry import AVAILABLE_AGENTS
from schemas.chat import ChatCompletionRequest, ModelCard

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/v1/models")
async def get_models():
    models = [
        ModelCard(id=agent_id, created=int(time.time())).model_dump()
        for agent_id in AVAILABLE_AGENTS.keys()
    ]
    return {"object": "list", "data": models}


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    agent = AVAILABLE_AGENTS.get(request.model)

    if not agent:
        raise HTTPException(status_code=404, detail="Model not found in registry")

    started = time.perf_counter()
    latest = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        "",
    )
    logger.info(
        "chat/completions model=%s msgs=%s latest=%r",
        request.model,
        len(request.messages),
        (latest or "")[:120],
    )

    response_content = await agent.run(request.messages)

    logger.info(
        "chat/completions done model=%s in %.2fs out_chars=%s",
        request.model,
        time.perf_counter() - started,
        len(response_content or ""),
    )

    return {
        "id": f"chatcmpl-{request.model}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                },
                "finish_reason": "stop",
            }
        ],
    }

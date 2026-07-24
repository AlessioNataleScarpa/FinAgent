from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from pipeline.graph import graph


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await graph.start()
    try:
        yield
    finally:
        await graph.stop()


app = FastAPI(title="DL-2026 Agent Backend", lifespan=lifespan)
app.include_router(router)

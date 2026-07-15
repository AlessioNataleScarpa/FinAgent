from fastapi import FastAPI

from api.routes import router

app = FastAPI(title="DL-2026 Agent Backend")
app.include_router(router)

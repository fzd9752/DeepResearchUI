from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.routes import config, debug, research, scenarios, upload

settings = get_settings()

app = FastAPI(
    title="Yunque DeepResearch API",
    version="1.0.0",
    description="Deep research Agent API service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research.router, prefix="/api/research", tags=["research"])
app.include_router(scenarios.router, prefix="/api/scenarios", tags=["scenarios"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(config.router, prefix="/api/config", tags=["config"])
app.include_router(debug.router, prefix="/api/debug", tags=["debug"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)

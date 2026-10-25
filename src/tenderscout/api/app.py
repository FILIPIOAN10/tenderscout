from fastapi import FastAPI

from tenderscout.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title="TenderScout API", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env}

    return app


app = create_app()

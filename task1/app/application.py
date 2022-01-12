
from app.routers import pricing
from fastapi import FastAPI

def create_app():
    app = FastAPI(
        title="Pricing micro-service",
        description="Pricing prediction api endpoint",
        version="1.0.0",
        openapi_url="/openapi.json",
        docs_url="/",
        redoc_url="/redoc"
    )

    app.include_router(
        pricing.router,
        prefix="/api/v1/pricing",
        tags=["pricing"]
    )
    
    return app
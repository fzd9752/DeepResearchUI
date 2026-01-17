from fastapi import APIRouter

from api.config import get_settings


router = APIRouter()


@router.get("")
async def get_config():
    settings = get_settings()
    return {
        "available_models": settings.available_models,
        "default_model": settings.default_model,
        "max_rollouts": settings.max_rollouts,
        "default_rollouts": settings.default_rollouts,
        "features": settings.features,
    }

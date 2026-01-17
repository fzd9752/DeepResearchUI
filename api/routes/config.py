import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.config import get_settings, reload_settings
from api.services.state import research_service


router = APIRouter()


@router.get("")
async def get_config():
    settings = get_settings()
    return {
        "available_models": settings.available_models,
        "default_model": settings.default_model,
        "default_memory_model": settings.default_memory_model,
        "default_summary_model": settings.default_summary_model,
        "max_rollouts": settings.max_rollouts,
        "default_rollouts": settings.default_rollouts,
        "features": settings.features,
    }


class UpdateConfigRequest(BaseModel):
    default_model: Optional[str] = None
    memory_model: Optional[str] = None
    summary_model: Optional[str] = None
    available_models: Optional[List[str]] = Field(default=None)


def _update_env_file(env_path: Path, updates: dict) -> None:
    if not updates:
        return

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    existing = {}
    for idx, line in enumerate(lines):
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        existing[key] = idx

    for key, value in updates.items():
        line_value = f"{key}={value}"
        if key in existing:
            lines[existing[key]] = line_value
        else:
            lines.append(line_value)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@router.put("")
async def update_config(payload: UpdateConfigRequest):
    updates = {}
    if payload.default_model:
        updates["DEFAULT_MODEL"] = payload.default_model
        updates["LLM_MODEL"] = payload.default_model
    if payload.memory_model:
        updates["MEMORY_MODEL"] = payload.memory_model
    if payload.summary_model:
        updates["SUMMARY_MODEL_NAME"] = payload.summary_model
    if payload.available_models:
        updates["AVAILABLE_MODELS"] = ",".join(payload.available_models)

    if not updates:
        raise HTTPException(status_code=400, detail="No config updates provided")

    for key, value in updates.items():
        os.environ[key] = value

    env_path = Path(__file__).resolve().parents[2] / ".env"
    _update_env_file(env_path, updates)

    new_settings = reload_settings()
    research_service.settings = new_settings

    return {
        "available_models": new_settings.available_models,
        "default_model": new_settings.default_model,
        "default_memory_model": new_settings.default_memory_model,
        "default_summary_model": new_settings.default_summary_model,
        "max_rollouts": new_settings.max_rollouts,
        "default_rollouts": new_settings.default_rollouts,
        "features": new_settings.features,
    }

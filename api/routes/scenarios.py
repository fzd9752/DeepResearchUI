from fastapi import APIRouter

from api.services.scenario_store import load_scenarios


router = APIRouter()


@router.get("")
async def list_scenarios():
    return {"scenarios": load_scenarios()}

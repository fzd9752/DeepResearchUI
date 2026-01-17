from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from starlette.responses import StreamingResponse

from api.config import get_settings
from api.models.research import (
    CreateResearchRequest,
    ResearchOptions,
    TaskListItem,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
)
from api.services.state import research_service, task_manager


router = APIRouter()


@router.post("", response_model=TaskResponse)
async def create_research(request: CreateResearchRequest) -> TaskResponse:
    settings = get_settings()
    options = request.options or ResearchOptions()
    if options.rollout_count > settings.max_rollouts:
        raise HTTPException(
            status_code=400,
            detail=f"rollout_count must be <= {settings.max_rollouts}",
        )
    if options.model not in settings.available_models:
        raise HTTPException(status_code=400, detail="model is not available")

    task = await research_service.create_task(request)
    return research_service.get_task_response(task, include_stream_url=True)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_research(task_id: str) -> TaskResponse:
    task = task_manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return research_service.get_task_response(task, include_stream_url=True)


@router.get("/{task_id}/stream")
async def stream_research(task_id: str):
    task = task_manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return StreamingResponse(
        research_service.stream_events(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.delete("/{task_id}")
async def cancel_research(task_id: str):
    task = research_service.cancel_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task.id,
        "status": task.status.value,
        "message": "Task cancelled successfully",
    }


@router.get("", response_model=TaskListResponse)
async def list_research(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[TaskStatus] = None,
) -> TaskListResponse:
    tasks = task_manager.list()
    if status:
        tasks = [task for task in tasks if task.status == status]

    tasks.sort(key=lambda item: item.created_at, reverse=True)
    total = len(tasks)
    items = tasks[offset : offset + limit]

    return TaskListResponse(
        total=total,
        items=[
            TaskListItem(
                task_id=task.id,
                question=task.request.question,
                status=task.status,
                created_at=task.created_at.isoformat(),
                duration_seconds=task.duration_seconds,
            )
            for task in items
        ],
    )

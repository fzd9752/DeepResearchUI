import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from api.models.research import CreateResearchRequest, ResearchResult, RolloutStatus, TaskStatus
from api.services.event_emitter import EventHub


def generate_task_id() -> str:
    return f"task_{uuid.uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RolloutState:
    id: int
    status: RolloutStatus = RolloutStatus.PENDING
    rounds: int = 0
    current_round: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


@dataclass
class TaskState:
    id: str
    request: CreateResearchRequest
    status: TaskStatus
    created_at: datetime
    event_hub: EventHub
    loop: asyncio.AbstractEventLoop
    cancel_event: threading.Event
    rollouts: List[RolloutState]
    result: Optional[ResearchResult] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    memory_units: List[dict] = field(default_factory=list)
    supervisor_logs: List[dict] = field(default_factory=list)


class TaskManager:
    def __init__(self):
        self._tasks: Dict[str, TaskState] = {}
        self._lock = threading.Lock()

    def add(self, task: TaskState) -> None:
        with self._lock:
            self._tasks[task.id] = task

    def get(self, task_id: str) -> Optional[TaskState]:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> List[TaskState]:
        with self._lock:
            return list(self._tasks.values())

    def update(self, task_id: str, updater) -> Optional[TaskState]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            updater(task)
            return task

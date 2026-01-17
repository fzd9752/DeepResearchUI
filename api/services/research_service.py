import asyncio
import inspect
import os
import time
from pathlib import Path
import threading
from typing import Iterable, Tuple

import json5
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Dict, Optional, List

from api.config import APISettings
from api.models.events import SSEEvent
from api.models.research import (
    CreateResearchRequest,
    ResearchOptions,
    ResearchResult,
    ResearchStatistics,
    RolloutStatus,
    TaskResponse,
    TaskStatus,
)
from api.services.event_emitter import EventHub
from api.services.task_manager import RolloutState, TaskManager, TaskState, utc_now


class ResearchService:
    def __init__(self, settings: APISettings, task_manager: TaskManager):
        self.settings = settings
        self.task_manager = task_manager
        max_workers = int(os.getenv("API_WORKERS", "4"))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._debug_log_lock = threading.Lock()
        self._debug_log_dir = Path(self.settings.debug_log_dir)

    async def create_task(self, request: CreateResearchRequest) -> TaskState:
        options = request.options or ResearchOptions()
        rollout_count = options.rollout_count
        rollouts = [RolloutState(id=i) for i in range(1, rollout_count + 1)]
        loop = asyncio.get_running_loop()

        cancel_event = self._create_cancel_event()
        task = TaskState(
            id=self._generate_task_id(),
            request=request,
            status=TaskStatus.PENDING,
            created_at=utc_now(),
            event_hub=EventHub(),
            loop=loop,
            cancel_event=cancel_event,
            rollouts=rollouts,
        )
        self.task_manager.add(task)

        await self._enqueue_event(
            task,
            "task_start",
            {
                "task_id": task.id,
                "question": request.question,
                "rollout_count": rollout_count,
                "started_at": task.created_at.isoformat(),
            },
        )

        if self.settings.fake_mode:
            loop.run_in_executor(self.executor, self._run_fake, task.id)
        else:
            loop.run_in_executor(self.executor, self._run_research, task.id)
        return task

    def cancel_task(self, task_id: str) -> Optional[TaskState]:
        task = self.task_manager.get(task_id)
        if not task:
            return None
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return task
        task.cancel_event.set()
        task.status = TaskStatus.CANCELLED
        task.completed_at = utc_now()
        task.duration_seconds = (task.completed_at - task.created_at).total_seconds()
        task.error = "cancelled"
        self._emit_event_threadsafe(
            task,
            "task_error",
            {
                "task_id": task.id,
                "error_type": "cancelled",
                "message": "Task cancelled by user",
                "failed_at": task.completed_at.isoformat(),
            },
        )
        return task

    def get_task_response(self, task: TaskState, include_stream_url: bool = False) -> TaskResponse:
        rollouts = [
            {
                "id": rollout.id,
                "status": rollout.status,
                "rounds": rollout.rounds or None,
                "current_round": rollout.current_round,
                "duration_seconds": rollout.duration_seconds,
            }
            for rollout in task.rollouts
        ]

        result = task.result
        stream_url = f"/api/research/{task.id}/stream" if include_stream_url else None
        return TaskResponse(
            task_id=task.id,
            status=task.status,
            question=task.request.question,
            created_at=task.created_at.isoformat(),
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            duration_seconds=task.duration_seconds,
            rollouts=rollouts,
            result=result,
            stream_url=stream_url,
            memory_units=task.memory_units,
            supervisor_logs=task.supervisor_logs,
        )

    async def stream_events(self, task_id: str) -> AsyncGenerator[str, None]:
        task = self.task_manager.get(task_id)
        if not task:
            yield SSEEvent(event="error", data={"message": "Task not found"}).format()
            return
        queue, history = await task.event_hub.subscribe()
        try:
            for payload in history:
                event_type = payload.get("event")
                data = payload.get("data", {})
                yield SSEEvent(event=event_type, data=data).format()
                if event_type in ("task_complete", "task_error"):
                    return

            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue

                event_type = payload.get("event")
                data = payload.get("data", {})
                yield SSEEvent(event=event_type, data=data).format()

                if event_type in ("task_complete", "task_error"):
                    break
        finally:
            await task.event_hub.unsubscribe(queue)

    def _run_research(self, task_id: str) -> None:
        task = self.task_manager.get(task_id)
        if not task:
            return
        if task.cancel_event.is_set() or task.status == TaskStatus.CANCELLED:
            self._finalize_cancel(task)
            return

        task.status = TaskStatus.RUNNING
        task_start = time.time()

        options = task.request.options or ResearchOptions()

        for rollout_state in task.rollouts:
            if task.cancel_event.is_set():
                self._finalize_cancel(task)
                return

            rollout_state.status = RolloutStatus.RUNNING
            rollout_state.started_at = utc_now()
            self._emit_event_threadsafe(
                task,
                "rollout_start",
                {
                    "rollout_id": rollout_state.id,
                    "started_at": rollout_state.started_at.isoformat(),
                },
            )

            try:
                result = self._run_rollout(task, rollout_state.id, options)
                if task.cancel_event.is_set() or result.get("termination") == "cancelled":
                    self._finalize_cancel(task)
                    return
                rollout_state.status = RolloutStatus.COMPLETED
                rollout_state.completed_at = utc_now()
                rollout_state.duration_seconds = (
                    rollout_state.completed_at - rollout_state.started_at
                ).total_seconds()

                task.result = self._build_result(result)
                self._emit_event_threadsafe(
                    task,
                    "rollout_complete",
                    {
                        "rollout_id": rollout_state.id,
                        "duration_seconds": rollout_state.duration_seconds,
                        "status": rollout_state.status.value,
                        "rounds": rollout_state.rounds,
                        "answer_preview": self._build_answer_preview(task.result.answer),
                        "statistics": task.result.statistics.model_dump(),
                    },
                )
            except Exception as exc:  # pylint: disable=broad-except
                if self.settings.enable_debug_stream:
                    import traceback

                    self._emit_debug(
                        task,
                        rollout_state.id,
                        "stderr",
                        traceback.format_exc(),
                    )
                rollout_state.status = RolloutStatus.FAILED
                rollout_state.error = str(exc)
                rollout_state.completed_at = utc_now()
                rollout_state.duration_seconds = (
                    rollout_state.completed_at - rollout_state.started_at
                ).total_seconds()
                self._emit_event_threadsafe(
                    task,
                    "task_error",
                    {
                        "task_id": task.id,
                        "error_type": "rollout_failed",
                        "message": str(exc),
                        "failed_at": utc_now().isoformat(),
                    },
                )
                task.status = TaskStatus.FAILED
                task.completed_at = utc_now()
                task.duration_seconds = time.time() - task_start
                return

        task.status = TaskStatus.COMPLETED
        task.completed_at = utc_now()
        task.duration_seconds = time.time() - task_start

        result_payload = task.result or ResearchResult(
            answer="",
            sources=[],
            statistics=ResearchStatistics(),
        )
        result_payload.statistics = self._compute_statistics_from_result(result_payload)
        task.result = result_payload

        self._emit_event_threadsafe(
            task,
            "progress_update",
            {
                "overall_progress": 1.0,
                "elapsed_seconds": int(task.duration_seconds or 0),
                "estimated_remaining_seconds": 0,
                "llm_calls": {
                    "current": result_payload.statistics.llm_calls,
                    "max": result_payload.statistics.llm_calls,
                },
                "rollouts_status": [
                    {
                        "id": rollout.id,
                        "progress": 1.0 if rollout.status == RolloutStatus.COMPLETED else 0.0,
                        "rounds": rollout.rounds,
                    }
                    for rollout in task.rollouts
                ],
            },
        )

        self._emit_event_threadsafe(
            task,
            "task_complete",
            {
                "task_id": task.id,
                "status": task.status.value,
                "answer": result_payload.answer,
                "sources": result_payload.sources,
                "statistics": result_payload.statistics.model_dump(),
                "completed_at": task.completed_at.isoformat(),
            },
        )

    def _run_fake(self, task_id: str) -> None:
        from api.services.fake_log_replay import FakeLogReplay

        task = self.task_manager.get(task_id)
        if not task:
            return
        task.status = TaskStatus.RUNNING
        task_start = time.time()

        log_path = Path(self.settings.fake_log_path)
        if not log_path.exists():
            self._emit_event_threadsafe(
                task,
                "task_error",
                {
                    "task_id": task.id,
                    "error_type": "fake_log_missing",
                    "message": f"Fake log not found: {log_path}",
                    "failed_at": utc_now().isoformat(),
                },
            )
            task.status = TaskStatus.FAILED
            task.completed_at = utc_now()
            task.duration_seconds = time.time() - task_start
            return

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        replay = FakeLogReplay(log_text)
        rounds = max(1, self.settings.fake_rounds)
        delay = max(0, self.settings.fake_delay_ms)

        question = task.request.question or replay.parse_question()
        thinking_blocks = replay.parse_thinking_blocks(rounds)
        tool_calls = replay.parse_tool_calls(rounds)
        tool_responses = replay.parse_tool_responses(rounds)
        memory_blocks = replay.parse_memory_blocks(rounds)

        for rollout_state in task.rollouts:
            if task.cancel_event.is_set():
                self._finalize_cancel(task)
                return

            rollout_state.status = RolloutStatus.RUNNING
            rollout_state.started_at = utc_now()
            self._emit_event_threadsafe(
                task,
                "rollout_start",
                {
                    "rollout_id": rollout_state.id,
                    "started_at": rollout_state.started_at.isoformat(),
                },
            )

            for idx in range(rounds):
                if task.cancel_event.is_set():
                    self._finalize_cancel(task)
                    return

                round_id = idx + 1
                self._handle_agent_event(
                    task.id,
                    rollout_state.id,
                    "round_start",
                    {
                        "rollout_id": rollout_state.id,
                        "round": round_id,
                        "started_at": utc_now().isoformat(),
                    },
                )

                if idx < len(thinking_blocks):
                    self._handle_agent_event(
                        task.id,
                        rollout_state.id,
                        "round_thinking",
                        {
                            "rollout_id": rollout_state.id,
                            "round": round_id,
                            "content": thinking_blocks[idx],
                        },
                    )

                tool_name = "search"
                tool_args: Dict = {}
                if idx < len(tool_calls):
                    tool_name, tool_args = tool_calls[idx]

                self._handle_agent_event(
                    task.id,
                    rollout_state.id,
                    "round_acting",
                    {
                        "rollout_id": rollout_state.id,
                        "round": round_id,
                        "tool": tool_name,
                        "arguments": tool_args,
                    },
                )

                response = tool_responses[idx] if idx < len(tool_responses) else ""
                preview = response.replace("\n", " ")
                if len(preview) > 240:
                    preview = preview[:240] + "..."
                observing_payload = {
                    "rollout_id": rollout_state.id,
                    "round": round_id,
                    "tool": tool_name,
                    "result_summary": f"Tool returned {len(response)} chars",
                    "result_preview": preview,
                }
                if self.settings.full_tool_response and response:
                    observing_payload["result_full"] = response
                self._handle_agent_event(
                    task.id,
                    rollout_state.id,
                    "round_observing",
                    observing_payload,
                )

                self._handle_agent_event(
                    task.id,
                    rollout_state.id,
                    "round_complete",
                    {
                        "rollout_id": rollout_state.id,
                        "round": round_id,
                        "status": "success",
                        "tool": tool_name,
                        "duration_ms": 500,
                    },
                )

                if memory_blocks and idx < len(memory_blocks):
                    self._handle_agent_event(
                        task.id,
                        rollout_state.id,
                        "memory_update",
                        {
                            "rollout_id": rollout_state.id,
                            "round": round_id,
                            "action": "update",
                            "memory_unit": {
                                "summary": memory_blocks[idx],
                                "rounds_index": [round_id],
                                "sub_goal": "Fake memory from log",
                                "tools_log": [],
                            },
                        },
                    )

                llm_calls_used = (rollout_state.id - 1) * rounds + round_id
                overall_progress = llm_calls_used / max(1, rounds * len(task.rollouts))
                self._handle_agent_event(
                    task.id,
                    rollout_state.id,
                    "progress_update",
                    {
                        "overall_progress": float(f"{overall_progress:.4f}"),
                        "elapsed_seconds": int(time.time() - task_start),
                        "estimated_remaining_seconds": None,
                        "llm_calls": {
                            "current": llm_calls_used,
                            "max": rounds * len(task.rollouts),
                        },
                        "rollouts_status": [
                            {
                                "id": rollout_state.id,
                                "progress": float(f"{overall_progress:.4f}"),
                                "rounds": round_id,
                            }
                        ],
                    },
                )

                if delay:
                    time.sleep(delay / 1000)

            rollout_state.status = RolloutStatus.COMPLETED
            rollout_state.completed_at = utc_now()
            rollout_state.duration_seconds = (
                rollout_state.completed_at - rollout_state.started_at
            ).total_seconds()

            stats = self._compute_statistics_from_tool_calls(tool_calls)
            self._emit_event_threadsafe(
                task,
                "rollout_complete",
                {
                    "rollout_id": rollout_state.id,
                    "duration_seconds": rollout_state.duration_seconds,
                    "status": rollout_state.status.value,
                    "rounds": rollout_state.rounds,
                    "answer_preview": "Fake report preview from log-based stream...",
                    "statistics": stats.model_dump(),
                },
            )

        task.status = TaskStatus.COMPLETED
        task.completed_at = utc_now()
        task.duration_seconds = time.time() - task_start

        final_stats = self._compute_statistics_from_tool_calls(tool_calls)
        task.result = ResearchResult(
            answer="## Fake Report\n\nThis report is generated from a replay log for UI testing.",
            sources=[],
            statistics=final_stats,
        )

        self._emit_event_threadsafe(
            task,
            "progress_update",
            {
                "overall_progress": 1.0,
                "elapsed_seconds": int(task.duration_seconds),
                "estimated_remaining_seconds": 0,
                "llm_calls": {
                    "current": final_stats.llm_calls,
                    "max": final_stats.llm_calls,
                },
                "rollouts_status": [
                    {
                        "id": rollout.id,
                        "progress": 1.0,
                        "rounds": rollout.rounds,
                    }
                    for rollout in task.rollouts
                ],
            },
        )

        self._emit_event_threadsafe(
            task,
            "task_complete",
            {
                "task_id": task.id,
                "status": task.status.value,
                "answer": task.result.answer,
                "sources": task.result.sources,
                "statistics": task.result.statistics.model_dump(),
                "completed_at": task.completed_at.isoformat(),
            },
        )

    def _run_rollout(
        self, task: TaskState, rollout_id: int, options: ResearchOptions
    ) -> Dict:
        from inference.agent import Agent

        llm_cfg = {
            "model": self.settings.model_path,
            "generate_cfg": {
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "presence_penalty": self.settings.presence_penalty,
            },
        }

        agent = Agent(
            llm=llm_cfg,
            function_list=["search", "visit", "google_scholar", "CodeExecutor"],
            event_callback=lambda event_type, data: self._handle_agent_event(
                task.id, rollout_id, event_type, data
            ),
            cancel_checker=task.cancel_event.is_set,
            enable_context_management=options.enable_memory,
            enable_supervisor=options.enable_supervisor,
            model_name=options.model,
            emit_full_tool_response=self.settings.full_tool_response,
        )

        data = {
            "item": {
                "question": task.request.question,
                "answer": "",
            },
            "rollout_idx": rollout_id,
        }

        if self.settings.enable_debug_stream:
            from api.services.debug_stream import DebugEmitter, ensure_debug_streams

            stdout_mux, stderr_mux = ensure_debug_streams()
            stdout_emitter = DebugEmitter(
                lambda line: self._emit_debug(task, rollout_id, "stdout", line)
            )
            stderr_emitter = DebugEmitter(
                lambda line: self._emit_debug(task, rollout_id, "stderr", line)
            )
            stdout_mux.register(stdout_emitter.write)
            stderr_mux.register(stderr_emitter.write)
            try:
                result = agent._run(data, self.settings.model_path)
                if inspect.iscoroutine(result):
                    return asyncio.run(result)
                return result
            finally:
                stdout_emitter.flush()
                stderr_emitter.flush()
                stdout_mux.unregister()
                stderr_mux.unregister()

        result = agent._run(data, self.settings.model_path)
        if inspect.iscoroutine(result):
            return asyncio.run(result)
        return result

    def _handle_agent_event(
        self, task_id: str, rollout_id: int, event_type: str, data: Dict
    ) -> None:
        task = self.task_manager.get(task_id)
        if not task:
            return

        payload = dict(data or {})
        payload.setdefault("rollout_id", rollout_id)

        if event_type == "round_start":
            self._update_rollout(task_id, rollout_id, payload.get("round"))
        elif event_type == "round_complete":
            self._increment_rollout_rounds(task_id, rollout_id, payload.get("round"))
        elif event_type == "memory_update":
            memory_unit = payload.get("memory_unit") if isinstance(payload, dict) else None
            if not memory_unit and isinstance(payload, dict):
                memory_unit = payload.get("summary")
            if isinstance(memory_unit, dict):
                task.memory_units.append(memory_unit)
        elif event_type == "supervisor_event":
            task.supervisor_logs.append(payload)

        self._emit_event_threadsafe(task, event_type, payload)

    def _update_rollout(self, task_id: str, rollout_id: int, round_index: Optional[int]) -> None:
        def updater(task: TaskState) -> None:
            for rollout in task.rollouts:
                if rollout.id == rollout_id:
                    rollout.current_round = round_index
                    break

        self.task_manager.update(task_id, updater)

    def _increment_rollout_rounds(self, task_id: str, rollout_id: int, round_index: Optional[int]) -> None:
        def updater(task: TaskState) -> None:
            for rollout in task.rollouts:
                if rollout.id == rollout_id:
                    rollout.rounds += 1
                    rollout.current_round = round_index or rollout.rounds
                    break

        self.task_manager.update(task_id, updater)

    async def _enqueue_event(self, task: TaskState, event_type: str, data: Dict) -> None:
        await task.event_hub.publish({"event": event_type, "data": data})

    def _emit_event_threadsafe(self, task: TaskState, event_type: str, data: Dict) -> None:
        if not task.loop or task.loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(
            task.event_hub.publish({"event": event_type, "data": data}), task.loop
        )

    def _emit_debug(self, task: TaskState, rollout_id: int, source: str, message: str) -> None:
        self._append_debug_log(task.id, rollout_id, source, message)
        self._emit_event_threadsafe(
            task,
            "debug_log",
            {
                "rollout_id": rollout_id,
                "source": source,
                "message": message,
                "timestamp": utc_now().isoformat(),
            },
        )

    def _append_debug_log(self, task_id: str, rollout_id: int, source: str, message: str) -> None:
        try:
            self._debug_log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        log_path = self._debug_log_dir / f"{task_id}.log"
        timestamp = utc_now().isoformat()
        line = f"[{timestamp}] [{source}] [rollout:{rollout_id}] {message}\n"
        with self._debug_log_lock:
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                return

    def _build_result(self, raw_result: Dict) -> ResearchResult:
        answer = raw_result.get("prediction") or raw_result.get("answer") or ""
        statistics = self._compute_statistics(raw_result)
        return ResearchResult(
            answer=answer,
            sources=[],
            statistics=statistics,
        )

    def _build_answer_preview(self, answer: str, limit: int = 200) -> str:
        if not answer:
            return ""
        trimmed = answer.strip().replace("\n", " ")
        return trimmed if len(trimmed) <= limit else trimmed[:limit] + "..."

    def _extract_tool_calls(self, content: str) -> Iterable[Tuple[str, dict]]:
        if not content:
            return []
        calls = []
        start = 0
        while True:
            open_idx = content.find("<tool_call>", start)
            if open_idx == -1:
                break
            close_idx = content.find("</tool_call>", open_idx)
            if close_idx == -1:
                break
            block = content[open_idx + len("<tool_call>") : close_idx].strip()
            start = close_idx + len("</tool_call>")

            json_block = block
            if "<code>" in block:
                json_block = block.split("<code>", 1)[0].strip()
            try:
                payload = json5.loads(json_block)
            except Exception:
                continue
            tool_name = payload.get("name")
            tool_args = payload.get("arguments", {}) if isinstance(payload, dict) else {}
            if tool_name:
                calls.append((tool_name, tool_args))
        return calls

    def _compute_statistics(self, raw_result: Dict) -> ResearchStatistics:
        stats = ResearchStatistics()
        messages = raw_result.get("messages") or []
        stats.llm_calls = sum(
            1 for msg in messages if msg.get("role") == "assistant"
        )
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            for tool_name, tool_args in self._extract_tool_calls(content):
                normalized = tool_name.lower()
                if normalized in ("search", "google_scholar"):
                    stats.total_searches += 1
                elif normalized == "visit":
                    stats.total_visits += 1
                    urls = tool_args.get("url") or tool_args.get("urls")
                    if isinstance(urls, list):
                        stats.total_sources += len(urls)
                    elif isinstance(urls, str) and urls.strip():
                        stats.total_sources += 1
        return stats

    def _compute_statistics_from_result(self, result: ResearchResult) -> ResearchStatistics:
        return result.statistics or ResearchStatistics()

    def _compute_statistics_from_tool_calls(
        self, tool_calls: List[Tuple[str, Dict]]
    ) -> ResearchStatistics:
        stats = ResearchStatistics()
        stats.llm_calls = len(tool_calls)
        for tool_name, tool_args in tool_calls:
            normalized = tool_name.lower()
            if normalized in ("search", "google_scholar"):
                stats.total_searches += 1
            elif normalized == "visit":
                stats.total_visits += 1
                urls = tool_args.get("url") or tool_args.get("urls")
                if isinstance(urls, list):
                    stats.total_sources += len(urls)
                elif isinstance(urls, str) and urls.strip():
                    stats.total_sources += 1
        return stats

    def _finalize_cancel(self, task: TaskState) -> None:
        if task.error == "cancelled" and task.completed_at and task.duration_seconds:
            return
        task.status = TaskStatus.CANCELLED
        task.completed_at = utc_now()
        task.duration_seconds = (task.completed_at - task.created_at).total_seconds()
        self._emit_event_threadsafe(
            task,
            "task_error",
            {
                "task_id": task.id,
                "error_type": "cancelled",
                "message": "Task cancelled",
                "failed_at": task.completed_at.isoformat(),
            },
        )

    def _generate_task_id(self) -> str:
        from api.services.task_manager import generate_task_id

        return generate_task_id()

    def _create_cancel_event(self):
        import threading

        return threading.Event()

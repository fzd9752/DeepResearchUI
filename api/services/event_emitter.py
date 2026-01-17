import asyncio
from collections import deque
from typing import Deque, Dict, List, Tuple


class EventHub:
    def __init__(self, max_history: int = 200):
        self._history: Deque[Dict] = deque(maxlen=max_history)
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def publish(self, event: Dict) -> None:
        async with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for queue in subscribers:
            await queue.put(event)

    async def subscribe(self) -> Tuple[asyncio.Queue, List[Dict]]:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(queue)
            history = list(self._history)
        return queue, history

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

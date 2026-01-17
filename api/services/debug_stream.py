import sys
import threading
from typing import Callable, Dict, Optional, Tuple


class StreamMultiplexer:
    def __init__(self, base_stream):
        self._base_stream = base_stream
        self._lock = threading.Lock()
        self._sinks: Dict[int, Callable[[str], None]] = {}

    def register(self, sink: Callable[[str], None]) -> None:
        thread_id = threading.get_ident()
        with self._lock:
            self._sinks[thread_id] = sink

    def unregister(self) -> None:
        thread_id = threading.get_ident()
        with self._lock:
            self._sinks.pop(thread_id, None)

    def write(self, data: str) -> int:
        if not data:
            return 0
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        thread_id = threading.get_ident()
        sink = None
        with self._lock:
            sink = self._sinks.get(thread_id)
        if sink:
            sink(data)
        return self._base_stream.write(data)

    def flush(self) -> None:
        self._base_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._base_stream, "isatty", lambda: False)())


_stdout_mux: Optional[StreamMultiplexer] = None
_stderr_mux: Optional[StreamMultiplexer] = None
_install_lock = threading.Lock()


def ensure_debug_streams() -> Tuple[StreamMultiplexer, StreamMultiplexer]:
    global _stdout_mux, _stderr_mux
    with _install_lock:
        if _stdout_mux is None or _stderr_mux is None:
            _stdout_mux = StreamMultiplexer(sys.stdout)
            _stderr_mux = StreamMultiplexer(sys.stderr)
            sys.stdout = _stdout_mux
            sys.stderr = _stderr_mux
    return _stdout_mux, _stderr_mux


class DebugEmitter:
    def __init__(self, emit_fn: Callable[[str], None]):
        self._emit_fn = emit_fn
        self._buffer = ""

    def write(self, data: str) -> None:
        if not data:
            return
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self._emit_fn(line)

    def flush(self) -> None:
        if self._buffer:
            line = self._buffer.rstrip("\r")
            if line:
                self._emit_fn(line)
            self._buffer = ""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter()


@router.get("/sse", response_class=HTMLResponse)
async def sse_debug_page():
    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>SSE Debug</title>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; margin: 16px; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .pill { background: #f2f2f2; border-radius: 999px; padding: 4px 10px; font-size: 12px; }
    #log { white-space: pre-wrap; border: 1px solid #ddd; padding: 12px; height: 70vh; overflow: auto; background: #fafafa; }
    button { padding: 6px 10px; }
  </style>
</head>
<body>
  <div class=\"row\">
    <div class=\"pill\">SSE Debug</div>
    <div id=\"status\">Status: idle</div>
    <div id=\"task\">Task: -</div>
    <button id=\"stop\">Stop</button>
  </div>
  <div id=\"log\"></div>

  <script>
    const statusEl = document.getElementById('status');
    const taskEl = document.getElementById('task');
    const logEl = document.getElementById('log');
    const stopBtn = document.getElementById('stop');

    let currentTaskId = null;
    let es = null;
    let polling = true;

    function log(line) {
      const ts = new Date().toISOString();
      logEl.textContent += `[${ts}] ${line}\n`;
      logEl.scrollTop = logEl.scrollHeight;
    }

    function setStatus(text) {
      statusEl.textContent = `Status: ${text}`;
    }

    function connect(taskId) {
      if (!taskId) return;
      if (es) { es.close(); }
      currentTaskId = taskId;
      taskEl.textContent = `Task: ${taskId}`;
      setStatus('connecting');
      const url = `/api/research/${taskId}/stream`;
      es = new EventSource(url);
      es.onopen = () => setStatus('connected');
      es.onerror = () => setStatus('error');

      const eventNames = [
        'task_start', 'rollout_start', 'rollout_complete',
        'round_start', 'round_thinking', 'round_acting',
        'round_observing', 'round_complete', 'memory_update',
        'supervisor_event', 'progress_update', 'task_complete',
        'task_error', 'debug_log'
      ];

      eventNames.forEach((name) => {
        es.addEventListener(name, (evt) => {
          let payload = evt.data;
          try {
            payload = JSON.stringify(JSON.parse(evt.data));
          } catch (e) {
            // keep raw data
          }
          log(`${name} ${payload}`);
        });
      });
    }

    async function pollLatestTask() {
      if (!polling) return;
      try {
        const res = await fetch('/api/research?limit=1');
        if (!res.ok) return;
        const data = await res.json();
        const latest = data.items && data.items[0];
        if (latest && latest.task_id && latest.task_id !== currentTaskId) {
          log(`new task detected: ${latest.task_id}`);
          connect(latest.task_id);
        } else if (!latest) {
          setStatus('waiting for task');
        }
      } catch (e) {
        setStatus('poll error');
      }
    }

    stopBtn.onclick = () => {
      polling = false;
      if (es) { es.close(); }
      setStatus('stopped');
    };

    setInterval(pollLatestTask, 1000);
    pollLatestTask();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)

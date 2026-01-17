import { useEffect, useRef, useCallback } from 'react';
import { useResearchStore } from '../stores/researchStore';

export function useSSE(taskId) {
  const eventSourceRef = useRef(null);
  const {
    setStatus,
    updateProgress,
    addRollout,
    updateRollout,
    addRound,
    updateRound,
    addMemoryUnit,
    updateTokenUsage,
    addSupervisorLog,
    addSource,
    setResult,
    setError,
  } = useResearchStore();
  
  // Safe JSON parse helper
  const safeParse = (data) => {
    try {
      if (typeof data === 'object') return data;
      return JSON.parse(data);
    } catch (e) {
      console.warn('SSE Parse Error:', e, data);
      return {};
    }
  };

  const connect = useCallback(() => {
    if (!taskId) return;
    
    // Close existing connection if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const apiUrl = import.meta.env.VITE_API_URL || '';
    const url = `${apiUrl}/api/research/${taskId}/stream`;
    
    console.log(`Connecting to SSE: ${url}`);
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;
    
    eventSource.onopen = () => {
      console.log('SSE connection opened');
    };

    eventSource.addEventListener('task_start', (e) => {
      console.log('Event: task_start', e.data);
      setStatus('running');
    });
    
    eventSource.addEventListener('rollout_start', (e) => {
      const data = safeParse(e.data);
      addRollout({
        id: data.rollout_id,
        status: 'running',
        rounds: [],
        startedAt: data.started_at,
      });
    });
    
    eventSource.addEventListener('rollout_complete', (e) => {
      const data = safeParse(e.data);
      updateRollout(data.rollout_id, {
        status: 'completed',
        duration: data.duration_seconds,
      });
    });
    
    eventSource.addEventListener('round_start', (e) => {
      const data = safeParse(e.data);
      addRound(data.rollout_id, {
        index: data.round,
        status: 'pending',
        startedAt: data.started_at,
      });
    });
    
    eventSource.addEventListener('round_thinking', (e) => {
      const data = safeParse(e.data);
      updateRound(data.rollout_id, data.round - 1, {
        status: 'thinking',
        thinkContent: data.content,
      });
    });
    
    eventSource.addEventListener('round_acting', (e) => {
      const data = safeParse(e.data);
      updateRound(data.rollout_id, data.round - 1, {
        status: 'acting',
        tool: data.tool,
        toolArgs: data.arguments,
      });
      
      // Capture visit URLs
      if (data.tool === 'visit' && data.arguments?.url) {
        addSource({ title: 'Visited Page', url: data.arguments.url });
      }
      if (data.tool === 'visit' && Array.isArray(data.arguments?.urls)) {
        data.arguments.urls.forEach(url => addSource({ title: 'Visited Page', url }));
      }
    });
    
    eventSource.addEventListener('round_observing', (e) => {
      const data = safeParse(e.data);
      updateRound(data.rollout_id, data.round - 1, {
        status: 'observing',
        resultSummary: data.result_summary,
        resultPreview: data.result_preview,
      });
      
      // Parse search results for sources
      // Search output format: "1. [Title](url)..."
      if (data.resultPreview) {
        const linkRegex = /\[(.*?)\]\((https?:\/\/[^\s\)]+)\)/g;
        let match;
        while ((match = linkRegex.exec(data.resultPreview)) !== null) {
          addSource({ title: match[1], url: match[2] });
        }
      }
    });
    
    eventSource.addEventListener('round_complete', (e) => {
      const data = safeParse(e.data);
      updateRound(data.rollout_id, data.round - 1, {
        status: data.status,
        duration: data.duration_ms,
      });
    });
    
    eventSource.addEventListener('memory_update', (e) => {
      const data = safeParse(e.data);
      if (data.memory_unit) {
        addMemoryUnit({
          ...data.memory_unit,
          folded: data.action === 'fold',
        });
      }
      if (typeof data.tokens_before === 'number' && typeof data.tokens_after === 'number') {
        updateTokenUsage(data);
      }
    });
    
    eventSource.addEventListener('supervisor_event', (e) => {
      const data = safeParse(e.data);
      addSupervisorLog({
        timestamp: new Date().toISOString(),
        ...data,
      });
    });
    
    eventSource.addEventListener('progress_update', (e) => {
      const data = safeParse(e.data);
      updateProgress(data);
    });
    
    eventSource.addEventListener('task_complete', (e) => {
      const data = safeParse(e.data);
      setResult(data);
      eventSource.close();
      eventSourceRef.current = null;
    });
    
    eventSource.addEventListener('task_error', (e) => {
      const data = safeParse(e.data);
      setError(data.message);
      eventSource.close();
      eventSourceRef.current = null;
    });
    
    eventSource.onerror = (e) => {
      console.error('SSE connection error', e);
      // Depending on the error, we might want to close or retry
      // For now, let's close on error to avoid infinite loops in dev
      if (eventSource.readyState === EventSource.CLOSED) {
          eventSource.close();
          eventSourceRef.current = null;
      }
    };
    
  }, [taskId, setStatus, updateProgress, addRollout, updateRollout, addRound, updateRound, addMemoryUnit, updateTokenUsage, addSupervisorLog, setResult, setError]);
  
  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);
  
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);
  
  return { connect, disconnect };
}

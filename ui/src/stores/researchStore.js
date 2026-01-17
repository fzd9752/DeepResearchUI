import { create } from 'zustand';

const initialState = {
  // 任务状态
  taskId: null,
  status: 'idle', // idle | pending | running | completed | failed | cancelled
  question: '',
  error: null,
  
  // 进度
  overallProgress: 0,
  elapsedSeconds: 0,
  estimatedRemainingSeconds: null,
  llmCalls: { current: 0, max: 75 },
  
  // Rollouts
  rollouts: [],
  currentRolloutId: 1,
  
  // 记忆
  memoryUnits: [],
  tokenUsage: { current: 0, max: 110000, saved: 0 },
  
  // Sources (New)
  sources: [],
  
  // Supervisor
  supervisorLogs: [],
  
  // 结果
  result: null,
  
  // 配置
  options: {
    model: 'google/gemini-3-pro-preview',
    memoryModel: 'google/gemini-3-pro-preview',
    summaryModel: 'google/gemini-3-pro-preview',
    rolloutCount: 3,
    enableMemory: true,
    enableSupervisor: true,
    enableBrowser: true,
  },
  
  // Available models list (dynamic)
  availableModels: [
    'deepseek-ai/DeepSeek-V3.2',
    'google/gemini-3-pro-preview'
  ],
};

export const useResearchStore = create((set, get) => ({
  ...initialState,
  
  // Actions
  startResearch: (question, scenarioId = null) => {
    set({
      ...initialState,
      options: get().options, // Preserve current options
      availableModels: get().availableModels, // Preserve available models
      status: 'pending',
      question,
    });
  },
  
  setTaskId: (taskId) => set({ taskId }),
  
  setStatus: (status) => set({ status }),
  
  updateProgress: (data) => {
    set({
      overallProgress: data.overall_progress,
      elapsedSeconds: data.elapsed_seconds,
      estimatedRemainingSeconds: data.estimated_remaining_seconds,
      llmCalls: data.llm_calls,
    });
  },
  
  addRollout: (rollout) => {
    set((state) => ({
      rollouts: [...state.rollouts, rollout],
    }));
  },
  
  updateRollout: (rolloutId, updates) => {
    set((state) => ({
      rollouts: state.rollouts.map((r) =>
        r.id === rolloutId ? { ...r, ...updates } : r
      ),
    }));
  },
  
  addRound: (rolloutId, round) => {
    set((state) => ({
      rollouts: state.rollouts.map((r) =>
        r.id === rolloutId
          ? { ...r, rounds: [...(r.rounds || []), round] }
          : r
      ),
    }));
  },
  
  updateRound: (rolloutId, roundIndex, updates) => {
    set((state) => ({
      rollouts: state.rollouts.map((r) =>
        r.id === rolloutId
          ? {
              ...r,
              rounds: r.rounds.map((round, idx) =>
                idx === roundIndex ? { ...round, ...updates } : round
              ),
            }
          : r
      ),
    }));
  },
  
  setCurrentRollout: (rolloutId) => set({ currentRolloutId: rolloutId }),
  
  addMemoryUnit: (unit) => {
    set((state) => ({
      memoryUnits: [...state.memoryUnits, unit],
    }));
  },
  
  updateMemoryUnit: (index, updates) => {
    set((state) => ({
      memoryUnits: state.memoryUnits.map((u, idx) =>
        idx === index ? { ...u, ...updates } : u
      ),
    }));
  },
  
  updateTokenUsage: (data) => {
    set({
      tokenUsage: {
        current: data.tokens_after,
        max: 110000,
        saved: data.tokens_before - data.tokens_after,
      },
    });
  },
  
  addSource: (source) => {
    set((state) => {
      // Avoid duplicates based on URL
      if (state.sources.some(s => s.url === source.url)) {
        return state;
      }
      return { sources: [...state.sources, source] };
    });
  },

  addSupervisorLog: (log) => {
    set((state) => ({
      supervisorLogs: [log, ...state.supervisorLogs].slice(0, 50),
    }));
  },
  
  setResult: (result) => set({ 
    result, 
    status: 'completed',
    overallProgress: 1.0 
  }),
  
  setError: (error) => set({ error, status: 'failed' }),
  
  updateOptions: (options) => {
    set((state) => ({
      options: { ...state.options, ...options },
    }));
  },
  
  setAvailableModels: (models) => {
    set({ availableModels: models });
  },
  
  reset: () => set(initialState),
}));
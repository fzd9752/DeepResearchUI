export const TOOL_NAMES = {
  SEARCH: 'search',
  VISIT: 'visit',
  SCHOLAR: 'google_scholar',
  CODE: 'CodeExecutor',
};

export const STATUS = {
  IDLE: 'idle',
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
};

// Initial default models, will be overwritten by API config
export const LLM_MODELS = [
  'deepseek-ai/DeepSeek-V3.2',
  'google/gemini-3-pro-preview'
];
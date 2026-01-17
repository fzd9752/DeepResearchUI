import { useCallback } from 'react';
import { useResearchStore } from '../stores/researchStore';
import { useSSE } from './useSSE';
import { createResearch, cancelResearch, SCENARIOS_DATA } from '../utils/api';

export function useResearch() {
  const {
    taskId,
    status,
    question,
    startResearch,
    setTaskId,
    setStatus,
    reset,
    options,
  } = useResearchStore();
  
  useSSE(taskId);
  
  const start = useCallback(async (userInput, scenarioId = null, files = []) => {
    try {
      // Logic to wrap the prompt on the frontend so it's visible in the UI
      let fullQuestion = userInput;
      
      if (scenarioId) {
        const scenario = SCENARIOS_DATA.find(s => s.id === scenarioId);
        if (scenario && scenario.prompt_template) {
          fullQuestion = scenario.prompt_template.replace('{user_input}', userInput);
        }
      }

      // Update store with the FULL question so FlowGraph shows it
      startResearch(fullQuestion, scenarioId);
      
      // Send the FULL question to backend
      // We set scenario_id to null because we already applied the template
      // This prevents the backend from wrapping it again
      const response = await createResearch({
        question: fullQuestion,
        scenario_id: null, // intentionally null
        files: files && files.length ? files : undefined,
        options,
      });
      
      setTaskId(response.task_id);
    } catch (error) {
      console.error('Failed to start research:', error);
      setStatus('failed');
    }
  }, [options, startResearch, setTaskId, setStatus]);
  
  const cancel = useCallback(async () => {
    if (!taskId) return;
    
    try {
      await cancelResearch(taskId);
      setStatus('cancelled');
    } catch (error) {
      console.error('Failed to cancel research:', error);
    }
  }, [taskId, setStatus]);
  
  return {
    taskId,
    status,
    question,
    start,
    cancel,
    reset,
  };
}

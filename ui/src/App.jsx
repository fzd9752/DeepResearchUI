import { useEffect } from 'react';
import { useResearch } from './hooks/useResearch';
import { useResearchStore } from './stores/researchStore';
import { getConfig } from './utils/api';
import Header from './components/Header/Header';
import InputPanel from './components/InputPanel/InputPanel';
import ProgressBar from './components/ProgressBar/ProgressBar';
import FlowGraph from './components/FlowGraph/FlowGraph';
import RoundDetails from './components/RoundDetails/RoundDetails';
import MemoryPanel from './components/MemoryPanel/MemoryPanel';
import SupervisorLog from './components/SupervisorLog/SupervisorLog';
import ReportPanel from './components/ReportPanel/ReportPanel';
import StatusBar from './components/StatusBar/StatusBar';
import styles from './App.module.css';

function App() {
  const { start, cancel } = useResearch();
  const { updateOptions, setAvailableModels } = useResearchStore();

  useEffect(() => {
    // Fetch configuration from backend on startup
    const fetchConfig = async () => {
      try {
        const config = await getConfig();
        if (config) {
          if (config.available_models) {
            setAvailableModels(config.available_models);
          }
          
          updateOptions({
            model: config.default_model,
            memoryModel: config.default_memory_model,
            summaryModel: config.default_summary_model,
            enableMemory: config.features?.memory_management ?? true,
            enableSupervisor: config.features?.supervisor ?? true,
            enableBrowser: config.features?.browser_agent ?? true,
          });
        }
      } catch (error) {
        console.error("Failed to load config:", error);
      }
    };
    
    fetchConfig();
  }, [updateOptions, setAvailableModels]);

  const handleStart = (question, scenarioId, files) => {
    start(question, scenarioId, files);
  };

  return (
    <div className={styles.container}>
      <div className={styles.wrapper}>
        <Header />
        
        <InputPanel onSubmit={handleStart} onStop={cancel} />
        
        <ProgressBar />
        
        <FlowGraph />
        
        <RoundDetails />
        
        <div className={styles.grid2}>
          <MemoryPanel />
          <SupervisorLog />
        </div>
        
        <ReportPanel />
        
        <StatusBar />
      </div>
    </div>
  );
}

export default App;

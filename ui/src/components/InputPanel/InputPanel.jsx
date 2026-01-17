import { useState, useEffect } from 'react';
import { useResearchStore } from '../../stores/researchStore';
import { useScenarios } from '../../hooks/useScenarios';
import ScenarioCard from './ScenarioCard';
import AdvancedOptions from './AdvancedOptions';
import styles from './InputPanel.module.css';

export default function InputPanel({ onSubmit, onStop }) {
  const [question, setQuestion] = useState('');
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showConfig, setShowConfig] = useState(true);
  
  const { scenarios, loading } = useScenarios();
  const { options, updateOptions, status } = useResearchStore();

  const isDisabled = status === 'running' || status === 'pending';
  
  // Auto-collapse when running
  useEffect(() => {
    if (isDisabled) {
      setShowConfig(false);
    } else {
      setShowConfig(true);
    }
  }, [isDisabled]);
  
  const handleSubmit = () => {
    if (!question.trim()) return;
    onSubmit(question, selectedScenario);
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit();
    }
  };
  
  // Find the selected scenario object
  const activeScenario = scenarios.find(s => s.id === selectedScenario);
  const placeholderText = activeScenario 
    ? `[${activeScenario.name}] ${activeScenario.description}\n请输入具体内容...`
    : "请输入你的研究问题，按 Ctrl+Enter 开始...";
  
  return (
    <div className={styles.panel}>
      <div className={styles.title}>
        <span>─ 研究输入</span>
        
        {isDisabled && (
          <button
            className={styles.toggleBtn}
            onClick={onStop}
            style={{ marginLeft: '16px', color: 'var(--accent-red)' }}
          >
            [■ 停止任务]
          </button>
        )}

        {isDisabled && (
          <button 
            className={styles.toggleBtn} 
            onClick={() => setShowConfig(!showConfig)}
            style={{ marginLeft: 'auto' }}
          >
            {showConfig ? '[收起配置]' : '[展开配置]'}
          </button>
        )}
      </div>
      
      <div className={styles.inputWrapper}>
        <span className={styles.prompt}>&gt;</span>
        <textarea
          className={styles.input}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholderText}
          disabled={isDisabled}
          rows={isDisabled && !showConfig ? 1 : 2}
          style={isDisabled && !showConfig ? { minHeight: '30px', resize: 'none' } : {}}
        />
      </div>
      
      {showConfig && (
        <>
          <div className={styles.scenariosSection}>
            <div className={styles.sectionTitle}>─ 预设场景</div>
            <div className={styles.scenarioGrid}>
              {loading ? (
                 <div style={{ color: 'var(--text-muted)' }}>加载场景中...</div>
              ) : (
                scenarios.map((scenario) => (
                  <ScenarioCard
                    key={scenario.id}
                    scenario={scenario}
                    selected={selectedScenario === scenario.id}
                    onClick={() => setSelectedScenario(
                      selectedScenario === scenario.id ? null : scenario.id
                    )}
                    disabled={isDisabled}
                  />
                ))
              )}
            </div>
          </div>
          
          <div className={styles.advancedToggle}>
            <button
              className={styles.toggleBtn}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              ─ 高级选项 {showAdvanced ? '▼' : '▶'}
            </button>
          </div>
          
          {showAdvanced && (
            <AdvancedOptions
              options={options}
              onChange={updateOptions}
              disabled={isDisabled}
            />
          )}
        </>
      )}
      
      {(!isDisabled || showConfig) && (
        <div className={styles.actions}>
          <button
            className={styles.submitBtn}
            onClick={handleSubmit}
            disabled={isDisabled || !question.trim()}
          >
            {isDisabled ? '● 研究中...' : '▶ 开始深度研究'}
          </button>
        </div>
      )}
    </div>
  );
}

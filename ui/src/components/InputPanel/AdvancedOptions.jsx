import { useRef, useState } from 'react';
import { useResearchStore } from '../../stores/researchStore';
import { updateConfig } from '../../utils/api';

const containerStyle = {
  border: '1px dashed var(--border-default)',
  padding: 'var(--spacing-md)',
  marginBottom: 'var(--spacing-lg)',
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
  gap: 'var(--spacing-lg)',
};

const fieldStyle = {
  display: 'flex',
  flexDirection: 'column',
  gap: 'var(--spacing-sm)',
};

const labelStyle = {
  fontSize: 'var(--font-size-sm)',
  color: 'var(--text-secondary)',
};

const selectStyle = {
  background: 'var(--bg-tertiary)',
  border: '1px solid var(--border-default)',
  color: 'var(--text-primary)',
  padding: '4px 8px',
  fontFamily: 'var(--font-mono)',
  width: '100%',
};

const checkboxContainerStyle = {
  display: 'flex',
  flexDirection: 'column',
  gap: '8px',
};

const checkboxLabelStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  cursor: 'pointer',
  fontSize: 'var(--font-size-sm)',
};

const statusStyle = {
  fontSize: '12px',
  color: 'var(--text-muted)',
  gridColumn: '1 / -1',
};

export default function AdvancedOptions({ options, onChange, disabled }) {
  const { availableModels } = useResearchStore();
  const [saveStatus, setSaveStatus] = useState(null);
  const saveTokenRef = useRef(0);
  
  const handleChange = (key, value) => {
    const nextOptions = { ...options, [key]: value };
    onChange({ [key]: value });
    if (key === 'model' || key === 'memoryModel' || key === 'summaryModel') {
      const token = ++saveTokenRef.current;
      setSaveStatus('saving');
      updateConfig({
        default_model: nextOptions.model,
        memory_model: nextOptions.memoryModel,
        summary_model: nextOptions.summaryModel,
      })
        .then(() => {
          if (saveTokenRef.current === token) {
            setSaveStatus('saved');
            setTimeout(() => {
              if (saveTokenRef.current === token) {
                setSaveStatus(null);
              }
            }, 2000);
          }
        })
        .catch((error) => {
          console.error('Failed to update config:', error);
          if (saveTokenRef.current === token) {
            setSaveStatus('failed');
            setTimeout(() => {
              if (saveTokenRef.current === token) {
                setSaveStatus(null);
              }
            }, 4000);
          }
        });
    }
  };

  return (
    <div style={containerStyle}>
      <div style={fieldStyle}>
        <label style={labelStyle}>主模型 (Main Agent)</label>
        <select
          style={selectStyle}
          value={options.model}
          onChange={(e) => handleChange('model', e.target.value)}
          disabled={disabled}
        >
          {availableModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>

      <div style={fieldStyle}>
        <label style={labelStyle}>记忆模型 (Memory)</label>
        <select
          style={selectStyle}
          value={options.memoryModel}
          onChange={(e) => handleChange('memoryModel', e.target.value)}
          disabled={disabled}
        >
          {availableModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>
      
      <div style={fieldStyle}>
        <label style={labelStyle}>摘要模型 (Summary)</label>
        <select
          style={selectStyle}
          value={options.summaryModel}
          onChange={(e) => handleChange('summaryModel', e.target.value)}
          disabled={disabled}
        >
          {availableModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>

      <div style={fieldStyle}>
        <label style={labelStyle}>Rollout 数量</label>
        <select
          style={selectStyle}
          value={options.rolloutCount}
          onChange={(e) => handleChange('rolloutCount', Number(e.target.value))}
          disabled={disabled}
        >
          {[1, 3, 5, 10].map((num) => (
            <option key={num} value={num}>
              {num}
            </option>
          ))}
        </select>
      </div>

      <div style={checkboxContainerStyle}>
        <label style={checkboxLabelStyle}>
          <input
            type="checkbox"
            checked={options.enableMemory}
            onChange={(e) => handleChange('enableMemory', e.target.checked)}
            disabled={disabled}
          />
          启用记忆管理
        </label>
        <label style={checkboxLabelStyle}>
          <input
            type="checkbox"
            checked={options.enableSupervisor}
            onChange={(e) => handleChange('enableSupervisor', e.target.checked)}
            disabled={disabled}
          />
          启用 Supervisor
        </label>
        <label style={checkboxLabelStyle}>
          <input
            type="checkbox"
            checked={options.enableBrowser}
            onChange={(e) => handleChange('enableBrowser', e.target.checked)}
            disabled={disabled}
          />
          启用浏览器 Agent
        </label>
      </div>

      {saveStatus && (
        <div style={statusStyle}>
          {saveStatus === 'saving' && '模型配置保存中...'}
          {saveStatus === 'saved' && '模型配置已保存到 .env'}
          {saveStatus === 'failed' && '模型配置保存失败，请检查后端权限或网络'}
        </div>
      )}
    </div>
  );
}

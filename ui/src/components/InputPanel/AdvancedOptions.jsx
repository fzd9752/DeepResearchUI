import { useResearchStore } from '../../stores/researchStore';

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

export default function AdvancedOptions({ options, onChange, disabled }) {
  const { availableModels } = useResearchStore();
  
  const handleChange = (key, value) => {
    onChange({ [key]: value });
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
    </div>
  );
}
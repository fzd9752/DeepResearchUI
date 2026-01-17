import styles from './InputPanel.module.css';

// Using consistent non-shorthand properties to avoid React warnings
const baseStyle = {
  borderWidth: '1px',
  borderStyle: 'solid',
  borderColor: 'var(--border-default)',
  padding: 'var(--spacing-sm)',
  cursor: 'pointer',
  transition: 'all var(--transition-fast)',
  display: 'flex',
  flexDirection: 'column',
  gap: '4px',
  backgroundColor: 'var(--bg-secondary)',
  borderRadius: '4px',
};

const selectedCardStyle = {
  ...baseStyle,
  borderColor: 'var(--accent-green)',
  backgroundColor: 'var(--bg-tertiary)',
};

export default function ScenarioCard({ scenario, selected, onClick, disabled }) {
  const currentStyle = selected ? selectedCardStyle : baseStyle;
  
  return (
    <div 
      style={{
        ...currentStyle,
        opacity: disabled ? 0.5 : 1,
        pointerEvents: disabled ? 'none' : 'auto',
      }} 
      onClick={onClick}
    >
      <div style={{ fontSize: '20px', marginBottom: '4px' }}>{scenario.icon}</div>
      <div style={{ fontWeight: 'bold', fontSize: 'var(--font-size-base)', color: selected ? 'var(--accent-green)' : 'var(--text-primary)' }}>
        {scenario.name}
      </div>
      <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-muted)' }}>
        {scenario.description}
      </div>
    </div>
  );
}
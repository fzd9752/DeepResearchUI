export default function RolloutTabs({ rollouts, currentId, onChange }) {
  const containerStyle = {
    display: 'flex',
    gap: '12px',
    marginBottom: '16px',
  };

  return (
    <div style={containerStyle}>
      {rollouts.map((rollout) => {
        const isActive = rollout.id === currentId;
        const statusIcon = rollout.status === 'completed' ? '✓' : rollout.status === 'failed' ? '✗' : '●';
        const color = isActive ? 'var(--accent-green)' : 'var(--text-secondary)';
        
        return (
          <button
            key={rollout.id}
            onClick={() => onChange(rollout.id)}
            style={{
              background: 'none',
              border: 'none',
              color: color,
              fontFamily: 'var(--font-mono)',
              fontSize: '14px',
              cursor: 'pointer',
              padding: '4px 8px',
              borderBottom: isActive ? `2px solid ${color}` : '2px solid transparent',
              transition: 'all 0.2s',
            }}
          >
            [Rollout {rollout.id} {statusIcon}]
          </button>
        );
      })}
      
      {rollouts.length === 0 && (
         <div style={{ color: 'var(--text-muted)', fontSize: '14px' }}>等待开始...</div>
      )}
    </div>
  );
}

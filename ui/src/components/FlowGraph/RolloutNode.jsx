import { Handle, Position } from 'reactflow';
import { getStatusColor, formatDuration } from '../../utils/formatters';

export default function RolloutNode({ data }) {
  const statusColor = getStatusColor(data.status);
  
  const nodeStyle = {
    background: 'var(--bg-secondary)',
    border: `2px solid ${statusColor}`,
    borderRadius: '4px',
    padding: '10px',
    minWidth: '160px',
    textAlign: 'center',
    color: 'var(--text-primary)',
    fontSize: '12px',
    boxShadow: data.status === 'running' ? `0 0 10px ${statusColor}40` : 'none',
  };

  return (
    <div style={nodeStyle}>
      <Handle type="target" position={Position.Top} style={{ background: '#555' }} />
      
      <div style={{ fontWeight: 'bold', marginBottom: '4px', color: statusColor }}>
        Rollout {data.id}
      </div>
      
      <div style={{ marginBottom: '4px' }}>
        {data.status === 'running' ? '● 进行中' : 
         data.status === 'completed' ? '✓ 已完成' : 
         data.status === 'failed' ? '✗ 失败' : '等待中'}
      </div>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)' }}>
        <span>Round {data.rounds?.length || 0}</span>
        <span>⏱ {formatDuration(data.duration)}</span>
      </div>
      
      <Handle type="source" position={Position.Bottom} style={{ background: '#555' }} />
    </div>
  );
}

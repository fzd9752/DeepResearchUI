import { Handle, Position } from 'reactflow';
import { getStatusColor } from '../../utils/formatters';

export default function AnswerNode({ data }) {
  const statusColor = getStatusColor(data.status);
  
  const nodeStyle = {
    background: 'var(--bg-tertiary)',
    border: `2px solid ${statusColor}`,
    borderRadius: '4px',
    padding: '12px',
    minWidth: '180px',
    textAlign: 'center',
    color: 'var(--text-primary)',
    fontSize: '12px',
  };

  return (
    <div style={nodeStyle}>
      <Handle type="target" position={Position.Top} style={{ background: '#555' }} />
      
      <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
        {data.status === 'completed' ? '✅ 最终答案' : '🔄 答案综合'}
      </div>
      
      <div style={{ color: 'var(--text-secondary)' }}>
        {data.status === 'completed' ? '点击查看报告' : '等待 Rollout 完成...'}
      </div>
    </div>
  );
}

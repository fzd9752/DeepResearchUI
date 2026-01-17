import { Handle, Position } from 'reactflow';

const nodeStyle = {
  background: 'var(--bg-tertiary)',
  border: '1px solid var(--border-default)',
  borderRadius: '4px',
  padding: '10px',
  minWidth: '150px',
  textAlign: 'center',
  color: 'var(--text-primary)',
  fontSize: '12px',
};

export default function QuestionNode({ data }) {
  return (
    <div style={nodeStyle}>
      <div style={{ marginBottom: '4px' }}>📝 用户问题</div>
      <div style={{ color: 'var(--text-secondary)', fontStyle: 'italic' }}>
        "{data.question?.slice(0, 20)}{data.question?.length > 20 ? '...' : ''}"
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: '#555' }} />
    </div>
  );
}

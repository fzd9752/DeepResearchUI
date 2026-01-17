import { getStatusColor } from '../../utils/formatters';

export default function RolloutProgress({ rollout }) {
  const progress = rollout.rounds ? (rollout.rounds.length / 10) * 100 : 0; // Assuming max 10 rounds for vis
  const cappedProgress = Math.min(100, progress);
  const filledBlocks = Math.floor(cappedProgress / 4); // 25 blocks total
  const emptyBlocks = 25 - filledBlocks;
  
  const statusColor = getStatusColor(rollout.status);
  
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '12px' }}>
      <span style={{ width: '80px', color: 'var(--text-secondary)' }}>Rollout {rollout.id}</span>
      
      <div style={{ flex: 1, display: 'flex' }}>
        <span style={{ color: statusColor }}>
          {Array.from({ length: filledBlocks }).map(() => '█').join('')}
        </span>
        <span style={{ color: '#333' }}>
          {Array.from({ length: emptyBlocks }).map(() => '░').join('')}
        </span>
      </div>
      
      <span style={{ width: '100px', textAlign: 'right', color: statusColor }}>
        {rollout.status === 'running' ? '● 进行中' : 
         rollout.status === 'completed' ? '✓ 已完成' : 
         rollout.status === 'failed' ? '✗ 失败' : '等待中'}
      </span>
      
      <span style={{ width: '80px', textAlign: 'right', color: 'var(--text-muted)' }}>
        {rollout.rounds?.length || 0} Rounds
      </span>
    </div>
  );
}

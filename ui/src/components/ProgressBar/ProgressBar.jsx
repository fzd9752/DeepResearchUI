import { useResearchStore } from '../../stores/researchStore';
import RolloutProgress from './RolloutProgress';
import { formatDuration } from '../../utils/formatters';
import styles from './ProgressBar.module.css';

export default function ProgressBar() {
  const {
    status,
    overallProgress,
    elapsedSeconds,
    estimatedRemainingSeconds,
    llmCalls,
    rollouts,
  } = useResearchStore();
  
  if (status === 'idle') return null;
  
  const progressPercent = Math.min(100, Math.max(0, Math.round(overallProgress * 100)));
  const filledBlocks = Math.floor(progressPercent / 2); // 50 blocks total for 100%
  const emptyBlocks = 50 - filledBlocks;
  
  return (
    <div className={styles.panel}>
      <div className={styles.title}>─ 研究进度</div>
      
      <div className={styles.mainProgress}>
        <div className={styles.label}>整体进度</div>
        <div className={styles.barContainer}>
          <span style={{ color: 'var(--accent-green)' }}>
            {Array.from({ length: filledBlocks }).map((_, i) => '█').join('')}
          </span>
          <span style={{ color: '#333' }}>
            {Array.from({ length: emptyBlocks }).map((_, i) => '░').join('')}
          </span>
          <span className={styles.percent}>{progressPercent}%</span>
        </div>
      </div>
      
      <div className={styles.stats}>
        <div className={styles.stat}>
          <span className={styles.statIcon}>⏱</span>
          <span>已用时间: {formatDuration(elapsedSeconds)}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statIcon}>📊</span>
          <span>预计剩余: ~{formatDuration(estimatedRemainingSeconds)}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statIcon}>🔄</span>
          <span>LLM 调用: {llmCalls.current}/{llmCalls.max}</span>
        </div>
      </div>
      
      <div className={styles.rollouts}>
        {rollouts.map((rollout) => (
          <RolloutProgress key={rollout.id} rollout={rollout} />
        ))}
      </div>
    </div>
  );
}

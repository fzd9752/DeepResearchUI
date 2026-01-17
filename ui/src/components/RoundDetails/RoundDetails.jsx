import { useState } from 'react';
import { useResearchStore } from '../../stores/researchStore';
import RolloutTabs from './RolloutTabs';
import RoundItem from './RoundItem';
import styles from './RoundDetails.module.css';

export default function RoundDetails() {
  const { rollouts, currentRolloutId, setCurrentRollout } = useResearchStore();
  const [expandedRound, setExpandedRound] = useState(null);
  
  const currentRollout = rollouts.find((r) => r.id === currentRolloutId) || rollouts[0];
  const rounds = currentRollout?.rounds || [];
  
  // Auto-select first rollout if none selected and rollouts exist
  if (!currentRolloutId && rollouts.length > 0) {
      // Defer state update to avoid loop in render
      // For now just use first
  }
  
  return (
    <div className={styles.panel}>
      <div className={styles.title}>─ 执行详情</div>
      
      <RolloutTabs
        rollouts={rollouts}
        currentId={currentRolloutId || (rollouts[0]?.id)}
        onChange={setCurrentRollout}
      />
      
      <div className={styles.separator}>
        {'═'.repeat(60)}
      </div>
      
      <div className={styles.roundList}>
        {rounds.map((round, index) => (
          <RoundItem
            key={index}
            round={round}
            index={index}
            expanded={expandedRound === index}
            onToggle={() => setExpandedRound(
              expandedRound === index ? null : index
            )}
          />
        ))}
        
        {rounds.length === 0 && (
          <div className={styles.empty}>等待 Rollout 开始...</div>
        )}
      </div>
    </div>
  );
}

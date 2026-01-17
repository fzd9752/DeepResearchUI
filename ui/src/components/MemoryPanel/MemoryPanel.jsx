import { useState } from 'react';
import { useResearchStore } from '../../stores/researchStore';
import { formatLargeNumber } from '../../utils/formatters';
import styles from './MemoryPanel.module.css';

export default function MemoryPanel() {
  const { options, memoryUnits, tokenUsage } = useResearchStore();
  const [expandedUnits, setExpandedUnits] = useState({});
  const [showAllModal, setShowAllModal] = useState(false);
  
  if (!options.enableMemory) return null;

  const percentUsed = Math.min(100, (tokenUsage.current / tokenUsage.max) * 100);
  const truncate = (text, limit) => {
    if (!text) return '';
    return text.length > limit ? `${text.slice(0, limit)}...` : text;
  };

  const orderedUnits = memoryUnits
    .map((unit, index) => ({ unit, index }))
    .reverse();
  const latestUnits =
    memoryUnits.length > 0
      ? [{ unit: memoryUnits[memoryUnits.length - 1], index: memoryUnits.length - 1 }]
      : [];

  const renderUnit = ({ unit, index }) => {
    const isExpanded = !!expandedUnits[index];
    const subGoal = unit.sub_goal || 'Searching...';
    const summary = unit.summary || '';
    const showToggle = subGoal.length > 80 || summary.length > 120;

    return (
      <div key={index} className={styles.unit}>
        <div className={styles.unitHeader}>
          <span>📦 Sub-goal {index + 1}</span>
          <span>{unit.folded ? '[已折叠]' : '[活跃]'}</span>
        </div>
        <div className={styles.unitContent}>
          "{isExpanded ? subGoal : truncate(subGoal, 80)}"
        </div>
        {summary && (
          <div className={styles.unitSummary}>
            {isExpanded ? summary : truncate(summary, 120)}
          </div>
        )}
        {showToggle && (
          <button
            type="button"
            className={styles.toggleButton}
            onClick={() =>
              setExpandedUnits((prev) => ({
                ...prev,
                [index]: !prev[index],
              }))
            }
          >
            {isExpanded ? '收起' : '查看完整内容'}
          </button>
        )}
      </div>
    );
  };

  return (
    <div className={styles.panel}>
      <div className={styles.title}>
        <span>─ 记忆状态</span>
        <div className={styles.titleActions}>
          <span style={{ color: 'var(--accent-green)' }}>
            节省: {Math.round((tokenUsage.saved / (tokenUsage.current + tokenUsage.saved || 1)) * 100)}% ↓
          </span>
          {memoryUnits.length > 1 && (
            <button
              type="button"
              className={styles.viewAllButton}
              onClick={() => setShowAllModal(true)}
            >
              查看全部
            </button>
          )}
        </div>
      </div>
      
      <div className={styles.list}>
        {latestUnits.map(renderUnit)}
        {latestUnits.length === 0 && (
          <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '20px' }}>
            暂无记忆单元
          </div>
        )}
      </div>
      
      <div className={styles.tokenBar}>
        <div className={styles.tokenStats}>
          <span>Token 使用: {formatLargeNumber(tokenUsage.current)} / {formatLargeNumber(tokenUsage.max)}</span>
          <span>{percentUsed.toFixed(1)}%</span>
        </div>
        <div className={styles.barBg}>
          <div className={styles.barFill} style={{ width: `${percentUsed}%` }} />
        </div>
      </div>

      {showAllModal && (
        <div className={styles.overlay} onClick={() => setShowAllModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <div className={styles.modalTitle}>[全部记忆]</div>
              <button
                type="button"
                className={styles.closeBtn}
                onClick={() => setShowAllModal(false)}
              >
                ✕
              </button>
            </div>
            <div className={styles.modalContent}>
              {orderedUnits.map(renderUnit)}
              {orderedUnits.length === 0 && (
                <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '20px' }}>
                  暂无记忆单元
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

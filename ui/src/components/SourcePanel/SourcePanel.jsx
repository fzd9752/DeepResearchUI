import { useResearchStore } from '../../stores/researchStore';
import styles from './SourcePanel.module.css';

export default function SourcePanel() {
  const { sources } = useResearchStore();

  return (
    <div className={styles.panel}>
      <div className={styles.title}>
        <span>─ 📚 引用来源 ({sources.length})</span>
      </div>
      
      <div className={styles.list}>
        {sources.map((source, idx) => (
          <a 
            key={idx} 
            href={source.url} 
            target="_blank" 
            rel="noopener noreferrer" 
            className={styles.sourceItem}
          >
            <span className={styles.index}>[{idx + 1}]</span>
            <div className={styles.content}>
              <span className={styles.sourceTitle}>{source.title || 'Unknown Title'}</span>
              <span className={styles.sourceUrl}>{source.url}</span>
            </div>
          </a>
        ))}
        {sources.length === 0 && (
          <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '20px' }}>
            暂无来源...
          </div>
        )}
      </div>
    </div>
  );
}

import { useResearchStore } from '../../stores/researchStore';
import { formatTime } from '../../utils/formatters';
import styles from './SupervisorLog.module.css';

export default function SupervisorLog() {
  const { options, supervisorLogs, llmCalls } = useResearchStore();
  
  if (!options.enableSupervisor) return null;

  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(supervisorLogs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `supervisor-logs-${new Date().toISOString()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={styles.panel}>
      <div className={styles.title}>
        <span>─ Supervisor 日志</span>
        <div style={{ display: 'flex', gap: '12px' }}>
          {supervisorLogs.length > 0 && (
            <button 
              onClick={handleDownload}
              style={{ background: 'none', border: 'none', color: 'var(--accent-cyan)', cursor: 'pointer', fontSize: '12px' }}
            >
              [下载]
            </button>
          )}
          <span>LLM: {llmCalls.current}/{llmCalls.max}</span>
        </div>
      </div>
      
      <div className={styles.list}>
        {supervisorLogs.map((log, idx) => {
          let icon = 'ℹ';
          let color = 'var(--text-secondary)';
          
          if (log.type === 'correction') {
            icon = '✓';
            color = 'var(--accent-green)';
          } else if (log.type === 'warning' || log.level === 'warning') {
            icon = '⚠';
            color = 'var(--accent-yellow)';
          } else if (log.type === 'error' || log.level === 'error') {
            icon = '✗';
            color = 'var(--accent-red)';
          }
          
          return (
            <div key={idx} className={styles.logItem}>
              <span className={styles.time}>{formatTime(log.timestamp)}</span>
              <span className={styles.icon} style={{ color }}>{icon}</span>
              <span className={styles.content}>{log.message || JSON.stringify(log)}</span>
            </div>
          );
        })}
        {supervisorLogs.length === 0 && (
          <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '20px' }}>
            等待日志...
          </div>
        )}
      </div>
    </div>
  );
}

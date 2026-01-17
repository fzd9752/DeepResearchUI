import { useResearchStore } from '../../stores/researchStore';
import styles from './StatusBar.module.css';

export default function StatusBar() {
  const { options, status } = useResearchStore();
  
  return (
    <footer className={styles.statusBar}>
      <div className={styles.left}>
        <div className={styles.item}>
          Status: <span className={styles.value}>{status.toUpperCase()}</span>
        </div>
        <div className={styles.item}>
          Model: <span className={styles.value}>{options.model}</span>
        </div>
        <div className={styles.item}>
          Rollouts: <span className={styles.value}>{options.rolloutCount}</span>
        </div>
        <div className={styles.item}>
          Memory: <span className={styles.value}>{options.enableMemory ? 'ON' : 'OFF'}</span>
        </div>
        <div className={styles.item}>
          Supervisor: <span className={styles.value}>{options.enableSupervisor ? 'ON' : 'OFF'}</span>
        </div>
      </div>
      <div className={styles.right}>
        v1.0.0
      </div>
    </footer>
  );
}

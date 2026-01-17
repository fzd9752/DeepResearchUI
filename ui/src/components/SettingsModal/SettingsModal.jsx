import { useResearchStore } from '../../stores/researchStore';
import AdvancedOptions from '../InputPanel/AdvancedOptions';
import styles from './SettingsModal.module.css';

export default function SettingsModal({ onClose }) {
  const { options, updateOptions } = useResearchStore();

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <div className={styles.title}>[全局设置]</div>
          <button className={styles.closeBtn} onClick={onClose}>✕</button>
        </div>
        
        <div className={styles.content}>
          <AdvancedOptions 
            options={options} 
            onChange={updateOptions} 
          />
        </div>
        
        <div className={styles.footer}>
          <button className={styles.saveBtn} onClick={onClose}>
            确定
          </button>
        </div>
      </div>
    </div>
  );
}

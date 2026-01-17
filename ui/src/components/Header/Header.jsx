import { useState } from 'react';
import SettingsModal from '../SettingsModal/SettingsModal';
import styles from './Header.module.css';

export default function Header() {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <header className={styles.header}>
      <div className={styles.title}>
        ╔════ YUNQUE DEEPRESEARCH ════╗
      </div>
      <div className={styles.actions}>
        <button 
          className={styles.button}
          onClick={() => setShowSettings(true)}
        >
          [设置]
        </button>
        <button className={styles.button}>[帮助]</button>
      </div>

      {showSettings && (
        <SettingsModal onClose={() => setShowSettings(false)} />
      )}
    </header>
  );
}
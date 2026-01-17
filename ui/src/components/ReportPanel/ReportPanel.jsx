import Markdown from 'react-markdown';
import { useResearchStore } from '../../stores/researchStore';
import styles from './ReportPanel.module.css';

export default function ReportPanel() {
  const { result, status } = useResearchStore();
  
  if (!result && status !== 'completed') return null;

  // If status is completed but result is null (shouldn't happen but fallback)
  const content = result?.answer || result?.prediction || "研究已完成，正在生成报告...";
  
  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    alert('已复制到剪贴板');
  };

  const handleExport = () => {
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research-report-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={styles.panel}>
      <div className={styles.title}>
        <span>─ 研究报告</span>
        <div className={styles.actions}>
          <button className={styles.actionBtn} onClick={handleCopy}>[复制]</button>
          <button className={styles.actionBtn} onClick={handleExport}>[导出]</button>
        </div>
      </div>
      
      <div className={styles.reportContent}>
        <Markdown>{content}</Markdown>
      </div>
      
      {result && (
        <div className={styles.stats}>
          📊 研究统计
          <br />
          ├─ 耗时: {result.duration_seconds}s
          <br />
          ├─ LLM 调用: {result.llm_calls}
          <br />
          └─ 终止原因: {result.termination}
        </div>
      )}
    </div>
  );
}

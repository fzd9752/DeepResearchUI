import { useState, useEffect, useRef } from 'react';
import { useResearchStore } from '../../stores/researchStore';
import { useScenarios } from '../../hooks/useScenarios';
import { uploadFile } from '../../utils/api';
import ScenarioCard from './ScenarioCard';
import AdvancedOptions from './AdvancedOptions';
import styles from './InputPanel.module.css';

export default function InputPanel({ onSubmit, onStop }) {
  const [question, setQuestion] = useState('');
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showConfig, setShowConfig] = useState(true);
  const [images, setImages] = useState([]);
  const fileInputRef = useRef(null);
  const imagesRef = useRef([]);
  
  const { scenarios, loading } = useScenarios();
  const { options, updateOptions, status } = useResearchStore();

  const isDisabled = status === 'running' || status === 'pending';
  const isUploading = images.some((img) => img.status === 'uploading');
  const hasFailed = images.some((img) => img.status === 'failed');
  const uploadedFileIds = images
    .filter((img) => img.status === 'ready' && img.fileId)
    .map((img) => img.fileId);
  
  // Auto-collapse when running
  useEffect(() => {
    if (isDisabled) {
      setShowConfig(false);
    } else {
      setShowConfig(true);
    }
  }, [isDisabled]);

  useEffect(() => {
    imagesRef.current = images;
  }, [images]);

  useEffect(() => {
    return () => {
      imagesRef.current.forEach((img) => {
        if (img.previewUrl) {
          URL.revokeObjectURL(img.previewUrl);
        }
      });
    };
  }, []);
  
  const handleSubmit = () => {
    if (!question.trim()) return;
    if (isUploading) return;
    onSubmit(question, selectedScenario, uploadedFileIds);
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit();
    }
  };
  
  // Find the selected scenario object
  const activeScenario = scenarios.find(s => s.id === selectedScenario);
  const placeholderText = activeScenario 
    ? `[${activeScenario.name}] ${activeScenario.description}\n请输入具体内容...`
    : "请输入你的研究问题，按 Ctrl+Enter 开始...";

  const isImageFile = (name, mime) => {
    if (mime && mime.startsWith('image/')) return true;
    const lower = (name || '').toLowerCase();
    return (
      lower.endsWith('.png') ||
      lower.endsWith('.jpg') ||
      lower.endsWith('.jpeg') ||
      lower.endsWith('.webp') ||
      lower.endsWith('.gif') ||
      lower.endsWith('.bmp')
    );
  };

  const handleImagePick = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    files.forEach((file) => {
      const localId = `${Date.now()}_${Math.random().toString(16).slice(2)}`;
      const previewUrl = URL.createObjectURL(file);
      setImages((prev) => [
        ...prev,
        {
          id: localId,
          name: file.name,
          previewUrl,
          status: 'uploading',
        },
      ]);

      uploadFile(file)
        .then((res) => {
          setImages((prev) =>
            prev.map((img) =>
              img.id === localId
                ? {
                    ...img,
                    status: 'ready',
                    fileId: res.file_id,
                    size: res.size_bytes,
                    mime: res.mime_type,
                  }
                : img
            )
          );
        })
        .catch((err) => {
          console.error('Image upload failed:', err);
          setImages((prev) =>
            prev.map((img) =>
              img.id === localId
                ? { ...img, status: 'failed', error: '上传失败' }
                : img
            )
          );
        });
    });
    e.target.value = '';
  };

  const handleRemoveImage = (id) => {
    setImages((prev) => {
      const target = prev.find((img) => img.id === id);
      if (target?.previewUrl) {
        URL.revokeObjectURL(target.previewUrl);
      }
      return prev.filter((img) => img.id !== id);
    });
  };
  
  return (
    <div className={styles.panel}>
      <div className={styles.title}>
        <span>─ 研究输入</span>
        
        {isDisabled && (
          <button
            className={styles.toggleBtn}
            onClick={onStop}
            style={{ marginLeft: '16px', color: 'var(--accent-red)' }}
          >
            [■ 停止任务]
          </button>
        )}

        {isDisabled && (
          <button 
            className={styles.toggleBtn} 
            onClick={() => setShowConfig(!showConfig)}
            style={{ marginLeft: 'auto' }}
          >
            {showConfig ? '[收起配置]' : '[展开配置]'}
          </button>
        )}
      </div>
      
      <div className={styles.inputWrapper}>
        <span className={styles.prompt}>&gt;</span>
        <textarea
          className={styles.input}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholderText}
          disabled={isDisabled}
          rows={isDisabled && !showConfig ? 1 : 2}
          style={isDisabled && !showConfig ? { minHeight: '30px', resize: 'none' } : {}}
        />
      </div>

      <div className={styles.attachments}>
        <div className={styles.attachHeader}>
          <span className={styles.sectionTitle}>─ 文件输入</span>
          <div className={styles.attachActions}>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.txt,.zip,.png,.jpg,.jpeg,.webp,.gif,.bmp"
              multiple
              onChange={handleImagePick}
              disabled={isDisabled}
              style={{ display: 'none' }}
            />
            <button
              className={styles.attachBtn}
              onClick={() => fileInputRef.current?.click()}
              disabled={isDisabled}
            >
              + 添加文件
            </button>
          </div>
        </div>
        {images.length > 0 ? (
          <div className={styles.attachList}>
            {images.map((img) => (
              <div key={img.id} className={styles.attachItem}>
                <div className={styles.attachThumb}>
                  {isImageFile(img.name, img.mime) ? (
                    <img src={img.previewUrl} alt={img.name} />
                  ) : (
                    <div className={styles.fileBadge}>FILE</div>
                  )}
                </div>
                <div className={styles.attachMeta}>
                  <div className={styles.attachName}>{img.name}</div>
                  <div className={styles.attachStatus}>
                    {img.status === 'uploading' && '上传中...'}
                    {img.status === 'ready' && '已上传'}
                    {img.status === 'failed' && '上传失败'}
                  </div>
                </div>
                {!isDisabled && (
                  <button
                    className={styles.removeBtn}
                    onClick={() => handleRemoveImage(img.id)}
                    disabled={img.status === 'uploading'}
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className={styles.attachEmpty}>暂无文件</div>
        )}
        {(isUploading || hasFailed) && (
          <div className={styles.attachHint}>
            {isUploading && '文件上传中，完成后可开始研究。'}
            {!isUploading && hasFailed && '存在上传失败的文件，请删除后重试。'}
          </div>
        )}
      </div>
      
      {showConfig && (
        <>
          <div className={styles.scenariosSection}>
            <div className={styles.sectionTitle}>─ 预设场景</div>
            <div className={styles.scenarioGrid}>
              {loading ? (
                 <div style={{ color: 'var(--text-muted)' }}>加载场景中...</div>
              ) : (
                scenarios.map((scenario) => (
                  <ScenarioCard
                    key={scenario.id}
                    scenario={scenario}
                    selected={selectedScenario === scenario.id}
                    onClick={() => setSelectedScenario(
                      selectedScenario === scenario.id ? null : scenario.id
                    )}
                    disabled={isDisabled}
                  />
                ))
              )}
            </div>
          </div>
          
          <div className={styles.advancedToggle}>
            <button
              className={styles.toggleBtn}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              ─ 高级选项 {showAdvanced ? '▼' : '▶'}
            </button>
          </div>
          
          {showAdvanced && (
            <AdvancedOptions
              options={options}
              onChange={updateOptions}
              disabled={isDisabled}
            />
          )}
        </>
      )}
      
      {(!isDisabled || showConfig) && (
        <div className={styles.actions}>
          <button
            className={styles.submitBtn}
            onClick={handleSubmit}
            disabled={isDisabled || !question.trim() || isUploading || hasFailed}
          >
            {isDisabled ? '● 研究中...' : '▶ 开始深度研究'}
          </button>
        </div>
      )}
    </div>
  );
}

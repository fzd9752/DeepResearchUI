import { useState } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';

const itemStyle = {
  border: '1px solid var(--border-default)',
  borderRadius: '4px',
  overflow: 'hidden',
  background: 'var(--bg-tertiary)',
};

const headerStyle = {
  display: 'flex',
  alignItems: 'center',
  padding: '8px 12px',
  cursor: 'pointer',
  gap: '12px',
  fontSize: '14px',
};

const contentStyle = {
  padding: '12px',
  borderTop: '1px solid var(--border-default)',
  background: '#111',
  fontSize: '13px',
  fontFamily: 'var(--font-mono)',
};

const sectionTitleStyle = {
  color: 'var(--text-secondary)',
  fontWeight: 'bold',
  marginTop: '8px',
  marginBottom: '4px',
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
};

const observeLineStyle = {
  color: '#888',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  lineHeight: 1.5,
};

const observeHeaderStyle = {
  color: 'var(--text-primary)',
  fontWeight: 'bold',
  marginTop: '6px',
};

const observeItemStyle = {
  color: '#cbd5e1',
  paddingLeft: '12px',
  textIndent: '-12px',
  lineHeight: 1.5,
};

const observeSeparatorStyle = {
  height: '1px',
  background: '#1f2937',
  margin: '8px 0',
};

const showMoreBtnStyle = {
  background: 'none',
  border: 'none',
  color: 'var(--accent-cyan)',
  fontSize: '12px',
  cursor: 'pointer',
  padding: '4px 0',
  marginTop: '4px',
};

export default function RoundItem({ round, index, expanded, onToggle }) {
  const [showFullPreview, setShowFullPreview] = useState(false);
  const fullResult = round.resultFull || round.resultPreview || '';
  const previewResult =
    fullResult.length > 300 ? `${fullResult.slice(0, 300)}...` : fullResult;
  const dedupeList = (values) => {
    if (!Array.isArray(values)) return values ? [values] : [];
    return [...new Set(values.filter((item) => item))];
  };
  const mergeArg = (key) => {
    if (!round.toolArgs) return undefined;
    return round.toolArgs[key] ?? round.toolArgs.params?.[key];
  };
  const isSameValue = (a, b) => {
    try {
      return JSON.stringify(a) === JSON.stringify(b);
    } catch (e) {
      return a === b;
    }
  };
  const buildActText = () => {
    if (!round.toolArgs) return `> ${round.tool || 'tool'}()`;
    const tool = round.tool || 'tool';

    if (tool === 'search' || tool === 'google_scholar') {
      const queries = dedupeList(mergeArg('query'));
      if (queries.length === 0) return `> ${tool}()`;
      return [
        `> ${tool}`,
        '查询关键词:',
        ...queries.map((q, idx) => `${idx + 1}. ${q}`),
      ].join('\n');
    }

    if (tool === 'visit') {
      const urls = dedupeList(mergeArg('url'));
      const goal = mergeArg('goal');
      const lines = [`> ${tool}`];
      if (goal) lines.push(`目标: ${goal}`);
      if (urls.length) {
        lines.push('链接:');
        urls.forEach((u, idx) => lines.push(`${idx + 1}. ${u}`));
      }
      return lines.join('\n');
    }

    const cleanedArgs = { ...round.toolArgs };
    if (cleanedArgs.params && typeof cleanedArgs.params === 'object') {
      Object.keys(cleanedArgs.params).forEach((key) => {
        if (key in cleanedArgs && isSameValue(cleanedArgs[key], cleanedArgs.params[key])) {
          delete cleanedArgs.params[key];
        }
      });
      if (Object.keys(cleanedArgs.params).length === 0) {
        delete cleanedArgs.params;
      }
    }

    try {
      return `> ${tool}(\n${JSON.stringify(cleanedArgs, null, 2)}\n)`;
    } catch (e) {
      return `> ${tool}(${String(cleanedArgs)})`;
    }
  };
  const actText = buildActText();

  const renderResultLines = (text) =>
    text.split(/\r?\n/).map((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return <div key={idx} style={{ height: '6px' }} />;
      }
      if (trimmed === '=======') {
        return <div key={idx} style={observeSeparatorStyle} />;
      }
      if (trimmed.startsWith('## ')) {
        return (
          <div key={idx} style={observeHeaderStyle}>
            {trimmed.replace(/^##\s*/, '')}
          </div>
        );
      }
      if (/^\d+\.\s+/.test(trimmed)) {
        return (
          <div key={idx} style={observeItemStyle}>
            {trimmed}
          </div>
        );
      }
      return (
        <div key={idx} style={observeLineStyle}>
          {line}
        </div>
      );
    });

  // Explicit status mapping
  let statusIcon = '●';
  let statusColor = 'var(--accent-cyan)';

  switch (round.status) {
    case 'completed':
      statusIcon = '✓';
      statusColor = 'var(--accent-green)';
      break;
    case 'failed':
      statusIcon = '✗';
      statusColor = 'var(--accent-red)';
      break;
    case 'pending':
      statusIcon = '○';
      statusColor = 'var(--text-muted)';
      break;
    case 'thinking':
    case 'acting':
    case 'observing':
      statusIcon = '●'; // Active states
      statusColor = 'var(--accent-cyan)';
      break;
    default:
      statusIcon = '●';
      statusColor = 'var(--text-secondary)';
  }

  return (
    <div style={itemStyle}>
      <div style={headerStyle} onClick={onToggle}>
        <span style={{ color: 'var(--text-muted)' }}>
          {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </span>
        
        <span style={{ color: statusColor }}>{statusIcon}</span>
        
        <span style={{ fontWeight: 'bold' }}>Round {index + 1}</span>
        
        <span style={{ 
          background: '#333', 
          padding: '2px 6px', 
          borderRadius: '4px',
          fontSize: '12px' 
        }}>
          {round.tool || 'thinking...'}
        </span>
        
        <span style={{ marginLeft: 'auto', fontSize: '12px', color: 'var(--text-muted)' }}>
          {round.duration ? `${(round.duration / 1000).toFixed(1)}s` : ''}
        </span>
      </div>
      
      {expanded && (
        <div style={contentStyle}>
          {round.thinkContent && (
            <>
              <div style={sectionTitleStyle}>💭 THINK</div>
              <div style={{ color: 'var(--text-primary)', whiteSpace: 'pre-wrap' }}>
                {round.thinkContent}
              </div>
            </>
          )}
          
          {round.toolArgs && (
            <>
              <div style={{ ...sectionTitleStyle, marginTop: '16px' }}>🔧 ACT</div>
              <div style={{ color: '#a78bfa', whiteSpace: 'pre-wrap' }}>{actText}</div>
            </>
          )}
          
          {round.resultSummary && (
            <>
              <div style={{ ...sectionTitleStyle, marginTop: '16px' }}>👁 OBSERVE</div>
              <div style={{ color: 'var(--text-secondary)' }}>
                {round.resultSummary}
              </div>
            </>
          )}
          
          {fullResult && (
             <div style={{ marginTop: '8px', fontSize: '12px', color: '#555' }}>
               {showFullPreview ? (
                 <div>{renderResultLines(fullResult)}</div>
               ) : (
                 <div>{renderResultLines(previewResult)}</div>
               )}

               {fullResult.length > 300 && (
                 <button
                   style={showMoreBtnStyle}
                   onClick={(e) => {
                     e.stopPropagation();
                     setShowFullPreview(!showFullPreview);
                   }}
                 >
                   {showFullPreview ? '收起' : '展开'}
                 </button>
               )}
             </div>
          )}
        </div>
      )}
    </div>
  );
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return '--:--';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export function formatTime(isoString) {
  if (!isoString) return '';
  const date = new Date(isoString);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function formatLargeNumber(num) {
  if (num === null || num === undefined) return '0';
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}

export function getStatusColor(status) {
  switch (status) {
    case 'running': return 'var(--accent-green)';
    case 'completed': return 'var(--accent-cyan)';
    case 'failed': return 'var(--accent-red)';
    case 'pending': return 'var(--text-muted)';
    default: return 'var(--text-secondary)';
  }
}

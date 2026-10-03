/**
 * Shared step-score banding, used by both TraceHeatmap and StepTimeline so
 * their colours never drift apart. Split out so those files only export
 * components (fast refresh), mirroring config/runFilters.ts.
 *
 * step_scores sum to ~100 across the run, so most non-flagged steps sit under
 * 10; a step clearing 40 is a second signal worth a look even without the
 * model's own flag.
 */

export const CRITICAL_THRESHOLD = 40;
export const WARN_THRESHOLD = 10;

export type Band = 'critical' | 'warn' | 'pass';

export function bandFor(score: number, isFlagged: boolean): Band {
  if (isFlagged || score >= CRITICAL_THRESHOLD) return 'critical';
  if (score >= WARN_THRESHOLD) return 'warn';
  return 'pass';
}

export const BAND_LABEL: Record<Band, string> = {
  critical: 'flagged',
  warn: 'elevated',
  pass: 'healthy',
};

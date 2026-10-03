/**
 * Vertical execution timeline: a connected sequence of step badges, one row per
 * step in order. Agent steps execute sequentially, never branching, so a
 * straight connected line is the accurate shape for this data, not a
 * simplification of a real tree.
 *
 * Latency bars and their colour come from the real `duration_ms` recorded for
 * every step, compared against this run's own mean (a local, honestly-computed
 * outlier check, not the model). The model's per-step feature vector
 * (`duration_z` and the rest of the ten columns) is only computed and returned
 * for the flagged step, so that is the only row with the model's own
 * `duration_z` badge — showing it on every row would mean inventing numbers the
 * API never sent, which CLAUDE.md rules out.
 */

import { bandFor, CRITICAL_THRESHOLD, WARN_THRESHOLD } from '../lib/scoreBands';
import type { Band } from '../lib/scoreBands';
import type { DiagnosisResponse, StepDetail } from '../types/api';

const DOT_STYLE: Record<Band, string> = {
  critical: 'bg-critical shadow-[0_0_6px_var(--color-critical)]',
  warn: 'bg-warn shadow-[0_0_6px_var(--color-warn)]',
  pass: 'bg-pass/70',
};

const BAR_STYLE: Record<Band, string> = {
  critical: 'bg-critical/70',
  warn: 'bg-warn/70',
  pass: 'bg-muted/50',
};

/** Local latency-outlier check against this run's own steps, z-score based.
 * Distinct from the model's `duration_z`, which is only known for the flagged
 * step — this is computed client side from real `duration_ms` values only. */
function latencyBand(durationMs: number, mean: number, std: number): Band {
  if (std <= 0) return 'pass';
  const z = (durationMs - mean) / std;
  if (z >= 2) return 'critical';
  if (z >= 1) return 'warn';
  return 'pass';
}

export interface StepTimelineProps {
  steps: StepDetail[];
  scores: number[];
  diagnosis: DiagnosisResponse | null;
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

export function StepTimeline({
  steps,
  scores,
  diagnosis,
  selectedIndex,
  onSelect,
}: StepTimelineProps) {
  const durations = steps.map((s) => s.duration_ms);
  const maxDuration = Math.max(1, ...durations);
  const mean = durations.reduce((a, b) => a + b, 0) / Math.max(1, durations.length);
  const variance = durations.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, durations.length);
  const std = Math.sqrt(variance);

  const flaggedIndex = diagnosis?.flagged_step_index ?? null;
  const durationZ =
    diagnosis && typeof diagnosis.evidence.duration_z === 'number'
      ? diagnosis.evidence.duration_z
      : null;

  return (
    <div className="hairline bg-panel p-4">
      <p className="data text-muted">Execution timeline</p>
      <ol className="mt-4 flex flex-col gap-1">
        {steps.map((step, i) => {
          const score = scores[i] ?? 0;
          const isFlagged = i === flaggedIndex;
          const band = bandFor(score, isFlagged);
          const isSelected = i === selectedIndex;
          const widthPct = Math.max(4, Math.round((step.duration_ms / maxDuration) * 100));
          const isLast = i === steps.length - 1;
          const lBand = latencyBand(step.duration_ms, mean, std);

          return (
            <li key={step.id} className="relative pl-11">
              {!isLast && (
                <span
                  className="absolute left-[13px] top-7 z-0 h-[calc(100%-8px)] w-px bg-border"
                  aria-hidden="true"
                />
              )}

              <span
                className="absolute left-0 top-0 z-10 flex size-7 items-center justify-center rounded-full border border-border bg-panel"
                aria-hidden="true"
              >
                <span className="data text-text">{i}</span>
                <span
                  className={`absolute -right-0.5 -top-0.5 block size-2.5 rounded-full border-2 border-panel ${DOT_STYLE[band]}`}
                />
              </span>

              <button
                type="button"
                onClick={() => onSelect(i)}
                className={`group relative z-10 flex w-full items-start gap-3 rounded border-l-2 px-2.5 py-2 text-left transition-colors ${
                  isSelected
                    ? 'border-accent bg-accent/10'
                    : 'border-transparent hover:border-accent/40 hover:bg-accent/[0.06]'
                }`}
              >
                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-center gap-2">
                    <span className="data truncate text-text">
                      {step.tool_name ?? step.action_type}
                    </span>
                    {step.error_flag && (
                      <span className="data shrink-0 rounded border border-critical/40 bg-critical/10 px-1 py-0.5 text-critical shadow-[0_0_6px_-2px_var(--color-critical)]">
                        error
                      </span>
                    )}
                    {isFlagged && (
                      <span className="data shrink-0 rounded border border-critical/40 bg-critical/10 px-1 py-0.5 text-critical shadow-[0_0_6px_-2px_var(--color-critical)]">
                        flagged
                      </span>
                    )}
                    {isFlagged && durationZ !== null && (
                      <span
                        className={`data shrink-0 rounded border px-1 py-0.5 ${
                          Math.abs(durationZ) >= 3
                            ? 'border-critical/40 bg-critical/10 text-critical'
                            : 'border-warn/40 bg-warn/10 text-warn'
                        }`}
                        title="Model feature duration_z, computed for the flagged step only"
                      >
                        duration_z {durationZ.toFixed(2)}
                      </span>
                    )}
                  </span>

                  <span className="mt-1.5 flex items-center gap-2">
                    <span
                      className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg"
                      title={`${step.duration_ms} ms vs run mean ${Math.round(mean)} ms`}
                    >
                      <span
                        className={`block h-full rounded-full ${BAR_STYLE[lBand]}`}
                        style={{ width: `${widthPct}%` }}
                      />
                    </span>
                    <span className="data w-16 shrink-0 text-right text-muted">
                      {step.duration_ms} ms
                    </span>
                  </span>
                </span>

                <span className="flex shrink-0 flex-col items-end gap-1 pt-0.5">
                  <span className="data text-muted">{score}</span>
                  <span className="data inline-flex items-center gap-1 text-muted transition-colors group-hover:text-accent">
                    Inspect
                    <svg viewBox="0 0 16 16" className="size-3" aria-hidden="true">
                      <path
                        d="M6 4l4 4-4 4"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        fill="none"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      <p className="data mt-3 text-muted">
        score bands: &lt;{WARN_THRESHOLD} healthy · &lt;{CRITICAL_THRESHOLD} elevated · flagged step
        always critical
      </p>
    </div>
  );
}

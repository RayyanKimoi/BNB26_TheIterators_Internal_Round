/**
 * Horizontal per-step blame heatmap.
 *
 * Colour comes straight from `step_scores` (0 to 100, real, from the Hybrid
 * Sentry Engine) — nothing here is estimated. The flagged step is always
 * treated as critical regardless of its raw score, since that is the model's
 * actual call; other steps are bucketed by the thresholds below.
 */

import { useState } from 'react';

import { BAND_LABEL, bandFor } from '../lib/scoreBands';
import type { Band } from '../lib/scoreBands';
import type { StepDetail } from '../types/api';

const BAND_STYLE: Record<Band, { bg: string; ring: string }> = {
  critical: { bg: 'bg-critical', ring: 'shadow-[0_0_10px_-1px_var(--color-critical)]' },
  warn: { bg: 'bg-warn', ring: 'shadow-[0_0_8px_-2px_var(--color-warn)]' },
  pass: { bg: 'bg-pass/70', ring: '' },
};

export interface TraceHeatmapProps {
  steps: StepDetail[];
  scores: number[];
  flaggedIndex: number | null;
  /** Set when the invariant tier (not the classifier) flagged the step. */
  anomalySignal?: string | null;
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

export function TraceHeatmap({
  steps,
  scores,
  flaggedIndex,
  anomalySignal,
  selectedIndex,
  onSelect,
}: TraceHeatmapProps) {
  const [hovered, setHovered] = useState<number | null>(null);

  return (
    <div className="hairline bg-panel p-4">
      <div className="flex items-center justify-between">
        <p className="data text-muted">Per-step blame</p>
        <div className="flex items-center gap-3">
          <span className="data inline-flex items-center gap-1.5 text-muted">
            <span className="block size-1.5 rounded-full bg-pass/70" /> healthy
          </span>
          <span className="data inline-flex items-center gap-1.5 text-muted">
            <span className="block size-1.5 rounded-full bg-warn" /> elevated
          </span>
          <span className="data inline-flex items-center gap-1.5 text-muted">
            <span className="block size-1.5 rounded-full bg-critical" /> flagged
          </span>
        </div>
      </div>

      <div className="mt-3 flex items-end gap-1">
        {steps.map((step, i) => {
          const score = scores[i] ?? 0;
          const isFlagged = i === flaggedIndex;
          const band = bandFor(score, isFlagged);
          const style = BAND_STYLE[band];
          const height = 16 + Math.round(Math.min(100, score) * 0.4);
          const isSelected = i === selectedIndex;

          return (
            <div key={step.id} className="relative flex-1">
              <button
                type="button"
                onClick={() => onSelect(i)}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered((h) => (h === i ? null : h))}
                onFocus={() => setHovered(i)}
                onBlur={() => setHovered((h) => (h === i ? null : h))}
                aria-label={`Step ${i}: ${step.tool_name ?? step.action_type}, score ${score}, ${BAND_LABEL[band]}`}
                className={`block w-full rounded-sm transition-[transform,opacity,box-shadow] duration-150 ${style.bg} ${
                  isSelected
                    ? `-translate-y-1 opacity-100 ring-2 ring-accent shadow-[0_0_16px_-2px_var(--color-accent)]`
                    : `${style.ring} opacity-90 hover:opacity-100`
                }`}
                style={{ height }}
              />
              {hovered === i && (
                <div
                  role="tooltip"
                  className="data hairline pointer-events-none absolute bottom-full left-1/2 z-10 mb-2 w-max max-w-[220px] -translate-x-1/2 bg-panel px-2.5 py-1.5 text-text shadow-lg"
                >
                  <p>
                    step {i} · {BAND_LABEL[band]}
                  </p>
                  <p className="text-muted">{step.tool_name ?? step.action_type}</p>
                  <p className="text-muted">
                    {step.duration_ms} ms · score {score}
                  </p>
                  {isFlagged && anomalySignal && (
                    <p className="text-warn">signal: {anomalySignal}</p>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="mt-1.5 flex items-center justify-between">
        <span className="data text-muted">step 0</span>
        <span className="data text-muted">step {steps.length - 1}</span>
      </div>
    </div>
  );
}

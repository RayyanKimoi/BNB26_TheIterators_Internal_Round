/**
 * Side-by-side parent vs. child trace, shown once a fork completes.
 *
 * Deliberately a separate, compact renderer rather than reusing TraceHeatmap /
 * StepTimeline: this view's job is narrower (two columns, track one specific
 * step, no interactivity) and bending the main inspector components to fit it
 * would couple fork semantics into components the primary trace view also
 * depends on. The colour-banding rule itself (`bandFor`) is shared so the two
 * views never disagree about what counts as healthy.
 *
 * The target step (the one the fork patched) is marked with a mint ring on
 * both sides regardless of its score — that marker always means "this is the
 * step in question," while the bar's own colour always reflects that step's
 * real, independently-computed score. A fix can leave the model still
 * uncertain about a step without the fault being there any more, so forcing
 * the marked step green would be dishonest; showing its real score is not.
 */

import { motion, useReducedMotion } from 'framer-motion';

import { bandFor } from '../lib/scoreBands';
import type { Band } from '../lib/scoreBands';
import type { DiagnosisResponse, RunDetail, RunStatus, StepDetail } from '../types/api';

const BAND_BG: Record<Band, string> = {
  critical: 'bg-critical',
  warn: 'bg-warn',
  pass: 'bg-pass/70',
};

const DOT_BG: Record<Band, string> = {
  critical: 'bg-critical shadow-[0_0_6px_var(--color-critical)]',
  warn: 'bg-warn shadow-[0_0_6px_var(--color-warn)]',
  pass: 'bg-pass/70',
};

/**
 * Measured directly against the loaded 240-run corpus by
 * backend/seed_corpus.py: a fork flips a run to SUCCESS iff no step other than
 * the one being patched is still error-flagged afterward, since replay only
 * ever clears the patched step itself (the generator never writes the
 * `precondition_not_met` marker that would let a downstream step recover too).
 * The rate is bimodal, not a flat percentage: every premature_termination case
 * flips, no infinite_loop case can (replay cannot shorten a trace), and the
 * rest flip only when their fault does not cascade into later steps. Reported
 * as a fixed, cited corpus count, not a counter that updates per fork.
 */
const CORPUS_FLIP_SUMMARY =
  '33 of 144 failed runs in the loaded corpus flip this way: every premature_termination case does, no infinite_loop case can, and the rest flip when their fault does not cascade into later steps.';

function MiniHeatmap({
  steps,
  scores,
  flaggedIndex,
  targetIndex,
}: {
  steps: StepDetail[];
  scores: number[];
  flaggedIndex: number | null;
  targetIndex: number;
}) {
  return (
    <div className="flex items-end gap-1">
      {steps.map((step, i) => {
        const score = scores[i] ?? 0;
        const band = bandFor(score, i === flaggedIndex);
        const isTarget = i === targetIndex;
        const height = 14 + Math.round(Math.min(100, score) * 0.32);
        return (
          <div
            key={step.id}
            className="relative flex-1"
            title={`step ${i}: ${step.tool_name ?? step.action_type}, score ${score}`}
          >
            <div
              className={`rounded-sm ${BAND_BG[band]} ${
                isTarget ? 'ring-2 ring-accent shadow-[0_0_10px_-1px_var(--color-accent)]' : ''
              }`}
              style={{ height }}
            />
          </div>
        );
      })}
    </div>
  );
}

function MiniTimeline({
  steps,
  scores,
  flaggedIndex,
  targetIndex,
}: {
  steps: StepDetail[];
  scores: number[];
  flaggedIndex: number | null;
  targetIndex: number;
}) {
  return (
    <ol className="flex flex-col gap-1">
      {steps.map((step, i) => {
        const score = scores[i] ?? 0;
        const band = bandFor(score, i === flaggedIndex);
        const isTarget = i === targetIndex;
        return (
          <li
            key={step.id}
            className={`flex items-center gap-2 rounded border-l-2 px-2 py-1.5 ${
              isTarget ? 'border-accent bg-accent/10' : 'border-transparent'
            }`}
          >
            <span className={`block size-2 shrink-0 rounded-full ${DOT_BG[band]}`} />
            <span className="data w-5 shrink-0 text-muted">{i}</span>
            <span className="data min-w-0 flex-1 truncate text-text">
              {step.tool_name ?? step.action_type}
            </span>
            {step.error_flag && (
              <span className="data shrink-0 text-critical">error</span>
            )}
            <span className="data shrink-0 text-muted">{score}</span>
          </li>
        );
      })}
    </ol>
  );
}

/**
 * The subset of a fork's result this view actually renders. A real
 * `ForkResponse` satisfies this directly; a run opened because it already has
 * a `parent_run_id` (no fresh ForkResponse exists) is adapted to it from its
 * own RunDetail plus its parent's, with nothing invented for fields this view
 * does not display.
 */
export interface ForkComparisonData {
  outcome: RunStatus;
  parent_outcome: RunStatus;
  forked_at_step: number;
  steps: StepDetail[];
  diagnosis: DiagnosisResponse | null;
}

export interface TraceComparisonProps {
  parent: RunDetail;
  fork: ForkComparisonData;
  /** Omit when there is no separate run to navigate to, e.g. already viewing the child. */
  onOpenChild?: () => void;
  onDismiss: () => void;
}

export function TraceComparison({ parent, fork, onOpenChild, onDismiss }: TraceComparisonProps) {
  const reduced = useReducedMotion();
  const flipped = fork.parent_outcome === 'failed' && fork.outcome === 'success';
  const parentDiagnosis: DiagnosisResponse | null = parent.diagnosis;
  const childDiagnosis = fork.diagnosis;

  return (
    <motion.div
      initial={reduced ? { opacity: 0 } : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="flex flex-col gap-4"
    >
      <div
        className={`hairline flex flex-col gap-2 p-4 ${
          flipped ? 'border-pass/40 bg-pass/10' : 'border-warn/40 bg-warn/10'
        }`}
      >
        <p className={`font-display text-[15px] ${flipped ? 'text-pass' : 'text-warn'}`}>
          {flipped
            ? `Fork resolved the run: step ${fork.forked_at_step} flipped to SUCCESS`
            : `This fork did not change the outcome (still ${fork.outcome.toUpperCase()})`}
        </p>
        <p className="data text-muted">{CORPUS_FLIP_SUMMARY}</p>
        {!flipped && (
          <p className="data text-muted">
            {parent.injected_class === 'infinite_loop'
              ? 'infinite_loop never flips: replay re-executes the suffix deterministically and cannot shorten a trace, which is the only thing that would end this loop.'
              : 'A downstream step beyond the one just patched is still error-flagged, so the run as a whole is still a failure even though the patched step itself is now clean.'}
          </p>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="hairline flex flex-col gap-3 bg-panel p-4">
          <div className="flex items-center justify-between">
            <p className="data text-muted">Parent · original run</p>
            <span className="data inline-flex items-center gap-1.5 rounded border border-critical/40 bg-critical/10 px-1.5 py-0.5 text-critical">
              <span className="block size-1.5 rounded-full bg-critical shadow-[0_0_6px_currentColor]" />
              {fork.parent_outcome.toUpperCase()}
            </span>
          </div>
          <MiniHeatmap
            steps={parent.steps}
            scores={parentDiagnosis?.step_scores ?? []}
            flaggedIndex={parentDiagnosis?.flagged_step_index ?? null}
            targetIndex={fork.forked_at_step}
          />
          <MiniTimeline
            steps={parent.steps}
            scores={parentDiagnosis?.step_scores ?? []}
            flaggedIndex={parentDiagnosis?.flagged_step_index ?? null}
            targetIndex={fork.forked_at_step}
          />
        </div>

        <div className="hairline flex flex-col gap-3 bg-panel p-4">
          <div className="flex items-center justify-between">
            <p className="data text-muted">Child · forked run</p>
            <span
              className={`data inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 ${
                fork.outcome === 'success'
                  ? 'border-pass/40 bg-pass/10 text-pass'
                  : 'border-critical/40 bg-critical/10 text-critical'
              }`}
            >
              <span
                className={`block size-1.5 rounded-full shadow-[0_0_6px_currentColor] ${
                  fork.outcome === 'success' ? 'bg-pass' : 'bg-critical'
                }`}
              />
              {fork.outcome.toUpperCase()}
            </span>
          </div>
          <MiniHeatmap
            steps={fork.steps}
            scores={childDiagnosis?.step_scores ?? []}
            flaggedIndex={childDiagnosis?.flagged_step_index ?? null}
            targetIndex={fork.forked_at_step}
          />
          <MiniTimeline
            steps={fork.steps}
            scores={childDiagnosis?.step_scores ?? []}
            flaggedIndex={childDiagnosis?.flagged_step_index ?? null}
            targetIndex={fork.forked_at_step}
          />
        </div>
      </div>

      <div className="flex items-center gap-3">
        {onOpenChild && (
          <button
            type="button"
            onClick={onOpenChild}
            className="rounded border border-accent bg-accent/10 px-4 py-2 text-[13px] font-medium text-accent transition-colors hover:bg-accent/20"
          >
            Open child run in full trace view
          </button>
        )}
        <button
          type="button"
          onClick={onDismiss}
          className="data hairline rounded px-3 py-2 text-text transition-colors hover:border-accent/60 hover:text-accent"
        >
          Back to parent trace
        </button>
      </div>
    </motion.div>
  );
}

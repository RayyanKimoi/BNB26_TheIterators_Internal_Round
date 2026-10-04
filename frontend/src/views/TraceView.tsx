/**
 * Trace drill-down: one run's heatmap, timeline, and step inspector.
 *
 * Reached from the runs table's "Inspect Trace" button. If the run has no
 * diagnosis yet (a fresh fork, or a run nobody has scored), this gates on a
 * "Run Diagnosis" button rather than showing an empty heatmap or inventing
 * scores — POST /runs/{id}/diagnose is a real, if cheap, write, so it runs on
 * request, not silently on every visit.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { api } from '../api/client';
import { CopyableId } from '../components/CopyableId';
import { ErrorState } from '../components/AppShell';
import { StepInspector } from '../components/StepInspector';
import { StepTimeline } from '../components/StepTimeline';
import { TraceComparison } from '../components/TraceComparison';
import type { ForkComparisonData } from '../components/TraceComparison';
import { TraceHeatmap } from '../components/TraceHeatmap';
import { useToast } from '../hooks/useToast';
import type { ExplanationResponse, ForkResponse, RunDetail } from '../types/api';

type Tone = 'critical' | 'warn' | 'pass' | 'muted';

const CHIP_TONE: Record<Tone, string> = {
  critical: 'border-critical/40 bg-critical/10 text-critical',
  warn: 'border-warn/40 bg-warn/10 text-warn',
  pass: 'border-pass/40 bg-pass/10 text-pass',
  muted: 'border-border text-muted',
};

const DOT_TONE: Record<Tone, string> = {
  critical: 'bg-critical shadow-[0_0_6px_var(--color-critical)]',
  warn: 'bg-warn shadow-[0_0_6px_var(--color-warn)]',
  pass: 'bg-pass shadow-[0_0_6px_var(--color-pass)]',
  muted: 'bg-muted',
};

function Chip({ tone, title, children }: { tone: Tone; title?: string; children: ReactNode }) {
  return (
    <span
      title={title}
      className={`data inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 ${CHIP_TONE[tone]}`}
    >
      <span className={`block size-1.5 rounded-full ${DOT_TONE[tone]}`} />
      {children}
    </span>
  );
}

function TraceViewSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <div className="h-7 w-28 animate-pulse rounded bg-border" />
      <div className="hairline flex flex-wrap items-center gap-x-6 gap-y-3 bg-panel p-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="flex flex-col gap-1.5">
            <div className="h-2.5 w-14 animate-pulse rounded bg-border" />
            <div className="h-4 w-20 animate-pulse rounded bg-border" />
          </div>
        ))}
      </div>
      <div className="hairline bg-panel p-4">
        <div className="h-2.5 w-24 animate-pulse rounded bg-border" />
        <div className="mt-3 flex items-end gap-1">
          {Array.from({ length: 12 }).map((_, i) => (
            <div
              key={i}
              className="flex-1 animate-pulse rounded-sm bg-border"
              style={{ height: 16 + ((i * 13) % 34) }}
            />
          ))}
        </div>
      </div>
      <div className="hairline bg-panel p-4">
        <div className="h-2.5 w-32 animate-pulse rounded bg-border" />
        <div className="mt-4 flex flex-col gap-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3 pl-11">
              <div className="size-7 shrink-0 animate-pulse rounded-full bg-border" />
              <div className="h-4 flex-1 animate-pulse rounded bg-border" />
              <div className="h-4 w-16 animate-pulse rounded bg-border" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export interface TraceViewProps {
  runId: string;
  onBack: () => void;
  /** Switches the dashboard to another run's trace, e.g. a fresh fork's child. */
  onNavigateToRun?: (runId: string) => void;
}

export function TraceView({ runId, onBack, onNavigateToRun }: TraceViewProps) {
  const toast = useToast();
  const reduced = useReducedMotion();

  const [run, setRun] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [diagnosing, setDiagnosing] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [explanations, setExplanations] = useState<Record<number, ExplanationResponse>>({});
  const [forkResult, setForkResult] = useState<ForkResponse | null>(null);
  const [comparisonParent, setComparisonParent] = useState<RunDetail | null>(null);
  const [comparisonDismissed, setComparisonDismissed] = useState(false);

  useEffect(() => {
    // RunsView swaps its whole subtree between the table and this view, so
    // TraceView always mounts fresh for a given runId; the useState
    // initializers above already cover loading/error/selection, with nothing
    // to reset here.
    const controller = new AbortController();

    api
      .getRun(runId, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setRun(data);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [runId]);

  // No diagnosis yet: score it without making the user ask. The manual
  // button below stays for retries and for the case where this fails, but
  // arriving at a trace and finding it unscored is a dead end, not a choice.
  const autoDiagnosed = useRef<string | null>(null);
  useEffect(() => {
    if (!run || run.diagnosis || diagnosing) return;
    // Guard per runId: without it, the setRun this triggers re-runs the
    // effect and fires a second diagnosis before the first lands.
    if (autoDiagnosed.current === runId) return;
    autoDiagnosed.current = runId;
    void runDiagnosis();
    // runDiagnosis is stable for a given runId and intentionally omitted:
    // including it would re-run this on every render it is recreated on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, runId, diagnosing]);

  // Open on the step the engine blamed. Only until the user picks another:
  // once selectedIndex is set, this must never yank the selection back.
  const autoSelected = useRef<string | null>(null);
  useEffect(() => {
    const flagged = run?.diagnosis?.flagged_step_index;
    if (flagged === undefined || flagged === null) return;
    if (autoSelected.current === runId) return;
    autoSelected.current = runId;
    setSelectedIndex(flagged);
  }, [run?.diagnosis?.flagged_step_index, runId]);

  // This run is itself a fork: fetch its parent so the comparison view can
  // render without a fresh ForkResponse (there isn't one — nothing was just
  // forked this visit).
  useEffect(() => {
    // run.parent_run_id only ever goes from unset to set once per mount (the
    // fetch above populates `run` exactly once), so there is no later state
    // to reset back to null here; the initial useState(null) already covers
    // "no parent" for the lifetime of this component instance.
    if (!run?.parent_run_id) return;
    const controller = new AbortController();
    api
      .getRun(run.parent_run_id, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setComparisonParent(data);
      })
      .catch(() => {
        // Non-fatal: the comparison view just does not render without it,
        // and the child's own trace still works normally.
      });
    return () => controller.abort();
  }, [run?.parent_run_id]);

  const runDiagnosis = async () => {
    setDiagnosing(true);
    try {
      const result = await api.diagnoseRun(runId);
      setRun((prev) => (prev ? { ...prev, diagnosis: result } : prev));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not run diagnosis.';
      toast.push('error', message);
    } finally {
      setDiagnosing(false);
    }
  };

  if (loading) return <TraceViewSkeleton />;
  if (error) {
    return (
      <ErrorState
        error={error}
        onRetry={() => {
          setError(null);
          setLoading(true);
          // Re-trigger the fetch effect by toggling runId is not possible here;
          // simplest correct retry is a full back-and-reopen, which this error
          // state's "Retry" wording does not promise, so send them back instead.
          onBack();
        }}
      />
    );
  }
  if (!run) return null;

  const diagnosis = run.diagnosis;
  const scores = diagnosis?.step_scores ?? null;
  const selectedStep =
    selectedIndex !== null ? (run.steps.find((s) => s.step_index === selectedIndex) ?? null) : null;

  // Two ways into the comparison view: we just forked from here (forkResult,
  // real ForkResponse), or this run already had a parent when opened (its
  // sibling fetched above). Either way, adapt down to the one shape
  // TraceComparison needs so nothing unused gets invented for the second case.
  const comparison: { parent: RunDetail; fork: ForkComparisonData; showOpenChild: boolean } | null =
    forkResult && run
      ? { parent: run, fork: forkResult, showOpenChild: true }
      : comparisonParent
        ? {
            parent: comparisonParent,
            fork: {
              outcome: run.status,
              parent_outcome: comparisonParent.status,
              forked_at_step: run.forked_at_step ?? 0,
              steps: run.steps,
              diagnosis: run.diagnosis,
            },
            showOpenChild: false,
          }
        : null;

  const predictedTone: Tone | null = !diagnosis
    ? null
    : diagnosis.predicted_class === 'unknown'
      ? 'warn'
      : diagnosis.predicted_class === run.injected_class
        ? 'pass'
        : 'critical';

  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.08 } },
  };
  const item: Variants = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
  };

  return (
    <div className="flex flex-col gap-4">
      <button
        type="button"
        onClick={onBack}
        className="data hairline inline-flex w-fit items-center gap-2 rounded px-3 py-1.5 text-text transition-colors hover:border-accent/60 hover:text-accent"
      >
        <svg viewBox="0 0 16 16" className="size-3.5" aria-hidden="true">
          <path
            d="M10 3l-5 5 5 5"
            stroke="currentColor"
            strokeWidth="1.6"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        Back to Runs
      </button>

      <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-4">
        <motion.div
          variants={item}
          className="hairline flex flex-wrap items-center gap-x-6 gap-y-3 bg-panel p-4"
        >
          <div>
            <p className="data text-muted">run id</p>
            <CopyableId id={run.id} />
          </div>
          <div>
            <p className="data text-muted">status</p>
            <Chip tone={run.status === 'success' ? 'pass' : 'critical'}>
              {run.status === 'success' ? 'SUCCESS' : 'FAIL'}
            </Chip>
          </div>
          <div>
            <p className="data text-muted">task</p>
            <span className="data rounded border border-border px-1.5 py-0.5 text-text">
              {run.task_type}
            </span>
          </div>
          <div>
            <p className="data text-muted">fault class</p>
            <Chip tone={run.injected_class ? 'critical' : 'pass'}>
              {run.injected_class ?? 'clean'}
            </Chip>
          </div>
          {diagnosis && predictedTone && (
            <div>
              <p className="data text-muted">predicted</p>
              <Chip
                tone={predictedTone}
                title={
                  diagnosis.anomaly_signal
                    ? `Distribution-relative invariant signal: ${diagnosis.anomaly_signal}`
                    : undefined
                }
              >
                {diagnosis.predicted_class}
                {diagnosis.anomaly_signal ? ` · ${diagnosis.anomaly_signal}` : ''}
              </Chip>
            </div>
          )}
        </motion.div>

        {comparison && !comparisonDismissed ? (
          <motion.div variants={item}>
            <TraceComparison
              parent={comparison.parent}
              fork={comparison.fork}
              onOpenChild={
                comparison.showOpenChild && forkResult && onNavigateToRun
                  ? () => onNavigateToRun(forkResult.child_run_id)
                  : undefined
              }
              onDismiss={() => setComparisonDismissed(true)}
            />
          </motion.div>
        ) : !diagnosis ? (
          <motion.div
            variants={item}
            className="hairline flex flex-col items-start gap-3 bg-panel p-6"
          >
            <p className="data text-text">This run has not been diagnosed yet.</p>
            <p className="data text-muted">
              Run the Hybrid Sentry Engine to score every step before inspecting the trace.
            </p>
            <button
              type="button"
              onClick={runDiagnosis}
              disabled={diagnosing}
              className="rounded border border-accent bg-accent/10 px-4 py-2 text-[13px] font-medium text-accent transition-colors hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {diagnosing ? 'Running diagnosis...' : 'Run Diagnosis'}
            </button>
          </motion.div>
        ) : (
          <>
            {/* The headline verdict. `confidence` is the flagged step's own
                share of step_scores, and `anomaly_signal` being set means the
                invariant tier fired rather than the classifier, in which case
                predicted_class is "unknown" by contract and this must not
                name a class it does not have. */}
            <motion.div
              variants={item}
              className="hairline border-critical/50 bg-critical/5 p-4"
              role="status"
            >
              <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                <span className="flex items-center gap-2">
                  <span
                    className="block size-2 rounded-full bg-critical shadow-[0_0_8px_var(--color-critical)]"
                    aria-hidden="true"
                  />
                  <span className="font-display text-[15px] font-bold tracking-tight text-critical">
                    ROOT CAUSE LOCATED
                  </span>
                </span>

                <span className="data rounded border border-critical/40 bg-critical/10 px-2 py-0.5 text-critical">
                  step {diagnosis.flagged_step_index}
                </span>

                <span className="data rounded border border-border px-2 py-0.5 text-muted">
                  confidence {Math.round((diagnosis.confidence ?? 0) * 100)}%
                </span>

                {diagnosis.predicted_class !== 'unknown' && (
                  <span className="data rounded border border-warn/40 bg-warn/10 px-2 py-0.5 text-warn">
                    {diagnosis.predicted_class}
                    {diagnosis.class_confidence
                      ? ` ${Math.round(diagnosis.class_confidence * 100)}%`
                      : ''}
                  </span>
                )}

                {diagnosis.anomaly_signal && (
                  <span className="data rounded border border-accent/40 bg-accent/10 px-2 py-0.5 text-accent">
                    {diagnosis.anomaly_signal}
                  </span>
                )}
              </div>

              {diagnosis.evidence_path && (
                <p className="data mt-2 text-muted">
                  evidence at <span className="text-text">{diagnosis.evidence_path}</span>
                </p>
              )}

              {diagnosis.unknown_reason && (
                <p className="data mt-2 leading-relaxed text-muted">
                  {diagnosis.unknown_reason}
                </p>
              )}
            </motion.div>

            <motion.div variants={item}>
              <TraceHeatmap
                steps={run.steps}
                scores={scores ?? []}
                flaggedIndex={diagnosis.flagged_step_index}
                anomalySignal={diagnosis.anomaly_signal}
                selectedIndex={selectedIndex}
                onSelect={setSelectedIndex}
              />
            </motion.div>
            <motion.div variants={item}>
              <StepTimeline
                steps={run.steps}
                scores={scores ?? []}
                diagnosis={diagnosis}
                selectedIndex={selectedIndex}
                onSelect={setSelectedIndex}
              />
            </motion.div>
          </>
        )}
      </motion.div>

      <StepInspector
        step={selectedStep}
        diagnosis={diagnosis}
        runId={run.id}
        explanation={selectedStep ? (explanations[selectedStep.step_index] ?? null) : null}
        onExplained={(resp) =>
          setExplanations((prev) => ({ ...prev, [resp.step_index]: resp }))
        }
        onForked={(result) => {
          setForkResult(result);
          setComparisonDismissed(false);
          setSelectedIndex(null);
          toast.push(
            result.outcome === 'success' && result.parent_outcome === 'failed'
              ? 'success'
              : 'info',
            result.outcome === 'success' && result.parent_outcome === 'failed'
              ? `Fork flipped step ${result.forked_at_step} from FAIL to SUCCESS.`
              : `Fork completed. Outcome: ${result.outcome.toUpperCase()}.`,
          );
        }}
        onJumpToFlagged={
          diagnosis ? () => setSelectedIndex(diagnosis.flagged_step_index) : undefined
        }
        onClose={() => setSelectedIndex(null)}
      />
    </div>
  );
}

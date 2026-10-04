/**
 * Slide-over drawer: step telemetry, SHAP attribution, and the Gemini
 * root-cause explanation for one step.
 *
 * Evidence and SHAP are only ever shown for the flagged step. The backend
 * computes both from the run's diagnosis, which describes the flagged step
 * specifically (see backend/main.py::explain_run) — rendering them against a
 * different step would be attributing someone else's evidence. The Explain
 * CTA is restricted the same way, for the same reason, rather than letting the
 * UI call an endpoint that would silently mix evidence from two different
 * steps.
 */

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useEffect, useState } from 'react';

import { api, ApiError } from '../api/client';
import { LoadingState } from './AppShell';
import type {
  DiagnosisResponse,
  ExplanationResponse,
  ForkResponse,
  StepDetail,
  SuggestedFix,
} from '../types/api';

/** Cycled while a fork request is in flight, matching the two real phases of
 * the work: writing the child run, then replaying its suffix through the
 * detector. Reduced motion shows the two joined as one static line instead. */
const FORK_LOADING_MESSAGES = [
  'Creating child run...',
  're-executing suffix steps through fault detector model...',
];

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

function JsonBlock({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <p className="data text-muted">{label}</p>
      <pre className="data hairline mt-1.5 max-h-48 overflow-auto bg-bg p-2.5 text-text">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

/**
 * The explain button, its loading/error state, and the rendered result.
 *
 * Given a fresh `key={step.id}` by the parent, so switching steps remounts
 * this instead of needing an effect to reset `loading`/`requestError` against
 * the new step — the idiomatic way to reset state on a prop change.
 */
function ExplainPanel({
  step,
  runId,
  explanation,
  onExplained,
}: {
  step: StepDetail;
  runId: string;
  explanation: ExplanationResponse | null;
  onExplained: (explanation: ExplanationResponse) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);

  const handleExplain = async () => {
    setLoading(true);
    setRequestError(null);
    try {
      const resp = await api.explainRun(runId, { step_index: step.step_index });
      onExplained(resp);
    } catch (err) {
      const message =
        err instanceof ApiError && err.isUpstreamFailure
          ? 'Gemini explainer is rate-limited or unavailable right now. Try again in a moment.'
          : err instanceof Error
            ? err.message
            : 'Could not fetch an explanation.';
      setRequestError(message);
    } finally {
      setLoading(false);
    }
  };

  if (explanation) {
    return (
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="data rounded border border-border px-1.5 py-0.5 text-muted">
            predicted: {explanation.predicted_class}
          </span>
          {explanation.cached && (
            <span className="data rounded border border-pass/40 bg-pass/10 px-1.5 py-0.5 text-pass">
              cached · 0 tokens used
            </span>
          )}
        </div>
        <div>
          <p className="data text-muted">root_cause</p>
          <p className="mt-1 text-[13px] leading-relaxed text-text">{explanation.root_cause}</p>
        </div>
        <div>
          <p className="data text-muted">evidence_summary</p>
          <ul className="mt-1 list-disc pl-4 text-[13px] leading-relaxed text-text">
            {explanation.evidence_summary.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </div>
        <div>
          <p className="data text-muted">proposed_fix</p>
          <pre className="data hairline mt-1.5 overflow-auto bg-bg p-2.5 text-pass">
            {explanation.proposed_fix}
          </pre>
        </div>
      </div>
    );
  }

  if (loading) return <LoadingState label="Asking Gemini for a root cause" />;

  return (
    <div className="flex flex-col gap-2.5">
      {requestError && (
        <p className="data text-critical" role="alert">
          {requestError}
        </p>
      )}
      <button
        type="button"
        onClick={handleExplain}
        className="self-start rounded border border-accent bg-accent/10 px-4 py-2 text-[13px] font-medium text-accent transition-colors hover:bg-accent/20"
      >
        Explain Root Cause with Gemini
      </button>
    </div>
  );
}

interface CandidateOutcome {
  rank: number;
  rationale: string;
  patch: Record<string, unknown>;
  result: ForkResponse | null;
  error: string | null;
}

/**
 * PRD item 14: fork every candidate Gemini proposed, in parallel, and show
 * which one actually flips the run.
 *
 * Each candidate is a real, independent fork against the live endpoint — the
 * ranking shown is the measured outcome of running them, not Gemini's own
 * confidence in its suggestions. A candidate that errors is reported as
 * errored rather than silently dropped, because "this patch could not even be
 * replayed" is a result worth seeing.
 */
function CandidateForkPanel({
  step,
  runId,
  candidates,
  onForked,
}: {
  step: StepDetail;
  runId: string;
  candidates: SuggestedFix[];
  onForked: (result: ForkResponse) => void;
}) {
  const [outcomes, setOutcomes] = useState<CandidateOutcome[] | null>(null);
  const [running, setRunning] = useState(false);
  const [savedFor, setSavedFor] = useState<number | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const forkAll = async () => {
    setRunning(true);
    setSaveError(null);
    setSavedFor(null);

    // Parallel: these are independent writes against different child runs,
    // and serialising them would make three model passes feel like a stall.
    const settled = await Promise.allSettled(
      candidates.map((candidate) =>
        api.forkRun(runId, {
          from_step: step.step_index,
          fix_payload: candidate.patch,
          override_code: `Candidate ${candidate.rank}: ${candidate.rationale}`,
        }),
      ),
    );

    const next: CandidateOutcome[] = candidates.map((candidate, i) => {
      const entry = settled[i];
      return {
        rank: candidate.rank,
        rationale: candidate.rationale,
        patch: candidate.patch,
        result: entry.status === 'fulfilled' ? entry.value : null,
        error:
          entry.status === 'rejected'
            ? entry.reason instanceof Error
              ? entry.reason.message
              : 'Fork failed'
            : null,
      };
    });

    setOutcomes(next);
    setRunning(false);

    // Surface the winner to the parent so the comparison view opens on a
    // fork that actually resolved the run; fall back to any successful call.
    const winner =
      next.find((o) => o.result && o.result.outcome === 'success' && o.result.parent_outcome === 'failed') ??
      next.find((o) => o.result);
    if (winner?.result) onForked(winner.result);
  };

  const saveRegressionTest = async (outcome: CandidateOutcome) => {
    if (!outcome.result) return;
    setSaveError(null);
    try {
      await api.createRegressionTest(runId, {
        assertion: {
          forked_at_step: outcome.result.forked_at_step,
          fix_payload: outcome.patch,
          expected_outcome: outcome.result.outcome,
          parent_outcome: outcome.result.parent_outcome,
          child_run_id: outcome.result.child_run_id,
          rationale: outcome.rationale,
        },
      });
      setSavedFor(outcome.rank);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Could not save the regression test.');
    }
  };

  if (running) {
    return (
      <div className="hairline flex items-center gap-3 bg-bg px-4 py-4">
        <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-accent" aria-hidden="true" />
        <span className="data text-muted">
          Forking {candidates.length} candidates in parallel and re-scoring each child run...
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {(outcomes ?? candidates.map((c) => ({ ...c, result: null, error: null }))).map((entry) => {
        const outcome = 'result' in entry ? (entry as CandidateOutcome) : null;
        const result = outcome?.result ?? null;
        const flipped = result ? result.outcome === 'success' && result.parent_outcome === 'failed' : false;

        return (
          <div
            key={entry.rank}
            className={`hairline bg-bg p-3 ${flipped ? 'border-pass/50' : ''}`}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="data rounded border border-border px-1.5 py-0.5 text-muted">
                candidate {entry.rank}
              </span>
              {result && (
                <span
                  className={`data rounded border px-1.5 py-0.5 ${
                    flipped
                      ? 'border-pass/40 bg-pass/10 text-pass'
                      : 'border-critical/40 bg-critical/10 text-critical'
                  }`}
                >
                  {flipped ? 'flipped to SUCCESS' : `still ${result.outcome.toUpperCase()}`}
                </span>
              )}
              {outcome?.error && (
                <span className="data rounded border border-critical/40 bg-critical/10 px-1.5 py-0.5 text-critical">
                  fork failed
                </span>
              )}
            </div>

            {entry.rationale && (
              <p className="mt-2 text-[13px] leading-relaxed text-muted">{entry.rationale}</p>
            )}

            <pre className="data hairline mt-2 overflow-x-auto bg-panel p-2.5 text-text">
              {JSON.stringify(entry.patch, null, 2)}
            </pre>

            {outcome?.error && (
              <p className="data mt-2 text-critical" role="alert">
                {outcome.error}
              </p>
            )}

            {flipped && (
              <button
                type="button"
                onClick={() => saveRegressionTest(outcome as CandidateOutcome)}
                disabled={savedFor === entry.rank}
                className="data hairline mt-2 rounded px-2.5 py-1 text-text transition-colors hover:border-accent/60 hover:text-accent disabled:cursor-not-allowed disabled:opacity-60"
              >
                {savedFor === entry.rank ? 'Saved as regression test' : 'Save as Regression Test'}
              </button>
            )}
          </div>
        );
      })}

      {saveError && (
        <p className="data text-critical" role="alert">
          {saveError}
        </p>
      )}

      {!outcomes && (
        <button
          type="button"
          onClick={forkAll}
          className="self-start rounded border border-accent bg-accent/10 px-4 py-2 text-[13px] font-medium text-accent transition-colors hover:bg-accent/20"
        >
          Fork All {candidates.length} Candidates
        </button>
      )}
    </div>
  );
}

/**
 * Optional fix fields, the fork action, and its loading/error state. Given a
 * fresh `key={step.id}` by the parent for the same reason as ExplainPanel:
 * a remount resets state on step change with no effect required.
 */
function ForkPanel({
  step,
  runId,
  onForked,
}: {
  step: StepDetail;
  runId: string;
  onForked: (result: ForkResponse) => void;
}) {
  const reduced = useReducedMotion();
  const [fixPayloadText, setFixPayloadText] = useState('');
  const [overrideCode, setOverrideCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    if (!loading || reduced) return;
    const id = setInterval(
      () => setPhase((p) => (p + 1) % FORK_LOADING_MESSAGES.length),
      1100,
    );
    return () => clearInterval(id);
  }, [loading, reduced]);

  const handleFork = async () => {
    setError(null);

    let fixPayload: Record<string, unknown> | null = null;
    const trimmed = fixPayloadText.trim();
    if (trimmed) {
      try {
        const parsed: unknown = JSON.parse(trimmed);
        if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
          throw new Error('must be a JSON object, e.g. {"currency": "EUR"}');
        }
        fixPayload = parsed as Record<string, unknown>;
      } catch (err) {
        setError(
          `Invalid fix_payload: ${err instanceof Error ? err.message : 'not valid JSON'}`,
        );
        return;
      }
    }

    setLoading(true);
    setPhase(0);
    try {
      const result = await api.forkRun(runId, {
        from_step: step.step_index,
        fix_payload: fixPayload,
        override_code: overrideCode.trim() || null,
      });
      onForked(result);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : 'Could not fork this run. Try again.',
      );
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="hairline flex items-center gap-3 bg-bg px-4 py-4">
        <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-accent" aria-hidden="true" />
        <span className="data text-muted">
          {reduced
            ? 'Creating child run and re-executing suffix steps through the fault detector model...'
            : FORK_LOADING_MESSAGES[phase]}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5">
      <label className="data flex flex-col gap-1.5 text-muted">
        fix_payload (JSON, optional)
        <textarea
          value={fixPayloadText}
          onChange={(e) => setFixPayloadText(e.target.value)}
          placeholder={'{\n  "currency": "EUR"\n}'}
          rows={3}
          spellCheck={false}
          className="hairline rounded bg-bg px-3 py-2 font-mono text-[12px] text-text placeholder:text-muted focus-visible:outline-none"
        />
      </label>
      <label className="data flex flex-col gap-1.5 text-muted">
        override_code (optional note, recorded for audit, never executed)
        <textarea
          value={overrideCode}
          onChange={(e) => setOverrideCode(e.target.value)}
          placeholder="e.g. manually validated the corrected payload against the schema"
          rows={2}
          spellCheck={false}
          className="hairline rounded bg-bg px-3 py-2 font-mono text-[12px] text-text placeholder:text-muted focus-visible:outline-none"
        />
      </label>
      {error && (
        <p className="data text-critical" role="alert">
          {error}
        </p>
      )}
      <button
        type="button"
        onClick={handleFork}
        className="self-start rounded border border-accent bg-accent/10 px-4 py-2 text-[13px] font-medium text-accent transition-colors hover:bg-accent/20"
      >
        Fork Run from Step {step.step_index}
      </button>
    </div>
  );
}

function ShapBars({ shap }: { shap: Partial<Record<string, number>> }) {
  const entries = Object.entries(shap)
    .filter((e): e is [string, number] => typeof e[1] === 'number')
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 6);
  if (entries.length === 0) return null;
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)));

  return (
    <div className="flex flex-col gap-2">
      {entries.map(([name, value]) => {
        const positive = value >= 0;
        const widthPct = Math.max(4, Math.round((Math.abs(value) / maxAbs) * 100));
        return (
          <div key={name} className="flex items-center gap-3">
            <span className="data w-36 shrink-0 truncate text-muted">{name}</span>
            <span className="hairline h-3 flex-1 overflow-hidden rounded bg-bg">
              <span
                className={`block h-full rounded ${positive ? 'bg-accent' : 'bg-muted/50'}`}
                style={{ width: `${widthPct}%` }}
              />
            </span>
            <span className={`data w-16 shrink-0 text-right ${positive ? 'text-accent' : 'text-muted'}`}>
              {value >= 0 ? '+' : ''}
              {value.toFixed(4)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export interface StepInspectorProps {
  step: StepDetail | null;
  diagnosis: DiagnosisResponse | null;
  runId: string;
  explanation: ExplanationResponse | null;
  onExplained: (explanation: ExplanationResponse) => void;
  onForked: (result: ForkResponse) => void;
  onJumpToFlagged?: () => void;
  onClose: () => void;
}

export function StepInspector({
  step,
  diagnosis,
  runId,
  explanation,
  onExplained,
  onForked,
  onJumpToFlagged,
  onClose,
}: StepInspectorProps) {
  const reduced = useReducedMotion();

  useEffect(() => {
    if (!step) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, onClose]);

  const isFlagged = step !== null && diagnosis !== null && step.step_index === diagnosis.flagged_step_index;

  return (
    <AnimatePresence>
      {step && (
        <>
          <motion.div
            key="backdrop"
            className="fixed inset-0 z-40 bg-bg/70 backdrop-blur-sm"
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
          />
          <motion.div
            key="panel"
            role="dialog"
            aria-modal="true"
            aria-label={`Step ${step.step_index} inspector`}
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l border-border bg-panel"
            initial={reduced ? { opacity: 0 } : { x: '100%' }}
            animate={{ x: 0, opacity: 1 }}
            exit={reduced ? { opacity: 0 } : { x: '100%' }}
            transition={{ duration: 0.28, ease: 'easeOut' }}
          >
            <div className="flex items-center justify-between border-b border-border px-5 py-4">
              <div>
                <p className="font-display text-[15px] text-text">
                  Step {step.step_index}
                  {isFlagged && <span className="ml-2 text-critical">flagged</span>}
                </p>
                <p className="data mt-0.5 text-muted">
                  {step.tool_name ?? step.action_type} · {step.duration_ms} ms · {step.tokens} tokens
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close inspector"
                className="data hairline rounded p-1.5 text-muted transition-colors hover:border-accent/60 hover:text-accent"
              >
                <svg viewBox="0 0 16 16" className="size-4" aria-hidden="true">
                  <path
                    d="M4 4l8 8M12 4l-8 8"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    fill="none"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4">
              <div className="flex flex-col gap-5">
                <section>
                  <p className="data mb-2 uppercase tracking-[0.08em] text-muted">Step telemetry</p>
                  <div className="flex flex-col gap-3">
                    <JsonBlock label="input" value={step.input} />
                    <JsonBlock label="output" value={step.output} />
                    <JsonBlock label="state_snapshot" value={step.state_snapshot} />
                  </div>
                  <p className="data mt-2 text-muted">
                    state_hash {step.state_hash || '(none)'} · error_flag{' '}
                    {step.error_flag ? 'true' : 'false'}
                  </p>
                </section>

                <section>
                  <p className="data mb-2 uppercase tracking-[0.08em] text-muted">Evidence</p>
                  {isFlagged && diagnosis ? (
                    <div className="hairline flex flex-col divide-y divide-border bg-bg">
                      {Object.entries(diagnosis.evidence).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between px-2.5 py-1.5">
                          <span className="data text-muted">{key}</span>
                          <span className="data text-text">{formatValue(value)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="data text-muted">
                      Evidence is recorded for the flagged step only
                      {diagnosis ? ` (step ${diagnosis.flagged_step_index}).` : '.'}
                      {diagnosis && onJumpToFlagged && (
                        <button
                          type="button"
                          onClick={onJumpToFlagged}
                          className="ml-2 text-accent transition-colors hover:underline"
                        >
                          Jump to it
                        </button>
                      )}
                    </p>
                  )}
                </section>

                <section>
                  <p className="data mb-2 uppercase tracking-[0.08em] text-muted">
                    SHAP feature attribution
                  </p>
                  {!isFlagged ? (
                    <p className="data text-muted">SHAP explains the flagged step only.</p>
                  ) : diagnosis?.shap ? (
                    <ShapBars shap={diagnosis.shap} />
                  ) : (
                    <p className="data text-muted">SHAP attribution unavailable for this diagnosis.</p>
                  )}
                </section>

                <section>
                  <p className="data mb-2 uppercase tracking-[0.08em] text-muted">
                    Root cause explanation
                  </p>

                  {!isFlagged ? (
                    <p className="data text-muted">
                      Explanations are generated for the flagged step, since that is the step the
                      diagnosis evidence describes.
                    </p>
                  ) : (
                    <ExplainPanel
                      key={step.id}
                      step={step}
                      runId={runId}
                      explanation={explanation}
                      onExplained={onExplained}
                    />
                  )}
                </section>

                {isFlagged && explanation && explanation.fix_candidates.length > 0 && (
                  <section>
                    <p className="data mb-2 uppercase tracking-[0.08em] text-muted">
                      Candidate fixes ({explanation.fix_candidates.length})
                    </p>
                    <CandidateForkPanel
                      key={`candidates-${step.id}`}
                      step={step}
                      runId={runId}
                      candidates={explanation.fix_candidates}
                      onForked={onForked}
                    />
                  </section>
                )}

                <section>
                  <p className="data mb-2 uppercase tracking-[0.08em] text-muted">
                    Fork &amp; replay suffix
                  </p>
                  {!isFlagged ? (
                    <p className="data text-muted">
                      Forking patches the flagged step and replays everything after it, so this
                      action is offered there.
                    </p>
                  ) : (
                    <ForkPanel key={step.id} step={step} runId={runId} onForked={onForked} />
                  )}
                </section>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

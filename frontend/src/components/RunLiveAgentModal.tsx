/**
 * Run Live Agent: pick a task and a fault, generate a fresh run, open it.
 *
 * Backed by POST /demo/run-live, which synthesizes a trace with the requested
 * fault injected, persists it and scores it with the real Hybrid Sentry
 * Engine before responding. So the run this opens is genuinely new and
 * genuinely diagnosed, not a canned one.
 *
 * Two notes on the stage readout, because it would be easy to make it lie:
 *
 * * The stages name what the backend actually does. An earlier draft of this
 *   spec said "Streaming OpenTelemetry Telemetry", which would have been
 *   false: /demo/run-live calls the synthetic generator, it does not receive
 *   OTel spans. The OTel path is a separate endpoint.
 * * The stages advance on a timer while the real request is in flight, and
 *   the modal waits for BOTH the request and a short floor before closing.
 *   The floor exists so the panel does not flash past unreadably, not to
 *   fake work. If the request takes longer than the floor, it simply takes
 *   longer, and the last stage stays up until it returns.
 */

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../api/client';
import { useToast } from '../hooks/useToast';
import type { DemoOptions, DemoRunResponse } from '../types/api';

/** Each stage names a real phase of what /demo/run-live is doing. */
const STAGES = [
  'Synthesizing trace with injected fault',
  'Extracting 10 column metric vectors',
  'Running local ML localization',
] as const;

const STAGE_MS = 500;
/** Minimum time the loader stays up, so it is readable rather than a flash. */
const FLOOR_MS = STAGES.length * STAGE_MS;

const pretty = (value: string) => value.replace(/_/g, ' ');

export interface RunLiveAgentModalProps {
  open: boolean;
  onClose: () => void;
  /** Called with the new run id once the run exists and is diagnosed. */
  onCreated: (runId: string) => void;
}

export function RunLiveAgentModal({ open, onClose, onCreated }: RunLiveAgentModalProps) {
  const reduced = useReducedMotion();
  const toast = useToast();

  const [options, setOptions] = useState<DemoOptions | null>(null);
  const [agentType, setAgentType] = useState('');
  const [glitch, setGlitch] = useState('');
  const [running, setRunning] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const dialogRef = useRef<HTMLDivElement | null>(null);

  // Options come from the generator itself, so this never drifts from what
  // the backend will actually accept.
  // Derived, not stored: options are in flight exactly while the modal is
  // open and neither a result nor an error has arrived. Deriving it during
  // render avoids a setState inside the effect, which would start a second
  // render pass purely to flip a boolean the existing state already implies.
  const loadingOptions = open && options === null && error === null;

  useEffect(() => {
    if (!open || options !== null) return;
    const controller = new AbortController();

    api
      .getDemoOptions(controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setOptions(result);
        setAgentType((current) => current || result.agent_types[0] || '');
        setGlitch((current) => current || result.glitches[0] || '');
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : String(err));
      });

    return () => controller.abort();
  }, [open, options]);

  // Escape closes, but never mid-run: cancelling the dialog would not cancel
  // the run the server is already committing.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !running) onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, running, onClose]);

  useEffect(() => {
    if (open) dialogRef.current?.focus();
  }, [open]);

  const execute = useCallback(async () => {
    setRunning(true);
    setStage(0);
    setError(null);

    const ticker = setInterval(
      () => setStage((s) => Math.min(s + 1, STAGES.length - 1)),
      STAGE_MS,
    );
    const startedAt = performance.now();

    try {
      const floor = new Promise((resolve) => setTimeout(resolve, reduced ? 0 : FLOOR_MS));
      const [result] = (await Promise.all([
        api.runLiveDemo({ agent_type: agentType, glitch: glitch || null }),
        floor,
      ])) as [DemoRunResponse, unknown];

      const elapsedMs = Math.round(performance.now() - startedAt);

      // Real elapsed time and the real outcome. The endpoint reports whether
      // the engine landed on the step the generator actually broke, so this
      // never claims a correct localization that did not happen.
      const verdict =
        result.localized_correctly === null
          ? 'clean run, no fault to localize'
          : result.localized_correctly
            ? `localized correctly in ${elapsedMs} ms`
            : `missed the true step, flagged ${result.flagged_step_index}`;

      toast.push(result.localized_correctly === false ? 'info' : 'success', `Live run created. ${verdict}`);
      onCreated(result.run_id);
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      clearInterval(ticker);
      setRunning(false);
    }
  }, [agentType, glitch, onClose, onCreated, reduced, toast]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="backdrop"
            className="fixed inset-0 z-40 bg-bg/80 backdrop-blur-sm"
            onClick={() => !running && onClose()}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
          />

          <motion.div
            key="dialog"
            ref={dialogRef}
            tabIndex={-1}
            role="dialog"
            aria-modal="true"
            aria-label="Run a live agent"
            className="hairline glow-accent fixed left-1/2 top-1/2 z-50 w-[min(92vw,30rem)] -translate-x-1/2 -translate-y-1/2 bg-panel p-5 outline-none"
            initial={reduced ? { opacity: 0 } : { opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduced ? { opacity: 0 } : { opacity: 0, y: 12, scale: 0.98 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="card-title">Run live agent</h2>
                <p className="data mt-1 leading-relaxed text-muted">
                  Generates a new trace with the fault you pick, stores it, and scores it with
                  the hybrid engine. The run is real and appears in the runs table.
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                disabled={running}
                aria-label="Close"
                className="data hairline shrink-0 rounded p-1.5 text-muted transition-colors hover:border-accent/60 hover:text-accent disabled:opacity-40"
              >
                <svg viewBox="0 0 16 16" className="size-3.5" aria-hidden="true">
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

            {error && (
              <p
                role="alert"
                className="data mt-4 border-l-2 border-critical/60 pl-3 leading-relaxed text-critical"
              >
                {error}
              </p>
            )}

            {running ? (
              <div className="hairline mt-5 bg-bg p-4">
                {STAGES.map((label, index) => {
                  const done = index < stage;
                  const active = index === stage;
                  return (
                    <p
                      key={label}
                      className={`data leading-relaxed ${
                        done ? 'text-pass' : active ? 'text-accent' : 'text-muted/50'
                      }`}
                    >
                      [ {label}
                      {done ? ' ok' : active ? '...' : ''} ]
                    </p>
                  );
                })}
              </div>
            ) : (
              <div className="mt-5 flex flex-col gap-4">
                <div>
                  <label htmlFor="demo-agent" className="data block text-muted">
                    Agent type
                  </label>
                  <select
                    id="demo-agent"
                    value={agentType}
                    disabled={loadingOptions || !options}
                    onChange={(event) => setAgentType(event.target.value)}
                    className="data mt-1.5 w-full rounded border border-border bg-bg px-3 py-2 text-text outline-none disabled:opacity-50"
                  >
                    {(options?.agent_types ?? []).map((value) => (
                      <option key={value} value={value} className="bg-bg text-text">
                        {pretty(value)}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="demo-glitch" className="data block text-muted">
                    Injected glitch
                  </label>
                  <select
                    id="demo-glitch"
                    value={glitch}
                    disabled={loadingOptions || !options}
                    onChange={(event) => setGlitch(event.target.value)}
                    className="data mt-1.5 w-full rounded border border-border bg-bg px-3 py-2 text-text outline-none disabled:opacity-50"
                  >
                    {(options?.glitches ?? []).map((value) => (
                      <option key={value} value={value} className="bg-bg text-text">
                        {pretty(value)}
                      </option>
                    ))}
                  </select>
                  <p className="data mt-2 leading-relaxed text-muted">
                    infinite loop and context truncation were held out of training, so the
                    classifier has never seen them. Those are the interesting ones.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() => void execute()}
                  disabled={loadingOptions || !options || !agentType}
                  className="hairline glow-accent w-full rounded px-3 py-2 text-[13px] text-accent transition-colors hover:border-accent disabled:opacity-40"
                >
                  {loadingOptions ? 'Loading options' : 'Execute live trace'}
                </button>
              </div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

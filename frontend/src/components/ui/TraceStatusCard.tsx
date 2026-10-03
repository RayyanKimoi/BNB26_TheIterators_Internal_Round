/**
 * Hero status card, styled after a flight-status widget.
 *
 * It loops a SIMULATED trace: steps tick in, the detector flags step 7, then a
 * fork flips the run from FAIL to SUCCESS. Labelled SIMULATED so it never reads
 * as a live metric. Status swaps cross-fade (opacity) and cells scale, so no
 * layout property is animated. Under reduced motion it holds one static frame.
 */

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useEffect, useState } from 'react';

const SCORES = [2, 3, 1, 5, 2, 4, 9, 88, 11, 6, 3, 2];
const FLAGGED = 7;
const STEP_MS = 230;
const PERIOD = SCORES.length + 8;

function cellColor(score: number, flaggedFixed: boolean): string {
  if (flaggedFixed) return 'bg-pass';
  if (score >= 40) return 'bg-critical';
  if (score >= 8) return 'bg-warn';
  if (score >= 4) return 'bg-pass/50';
  return 'bg-border';
}

export function TraceStatusCard() {
  const reduced = useReducedMotion();
  const [pos, setPos] = useState(reduced ? SCORES.length : 0);

  useEffect(() => {
    if (reduced) return;
    const id = setInterval(() => setPos((p) => (p + 1) % PERIOD), STEP_MS);
    return () => clearInterval(id);
  }, [reduced]);

  const fillCount = Math.min(pos, SCORES.length);
  const fixed = pos > SCORES.length;
  const durationMs = 120 + fillCount * 164;

  return (
    <div className="hairline w-full max-w-lg bg-panel/85 p-5 backdrop-blur">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="data rounded border border-border px-1.5 py-0.5 text-muted">SIMULATED</span>
          <span className="data text-text">trace f40b06ef</span>
        </div>
        <AnimatePresence mode="wait" initial={false}>
          {fixed ? (
            <motion.span
              key="ok"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="data inline-flex items-center gap-1.5 rounded border border-pass/40 bg-pass/10 px-1.5 py-0.5 text-pass"
            >
              <span className="block size-1.5 rounded-full bg-pass shadow-[0_0_6px_currentColor]" />
              SUCCESS
            </motion.span>
          ) : (
            <motion.span
              key="fail"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="data inline-flex items-center gap-1.5 rounded border border-critical/40 bg-critical/10 px-1.5 py-0.5 text-critical"
            >
              <span className="block size-1.5 rounded-full bg-critical shadow-[0_0_6px_currentColor]" />
              FAIL
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      <div className="mt-4 flex items-center justify-between">
        <span className="data text-muted">Per-step blame</span>
        <span className="data text-muted">{durationMs} ms</span>
      </div>
      <div className="mt-2 flex items-end gap-1" style={{ height: 48 }}>
        {SCORES.map((score, i) => {
          const shown = i < fillCount;
          const isFlagged = i === FLAGGED;
          const height = 10 + Math.round((score / 88) * 34);
          return (
            <div key={i} className="relative flex-1" style={{ height }} title={`step ${i}: ${score}`}>
              <motion.div
                className={`absolute inset-0 origin-bottom rounded-sm ${cellColor(score, isFlagged && fixed)}`}
                initial={false}
                animate={{ scaleY: shown ? 1 : 0.12, opacity: shown ? 1 : 0.3 }}
                transition={{ duration: 0.25, ease: 'easeOut' }}
                style={
                  isFlagged && !fixed && shown
                    ? { boxShadow: '0 0 10px #e0574c' }
                    : undefined
                }
              />
            </div>
          );
        })}
      </div>
      <div className="mt-1.5 flex items-center justify-between">
        <span className="data text-muted">step 0</span>
        <span className={`data ${fixed ? 'text-pass' : 'text-critical'}`}>
          {fixed ? 'fork replayed' : `flagged: step ${FLAGGED}`}
        </span>
        <span className="data text-muted">step {SCORES.length - 1}</span>
      </div>

      <div className="hairline mt-4 bg-bg p-3">
        <p className="data text-muted">step[7].input.payload</p>
        <pre className="data mt-1.5 overflow-x-auto leading-relaxed">
          <span className="text-critical">- &quot;currency&quot;: &quot;USD&quot;</span>
          {'\n'}
          <span className="text-pass">+ &quot;currency&quot;: &quot;EUR&quot;</span>
        </pre>
      </div>

      <div className="mt-3 flex items-center justify-between">
        <span className="data text-muted">schema_violation</span>
        <span className="data text-pass">fork replay: FAIL to SUCCESS</span>
      </div>
    </div>
  );
}

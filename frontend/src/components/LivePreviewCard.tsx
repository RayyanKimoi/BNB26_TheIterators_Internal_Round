/**
 * Landing-page teaser styled after a status card: a compact blame heatmap over a
 * sample trace, plus the JSON diff a fork would apply. It is explicitly a SAMPLE,
 * labelled as such, so it never reads as a live metric. The real version of this
 * view is the Step 3 trace inspector.
 */

import { motion, useReducedMotion } from 'framer-motion';

const SAMPLE_SCORES = [2, 3, 1, 5, 2, 4, 9, 88, 11, 6, 3, 2];
const FLAGGED_INDEX = 7;

function cellClass(score: number): string {
  if (score >= 40) return 'bg-critical/80';
  if (score >= 8) return 'bg-warn/70';
  if (score >= 4) return 'bg-pass/40';
  return 'bg-border';
}

export function LivePreviewCard() {
  const reduced = useReducedMotion();

  return (
    <motion.div
      initial={reduced ? { opacity: 0 } : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut', delay: 0.15 }}
      className="hairline w-full max-w-md bg-panel/90 p-4 backdrop-blur"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="data rounded border border-border px-1.5 py-0.5 text-muted">SAMPLE</span>
          <span className="data text-text">trace f40b06ef</span>
        </div>
        <span className="data inline-flex items-center gap-1.5 rounded border border-critical/40 bg-critical/10 px-1.5 py-0.5 text-critical">
          <span className="block size-1.5 rounded-full bg-critical shadow-[0_0_6px_currentColor]" />
          FAIL
        </span>
      </div>

      <p className="data mt-4 text-muted">Per-step blame</p>
      <div className="mt-2 flex items-end gap-1">
        {SAMPLE_SCORES.map((score, i) => {
          const flagged = i === FLAGGED_INDEX;
          const height = 10 + Math.round((score / 88) * 34);
          return (
            <div key={i} className="group relative flex-1" title={`step ${i}: score ${score}`}>
              <motion.div
                className={`w-full rounded-sm ${cellClass(score)}`}
                style={{ height }}
                animate={
                  flagged && !reduced
                    ? { boxShadow: ['0 0 0px #e0574c', '0 0 12px #e0574c', '0 0 0px #e0574c'] }
                    : undefined
                }
                transition={flagged && !reduced ? { duration: 1.6, repeat: Infinity } : undefined}
              />
            </div>
          );
        })}
      </div>
      <div className="mt-1.5 flex items-center justify-between">
        <span className="data text-muted">step 0</span>
        <span className="data text-critical">flagged: step {FLAGGED_INDEX}</span>
        <span className="data text-muted">step {SAMPLE_SCORES.length - 1}</span>
      </div>

      <p className="data mt-4 text-muted">Proposed fork fix</p>
      <div className="hairline mt-2 bg-bg p-3">
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
    </motion.div>
  );
}

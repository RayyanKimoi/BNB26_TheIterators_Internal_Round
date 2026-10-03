/**
 * High-density runs table.
 *
 * Rows animate in and out with layout so filtering and sorting feel continuous
 * rather than a hard swap. All identifiers and counts are rendered in the mono
 * data style per the project rules. Nothing is fabricated: step counts and fault
 * classes come straight off the run summary.
 */

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

import { CopyableId } from './CopyableId';
import type { RunSummary } from '../types/api';

const COLUMNS = ['Run ID', 'Origin', 'Task', 'Status', 'Steps', 'Fault Class', ''] as const;

function OriginBadge({ isFork, forkStep }: { isFork: boolean; forkStep: number | null }) {
  if (isFork) {
    return (
      <span
        className="data inline-flex items-center gap-1 rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 text-accent"
        title={forkStep !== null ? `Forked at step ${forkStep}` : 'Fork'}
      >
        fork{forkStep !== null ? ` @${forkStep}` : ''}
      </span>
    );
  }
  return (
    <span className="data inline-flex items-center rounded border border-border px-1.5 py-0.5 text-muted">
      original
    </span>
  );
}

function StatusBadge({ status }: { status: string }) {
  const ok = status === 'success';
  return (
    <span
      className={`data inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 ${
        ok
          ? 'border border-pass/40 bg-pass/10 text-pass'
          : 'border border-critical/40 bg-critical/10 text-critical'
      }`}
    >
      <span
        className={`block size-1.5 rounded-full ${ok ? 'bg-pass' : 'bg-critical'} shadow-[0_0_6px_currentColor]`}
        aria-hidden="true"
      />
      {ok ? 'SUCCESS' : 'FAIL'}
    </span>
  );
}

function SkeletonRows({ rows = 6 }: { rows?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i} className="border-b border-border">
          {COLUMNS.map((_col, c) => (
            <td key={c} className="px-3 py-2.5">
              <div className="h-3.5 animate-pulse rounded bg-border" style={{ width: `${40 + ((c * 17) % 50)}%` }} />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

export interface RunsTableProps {
  runs: RunSummary[];
  loading?: boolean;
  onInspect: (run: RunSummary) => void;
}

export function RunsTable({ runs, loading = false, onInspect }: RunsTableProps) {
  const reduced = useReducedMotion();

  return (
    <div className="hairline overflow-x-auto bg-panel">
      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b border-border">
            {COLUMNS.map((col, i) => (
              <th
                key={col || `col-${i}`}
                className="data whitespace-nowrap px-3 py-2.5 font-medium uppercase tracking-[0.08em] text-muted"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <SkeletonRows />
          ) : (
            <AnimatePresence initial={false}>
              {runs.map((run) => {
                const isFork = run.parent_run_id !== null;
                const fault = run.injected_class ?? (run.status === 'success' ? 'Clean' : null);
                return (
                  <motion.tr
                    key={run.id}
                    layout={!reduced}
                    initial={reduced ? { opacity: 0 } : { opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={reduced ? { opacity: 0 } : { opacity: 0, y: -4 }}
                    transition={{ duration: 0.18, ease: 'easeOut' }}
                    className="border-b border-border transition-colors hover:bg-text/[0.03]"
                  >
                    <td className="px-3 py-2.5">
                      <CopyableId id={run.id} displayLength={8} />
                    </td>
                    <td className="px-3 py-2.5">
                      <OriginBadge isFork={isFork} forkStep={run.forked_at_step} />
                    </td>
                    <td className="px-3 py-2.5">
                      <span className="data rounded border border-border px-1.5 py-0.5 text-text">
                        {run.task_type}
                      </span>
                    </td>
                    <td className="px-3 py-2.5">
                      <StatusBadge status={run.status} />
                    </td>
                    <td className="data px-3 py-2.5 text-text">{run.total_steps}</td>
                    <td className="px-3 py-2.5">
                      {fault ? (
                        <span
                          className={`data ${fault === 'Clean' ? 'text-muted' : 'text-warn'}`}
                        >
                          {fault}
                        </span>
                      ) : (
                        <span className="data text-muted">&ndash;</span>
                      )}
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      <button
                        type="button"
                        onClick={() => onInspect(run)}
                        className="data hairline whitespace-nowrap rounded px-2.5 py-1 text-text transition-colors hover:border-accent/60 hover:text-accent"
                      >
                        Inspect Trace
                      </button>
                    </td>
                  </motion.tr>
                );
              })}
            </AnimatePresence>
          )}
        </tbody>
      </table>
    </div>
  );
}

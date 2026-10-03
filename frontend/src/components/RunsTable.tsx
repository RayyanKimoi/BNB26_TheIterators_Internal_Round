/**
 * High-density runs table.
 *
 * Rows animate in and out with layout so filtering and sorting feel continuous
 * rather than a hard swap. All identifiers and counts are rendered in the mono
 * data style per the project rules. Nothing is fabricated: step counts and fault
 * classes come straight off the run summary.
 */

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useState } from 'react';

import type { RunSummary } from '../types/api';

const COLUMNS = ['Run ID', 'Origin', 'Task', 'Status', 'Steps', 'Fault Class', ''] as const;

function CopyableId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  const reduced = useReducedMotion();

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(id);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // Clipboard can be unavailable over plain http; fail quietly.
    }
  };

  return (
    <button
      type="button"
      onClick={copy}
      title={`Copy ${id}`}
      className="data group inline-flex items-center gap-1.5 text-text transition-colors hover:text-accent"
    >
      <span>{id.slice(0, 8)}</span>
      <AnimatePresence mode="wait" initial={false}>
        {copied ? (
          <motion.span
            key="done"
            initial={reduced ? { opacity: 0 } : { opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16 }}
            className="inline-flex items-center gap-1 text-accent"
          >
            <svg viewBox="0 0 16 16" className="size-3" aria-hidden="true">
              <path
                d="M3.5 8.5l3 3 6-7"
                stroke="currentColor"
                strokeWidth="1.6"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            copied
          </motion.span>
        ) : (
          <motion.span
            key="icon"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16 }}
            className="text-muted opacity-0 transition-opacity group-hover:opacity-100"
          >
            <svg viewBox="0 0 16 16" className="size-3" aria-hidden="true">
              <rect x="5" y="5" width="8" height="8" rx="1" stroke="currentColor" strokeWidth="1.3" fill="none" />
              <path d="M3 11V3h8" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinecap="round" />
            </svg>
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );
}

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
                      <CopyableId id={run.id} />
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

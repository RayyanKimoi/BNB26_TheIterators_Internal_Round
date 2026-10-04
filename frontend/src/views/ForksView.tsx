/**
 * Forks tab: fork lineage, and a comparison of any two runs.
 *
 * The lineage tree is built client side from the run list's `parent_run_id`
 * links — the API already returns everything needed, so this costs one
 * request rather than a new endpoint. Only roots that actually have children
 * are shown: a tree of 250 childless runs would be a list, not a lineage.
 *
 * The comparison picker uses GET /runs/{id}/compare/{other_id}, which diffs
 * any two runs, not only a parent and its fork.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';

import { api } from '../api/client';
import { EmptyState, ErrorState } from '../components/AppShell';
import { CopyableId } from '../components/CopyableId';
import type { CompareResponse, RunSummary } from '../types/api';

interface LineageNode {
  run: RunSummary;
  children: RunSummary[];
}

function StatusPill({ status }: { status: string }) {
  const ok = status === 'success';
  return (
    <span
      className={`data inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 ${
        ok ? 'border-pass/40 bg-pass/10 text-pass' : 'border-critical/40 bg-critical/10 text-critical'
      }`}
    >
      <span
        className={`block size-1.5 rounded-full ${ok ? 'bg-pass' : 'bg-critical'} shadow-[0_0_6px_currentColor]`}
      />
      {ok ? 'SUCCESS' : 'FAIL'}
    </span>
  );
}

function ForksSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <div className="hairline bg-panel p-4">
        <div className="h-2.5 w-32 animate-pulse rounded bg-border" />
        <div className="mt-4 flex flex-col gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-10 animate-pulse rounded bg-border" />
          ))}
        </div>
      </div>
    </div>
  );
}

function DiffTable({ comparison }: { comparison: CompareResponse }) {
  const changed = comparison.diffs.filter(
    (d) => d.tool_changed || d.error_flag_changed || d.output_changed || !d.a_present || !d.b_present,
  );

  return (
    <div className="hairline overflow-x-auto bg-panel">
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 pt-4">
        <p className="data text-muted">
          {comparison.steps_compared} steps aligned · {changed.length} differ
        </p>
        <div className="flex items-center gap-3">
          <span className="data text-muted">
            A <StatusPill status={comparison.run_a_status} />
          </span>
          <span className="data text-muted">
            B <StatusPill status={comparison.run_b_status} />
          </span>
        </div>
      </div>

      {changed.length === 0 ? (
        <p className="data px-4 py-6 text-muted">
          These two runs are step-for-step identical in tool, error flag and output.
        </p>
      ) : (
        <table className="mt-3 w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-border">
              {['Step', 'A tool', 'B tool', 'A error', 'B error', 'Changed output keys'].map((h) => (
                <th
                  key={h}
                  className="data whitespace-nowrap px-4 py-2.5 font-medium uppercase tracking-[0.08em] text-muted"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {changed.map((d) => (
              <tr key={d.step_index} className="border-b border-border last:border-0">
                <td className="data px-4 py-2.5 text-text">{d.step_index}</td>
                <td className={`data px-4 py-2.5 ${d.tool_changed ? 'text-warn' : 'text-muted'}`}>
                  {d.a_present ? (d.a_tool_name ?? '–') : 'absent'}
                </td>
                <td className={`data px-4 py-2.5 ${d.tool_changed ? 'text-warn' : 'text-muted'}`}>
                  {d.b_present ? (d.b_tool_name ?? '–') : 'absent'}
                </td>
                <td
                  className={`data px-4 py-2.5 ${d.a_error_flag ? 'text-critical' : 'text-muted'}`}
                >
                  {d.a_present ? String(d.a_error_flag) : '–'}
                </td>
                <td
                  className={`data px-4 py-2.5 ${
                    d.error_flag_changed && d.b_error_flag === false ? 'text-pass' : 'text-muted'
                  }`}
                >
                  {d.b_present ? String(d.b_error_flag) : '–'}
                </td>
                <td className="data px-4 py-2.5 text-muted">
                  {d.changed_output_keys.length ? d.changed_output_keys.join(', ') : '–'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export interface ForksViewProps {
  onOpenTrace: (runId: string) => void;
}

export function ForksView({ onOpenTrace }: ForksViewProps) {
  const reduced = useReducedMotion();

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  const [runA, setRunA] = useState('');
  const [runB, setRunB] = useState('');
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .listRuns({}, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setRuns(data);
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err : new Error(String(err)));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [reloadKey]);

  const lineage = useMemo<LineageNode[]>(() => {
    const byId = new Map(runs.map((r) => [r.id, r]));
    const childrenOf = new Map<string, RunSummary[]>();
    for (const run of runs) {
      if (!run.parent_run_id) continue;
      const list = childrenOf.get(run.parent_run_id) ?? [];
      list.push(run);
      childrenOf.set(run.parent_run_id, list);
    }
    return [...childrenOf.entries()]
      .map(([parentId, children]) => {
        const parent = byId.get(parentId);
        return parent ? { run: parent, children } : null;
      })
      .filter((node): node is LineageNode => node !== null);
  }, [runs]);

  const runComparison = async () => {
    if (!runA || !runB) return;
    setComparing(true);
    setCompareError(null);
    try {
      setComparison(await api.compareRuns(runA, runB));
    } catch (err) {
      setCompareError(err instanceof Error ? err.message : 'Could not compare these runs.');
    } finally {
      setComparing(false);
    }
  };

  if (loading) return <ForksSkeleton />;
  if (error) {
    return (
      <ErrorState
        error={error}
        onRetry={() => {
          setError(null);
          setLoading(true);
          setReloadKey((k) => k + 1);
        }}
      />
    );
  }

  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.08 } },
  };
  const item: Variants = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
  };

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-4">
      <motion.div variants={item} className="hairline bg-panel p-4">
        <p className="data text-muted">Fork lineage</p>
        {lineage.length === 0 ? (
          <p className="data mt-3 text-muted">
            No forks yet. Open a failed run, inspect the flagged step, and fork it to create one.
          </p>
        ) : (
          <ol className="mt-4 flex flex-col gap-5">
            {lineage.map((node) => (
              <li key={node.run.id} className="relative">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="block size-2.5 rounded-full bg-critical shadow-[0_0_6px_var(--color-critical)]" />
                  <CopyableId id={node.run.id} displayLength={8} />
                  <span className="data rounded border border-border px-1.5 py-0.5 text-text">
                    {node.run.task_type}
                  </span>
                  <StatusPill status={node.run.status} />
                  <span className="data text-muted">{node.run.injected_class ?? 'clean'}</span>
                  <button
                    type="button"
                    onClick={() => onOpenTrace(node.run.id)}
                    className="data hairline ml-auto rounded px-2.5 py-1 text-text transition-colors hover:border-accent/60 hover:text-accent"
                  >
                    Open parent
                  </button>
                </div>

                <ul className="mt-2 flex flex-col gap-2 border-l border-border pl-6">
                  {node.children.map((child) => (
                    <li key={child.id} className="flex flex-wrap items-center gap-3">
                      <span
                        className={`block size-2 rounded-full ${
                          child.status === 'success'
                            ? 'bg-pass shadow-[0_0_6px_var(--color-pass)]'
                            : 'bg-critical'
                        }`}
                      />
                      <CopyableId id={child.id} displayLength={8} />
                      <span className="data text-muted">
                        forked at step {child.forked_at_step ?? '–'}
                      </span>
                      <StatusPill status={child.status} />
                      {node.run.status === 'failed' && child.status === 'success' && (
                        <span className="data rounded border border-pass/40 bg-pass/10 px-1.5 py-0.5 text-pass">
                          flipped to SUCCESS
                        </span>
                      )}
                      <button
                        type="button"
                        onClick={() => onOpenTrace(child.id)}
                        className="data hairline ml-auto rounded px-2.5 py-1 text-text transition-colors hover:border-accent/60 hover:text-accent"
                      >
                        Open fork
                      </button>
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ol>
        )}
      </motion.div>

      <motion.div variants={item} className="hairline bg-panel p-4">
        <p className="data text-muted">Compare any two runs</p>
        <div className="mt-3 flex flex-col gap-3 lg:flex-row lg:items-end">
          {(
            [
              ['Run A', runA, setRunA] as const,
              ['Run B', runB, setRunB] as const,
            ]
          ).map(([label, value, setValue]) => (
            <label key={label} className="data flex min-w-0 flex-1 flex-col gap-1.5 text-muted">
              {label}
              <select
                value={value}
                onChange={(e) => setValue(e.target.value)}
                className="hairline rounded bg-bg px-2.5 py-2 font-mono text-[12px] text-text focus-visible:outline-none"
              >
                <option value="">select a run</option>
                {runs.map((run) => (
                  <option key={run.id} value={run.id}>
                    {run.id.slice(0, 8)} · {run.task_type} · {run.status}
                    {run.parent_run_id ? ' · fork' : ''}
                  </option>
                ))}
              </select>
            </label>
          ))}
          <button
            type="button"
            onClick={runComparison}
            disabled={!runA || !runB || comparing}
            className="shrink-0 rounded border border-accent bg-accent/10 px-4 py-2 text-[13px] font-medium text-accent transition-colors hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {comparing ? 'Comparing...' : 'Compare'}
          </button>
        </div>
        {runA && runA === runB && (
          <p className="data mt-2 text-warn">Pick two different runs to see a meaningful diff.</p>
        )}
        {compareError && (
          <p className="data mt-2 text-critical" role="alert">
            {compareError}
          </p>
        )}
      </motion.div>

      {comparison && (
        <motion.div variants={item}>
          <DiffTable comparison={comparison} />
        </motion.div>
      )}

      {!comparison && lineage.length === 0 && (
        <EmptyState
          title="Nothing to compare yet"
          hint="Fork a run from the trace inspector, or pick any two runs above."
        />
      )}
    </motion.div>
  );
}

/**
 * Runs dashboard: metrics header, filter bar, and the runs table.
 *
 * Fetches the run list once and filters it in the browser, so the header counts
 * and the table stay in sync with the filter bar without extra requests. The two
 * accuracy figures in the header are measured model results, verified against
 * model/artifacts/evaluation.json; see the constants below.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';

import { api, ApiError } from '../api/client';
import { ErrorState, EmptyState } from '../components/AppShell';
import { FilterBar } from '../components/FilterBar';
import { MetricsHeader } from '../components/MetricsHeader';
import { RunsTable } from '../components/RunsTable';
import { CLEAN_CLASS, EMPTY_FILTERS } from '../config/runFilters';
import type { RunTableFilters } from '../config/runFilters';
import { useToast } from '../hooks/useToast';
import type { RunSummary } from '../types/api';
import { TraceView } from './TraceView';

/*
 * Measured on the held-out evaluation, not estimated. Sourced from
 * model/artifacts/evaluation.json: test_seen_top1 = 0.95 (trained-class top-1)
 * and hybrid_heldout_top1 = 0.525 (hybrid engine on held-out classes). The
 * Model Benchmarks tab reads these live from GET /model/evaluation; here they
 * are pinned so the dashboard header needs no second request on load.
 */
const TRAINED_CLASS_ACCURACY = 95.0;
const HYBRID_ENGINE_ACCURACY = 52.5;

function distinct(values: (string | null)[]): string[] {
  return [...new Set(values.filter((v): v is string => v !== null))].sort();
}

export function RunsView() {
  const toast = useToast();
  const reduced = useReducedMotion();

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  const [filters, setFilters] = useState<RunTableFilters>(EMPTY_FILTERS);
  const [inspectingId, setInspectingId] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    api
      .listRuns({}, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setRuns(data);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        const normalized = err instanceof Error ? err : new Error(String(err));
        setError(normalized);
        const hint =
          err instanceof ApiError && err.isNetworkError
            ? 'Start it with: uvicorn backend.main:app --reload'
            : normalized.message;
        toast.push('error', `Could not load runs. ${hint}`);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
    // reloadKey drives manual retry; toast identity is stable from the shell.
  }, [reloadKey, toast]);

  const taskTypes = useMemo(() => distinct(runs.map((r) => r.task_type)), [runs]);
  const faultClasses = useMemo(() => distinct(runs.map((r) => r.injected_class)), [runs]);
  const hasClean = useMemo(() => runs.some((r) => r.injected_class === null), [runs]);

  const filtered = useMemo(() => {
    const query = filters.search.trim().toLowerCase();
    return runs.filter((run) => {
      if (filters.status !== 'all' && run.status !== filters.status) return false;
      if (filters.taskType !== 'all' && run.task_type !== filters.taskType) return false;
      if (filters.faultClass === CLEAN_CLASS) {
        if (run.injected_class !== null) return false;
      } else if (filters.faultClass !== 'all' && run.injected_class !== filters.faultClass) {
        return false;
      }
      if (query) {
        const haystack = [
          run.id,
          run.task_type,
          run.status,
          run.source,
          run.injected_class ?? '',
        ]
          .join(' ')
          .toLowerCase();
        if (!haystack.includes(query)) return false;
      }
      return true;
    });
  }, [runs, filters]);

  const detectionRate = useMemo(() => {
    const failed = runs.filter((r) => r.status === 'failed');
    if (failed.length === 0) return null;
    const diagnosed = failed.filter((r) => r.has_diagnosis).length;
    return (diagnosed / failed.length) * 100;
  }, [runs]);

  const onInspect = (run: RunSummary) => {
    setInspectingId(run.id);
  };

  const retry = () => {
    // Reset from the event that triggered the reload, not inside the effect,
    // so the fetch effect never calls setState synchronously on render.
    setError(null);
    setLoading(true);
    setReloadKey((k) => k + 1);
  };

  if (inspectingId) {
    return (
      <TraceView
        runId={inspectingId}
        onBack={() => setInspectingId(null)}
        onNavigateToRun={setInspectingId}
      />
    );
  }

  if (error) {
    return <ErrorState error={error} onRetry={retry} />;
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
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="flex flex-col gap-4"
    >
      <MetricsHeader
        totalRuns={runs.length}
        trainedAccuracy={TRAINED_CLASS_ACCURACY}
        hybridAccuracy={HYBRID_ENGINE_ACCURACY}
        detectionRate={detectionRate}
        loading={loading}
      />

      <motion.div variants={item}>
        <FilterBar
          filters={filters}
          onChange={setFilters}
          taskTypes={taskTypes}
          faultClasses={faultClasses}
          hasClean={hasClean}
          shown={filtered.length}
          total={runs.length}
        />
      </motion.div>

      <motion.div variants={item}>
        {!loading && runs.length === 0 ? (
          <EmptyState
            title="No runs in the database yet"
            hint="Load the synthetic corpus into Supabase, then reload. GET /runs returned an empty list."
          />
        ) : !loading && filtered.length === 0 ? (
          <EmptyState title="No runs match these filters" hint="Clear a filter or widen the search." />
        ) : (
          <RunsTable runs={filtered} loading={loading} onInspect={onInspect} />
        )}
      </motion.div>
    </motion.div>
  );
}

/**
 * Client-side filter bar for the runs table.
 *
 * Every control filters the already-loaded run list in the browser, so there is
 * no network round trip as the user narrows down. The active option in each
 * group is marked by a single shared highlight that slides between options
 * (Framer `layoutId`), which reads as one moving element rather than several
 * fading in and out.
 */

import { motion, useReducedMotion } from 'framer-motion';

import { CLEAN_CLASS, EMPTY_FILTERS } from '../config/runFilters';
import type { RunTableFilters } from '../config/runFilters';
import { AnimatedNumber } from './AnimatedNumber';

interface Option {
  value: string;
  label: string;
}

function PillGroup({
  label,
  layoutId,
  options,
  value,
  onChange,
}: {
  label: string;
  layoutId: string;
  options: Option[];
  value: string;
  onChange: (value: string) => void;
}) {
  const reduced = useReducedMotion();

  return (
    <div className="flex items-center gap-2">
      <span className="data shrink-0 text-muted">{label}</span>
      <div className="hairline flex flex-wrap items-center gap-0.5 rounded bg-bg p-0.5">
        {options.map((opt) => {
          const active = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onChange(opt.value)}
              aria-pressed={active}
              className="relative rounded px-2.5 py-1 text-[12px] transition-colors"
            >
              {active && (
                <motion.span
                  layoutId={layoutId}
                  className="absolute inset-0 rounded border border-accent/40 bg-accent/12"
                  transition={{ duration: reduced ? 0 : 0.2, ease: 'easeOut' }}
                />
              )}
              <span
                className={`relative z-10 ${active ? 'text-accent' : 'text-muted hover:text-text'}`}
              >
                {opt.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export interface FilterBarProps {
  filters: RunTableFilters;
  onChange: (filters: RunTableFilters) => void;
  taskTypes: string[];
  faultClasses: string[];
  /** Whether any loaded run has a null injected_class, i.e. needs a Clean pill. */
  hasClean: boolean;
  shown: number;
  total: number;
}

export function FilterBar({
  filters,
  onChange,
  taskTypes,
  faultClasses,
  hasClean,
  shown,
  total,
}: FilterBarProps) {
  const set = (patch: Partial<RunTableFilters>) => onChange({ ...filters, ...patch });

  const statusOptions: Option[] = [
    { value: 'all', label: 'All' },
    { value: 'success', label: 'SUCCESS' },
    { value: 'failed', label: 'FAIL' },
  ];

  const taskOptions: Option[] = [
    { value: 'all', label: 'All' },
    ...taskTypes.map((t) => ({ value: t, label: t })),
  ];

  const classOptions: Option[] = [
    { value: 'all', label: 'All' },
    ...faultClasses.map((c) => ({ value: c, label: c })),
    ...(hasClean ? [{ value: CLEAN_CLASS, label: 'Clean' }] : []),
  ];

  return (
    <div className="hairline bg-panel px-4 py-3">
      <div className="flex flex-col gap-3 lg:flex-row lg:flex-wrap lg:items-center">
        <PillGroup
          label="Status"
          layoutId="pill-status"
          options={statusOptions}
          value={filters.status}
          onChange={(v) => set({ status: v as RunTableFilters['status'] })}
        />

        {taskTypes.length > 0 && (
          <PillGroup
            label="Task"
            layoutId="pill-task"
            options={taskOptions}
            value={filters.taskType}
            onChange={(v) => set({ taskType: v })}
          />
        )}

        {classOptions.length > 1 && (
          <PillGroup
            label="Fault"
            layoutId="pill-class"
            options={classOptions}
            value={filters.faultClass}
            onChange={(v) => set({ faultClass: v })}
          />
        )}

        <div className="flex items-center gap-2 lg:ml-auto">
          <label htmlFor="run-search" className="sr-only">
            Search runs
          </label>
          <input
            id="run-search"
            type="text"
            value={filters.search}
            onChange={(e) => set({ search: e.target.value })}
            placeholder="Search id, task, class"
            className="data hairline w-full rounded bg-bg px-2.5 py-1.5 text-text placeholder:text-muted focus:border-accent/60 focus-visible:outline-none sm:w-56"
          />
        </div>
      </div>

      <div className="mt-2.5 flex items-center justify-between">
        <p className="data text-muted">
          Showing <AnimatedNumber value={shown} durationMs={300} /> of {total} runs
        </p>
        {shown !== total && (
          <button
            type="button"
            onClick={() => onChange(EMPTY_FILTERS)}
            className="data text-muted transition-colors hover:text-accent"
          >
            Clear filters
          </button>
        )}
      </div>
    </div>
  );
}

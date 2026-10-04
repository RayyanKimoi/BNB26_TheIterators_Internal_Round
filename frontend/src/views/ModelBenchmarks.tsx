/**
 * Model Benchmarks: the measured numbers behind the pitch, read live from
 * GET /model/evaluation (model/artifacts/evaluation.json + metrics.json).
 *
 * Every figure on this page is read from that response at render time — none
 * of the four headline numbers are hardcoded, even though their current
 * values are well known from the project's own evaluation runs. If the
 * artifact is regenerated, this page moves with it instead of going stale.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { api } from '../api/client';
import { AnimatedNumber } from '../components/AnimatedNumber';
import { ErrorState } from '../components/AppShell';
import type { EvaluationResponse } from '../types/api';

const CHART = {
  accent: '#7df9c4',
  warn: '#e8b14c',
  critical: '#e0574c',
  pass: '#5bc98c',
  muted: '#8a8a92',
  border: '#1f1f23',
  panel: '#111113',
};

const AXIS = { stroke: CHART.muted, fontSize: 11, fontFamily: "'Geist Mono', monospace" };

/**
 * Recharts types a label value as renderable text, not as a number, so the
 * formatter has to narrow before it can append the unit. Taking `unknown`
 * is deliberate: a wider parameter is still assignable to LabelFormatter.
 */
const pctLabel = (v: unknown) => (typeof v === 'number' ? `${v}%` : '');

const LABEL_STYLE = {
  fill: CHART.muted,
  fontSize: 10,
  fontFamily: "'Geist Mono', monospace",
} as const;

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number | string; color?: string }[];
  label?: string | number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="hairline bg-panel px-2.5 py-1.5 shadow-lg">
      {label !== undefined && <p className="data text-text">{label}</p>}
      {payload.map((entry, i) => (
        <p key={i} className="data text-muted">
          <span style={{ color: entry.color }}>{entry.name}</span>: {entry.value}%
        </p>
      ))}
    </div>
  );
}

/** The seven injected classes and which split each belongs to. Real, from
 * generator/output/manifest.json's held_out_classes — the corpus design, not
 * a guess: five classes are trained on, two are held out entirely. */
const CLASS_SPLIT: Record<string, 'seen' | 'heldout'> = {
  wrong_tool_chosen: 'seen',
  hallucinated_argument: 'seen',
  stale_retrieval: 'seen',
  premature_termination: 'seen',
  schema_violation: 'seen',
  infinite_loop: 'heldout',
  context_truncation: 'heldout',
};

function pct(value: number | null | undefined, decimals = 1): string {
  return value === null || value === undefined ? '–' : `${(value * 100).toFixed(decimals)}%`;
}

function BenchmarkCard({
  label,
  value,
  subtitle,
}: {
  label: string;
  value: number | null;
  subtitle: string;
}) {
  return (
    <div className="hairline bg-panel px-4 py-3.5 transition-[border-color,box-shadow] duration-200 hover:border-accent/60 hover:shadow-[0_0_28px_-10px_var(--color-accent)]">
      <p className="data uppercase tracking-[0.08em] text-muted">{label}</p>
      <p className="mt-1.5 font-display text-[28px] leading-none text-text">
        {value === null ? (
          <span className="text-muted">&ndash;</span>
        ) : (
          <AnimatedNumber value={value * 100} decimals={1} suffix="%" />
        )}
      </p>
      <p className="data mt-2 text-muted">{subtitle}</p>
    </div>
  );
}

function BenchmarksSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="hairline bg-panel px-4 py-3.5">
            <div className="h-2.5 w-24 animate-pulse rounded bg-border" />
            <div className="mt-3 h-7 w-16 animate-pulse rounded bg-border" />
            <div className="mt-2 h-2.5 w-32 animate-pulse rounded bg-border" />
          </div>
        ))}
      </div>
      <div className="hairline bg-panel p-4">
        <div className="h-2.5 w-40 animate-pulse rounded bg-border" />
        <div className="mt-3 flex flex-col gap-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-4 animate-pulse rounded bg-border" />
          ))}
        </div>
      </div>
      <div className="hairline bg-panel p-4">
        <div className="h-2.5 w-48 animate-pulse rounded bg-border" />
        <div className="mt-3 flex flex-col gap-2">
          {Array.from({ length: 7 }).map((_, i) => (
            <div key={i} className="h-5 animate-pulse rounded bg-border" />
          ))}
        </div>
      </div>
    </div>
  );
}

interface ClassRow {
  className: string;
  split: 'seen' | 'heldout';
  rawTop1: number | null;
  hybridTop1: number | null;
  loco: number | null;
}

type SortKey = 'className' | 'rawTop1' | 'hybridTop1' | 'loco';

function buildClassRows(data: EvaluationResponse): ClassRow[] {
  const { evaluation, metrics } = data;
  return Object.entries(CLASS_SPLIT).map(([className, split]) => ({
    className,
    split,
    rawTop1: metrics[`class_${className}_top1`] ?? null,
    hybridTop1: evaluation.hybrid_per_class?.[`${className}_${split}`] ?? null,
    loco: split === 'seen' ? (metrics[`loco_${className}`] ?? null) : null,
  }));
}

export function ModelBenchmarks() {
  const reduced = useReducedMotion();
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'className', dir: 1 });

  useEffect(() => {
    const controller = new AbortController();
    api
      .getEvaluation(controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setData(result);
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

  const rows = useMemo(() => {
    if (!data) return [];
    const built = buildClassRows(data);
    const { key, dir } = sort;
    return [...built].sort((a, b) => {
      const av = a[key];
      const bv = b[key];
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      if (typeof av === 'string' || typeof bv === 'string') {
        return dir * String(av).localeCompare(String(bv));
      }
      return dir * ((av as number) - (bv as number));
    });
  }, [data, sort]);

  const toggleSort = (key: SortKey) => {
    setSort((prev) => (prev.key === key ? { key, dir: prev.dir === 1 ? -1 : 1 } : { key, dir: 1 }));
  };

  if (loading) return <BenchmarksSkeleton />;
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
  if (!data) return null;

  const { evaluation } = data;
  const judgeBaseline = evaluation.baselines['LLM-as-judge (Gemini)'] ?? null;

  // Percentages rather than fractions, because Recharts axes and labels read
  // far better in whole percent than in 0.075.
  const toPct = (v: number | null | undefined) => Math.round((v ?? 0) * 1000) / 10;

  const baselineRows = [
    ...Object.entries(evaluation.baselines).map(([label, scores]) => ({
      label,
      seen: toPct(scores.seen),
      heldout: toPct(scores.heldout),
      isOurs: false,
    })),
    {
      label: 'Hybrid engine',
      seen: toPct(evaluation.hybrid_seen_top1),
      heldout: toPct(evaluation.hybrid_heldout_top1),
      isOurs: true,
    },
  ];

  const radarRows = rows.map((row) => ({
    // The radar's spokes get crowded fast, so the long class names are
    // shortened for the axis only; the table below keeps them in full.
    cls: row.className.replace(/_/g, ' ').replace('hallucinated', 'halluc.'),
    raw: toPct(row.rawTop1),
    hybrid: toPct(row.hybridTop1 ?? row.rawTop1),
  }));

  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.08 } },
  };
  const item: Variants = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
  };

  const columns: { key: SortKey; label: string }[] = [
    { key: 'className', label: 'Fault class' },
    { key: 'rawTop1', label: 'Classifier top-1' },
    { key: 'hybridTop1', label: 'Hybrid top-1' },
    { key: 'loco', label: 'LOCO' },
  ];

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-4">
      <motion.div variants={item} className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <BenchmarkCard
          label="Trained-Class Accuracy"
          value={evaluation.test_seen_top1}
          subtitle={`Top-1 on ${evaluation.test_seen_runs} seen-class runs`}
        />
        <BenchmarkCard
          label="Classifier Alone (Held-out)"
          value={evaluation.test_heldout_top1}
          subtitle="Raw model, no fallback, fails the gate"
        />
        <BenchmarkCard
          label="LLM-as-Judge (Held-out)"
          value={judgeBaseline?.heldout ?? null}
          subtitle="Gemini given the full taxonomy"
        />
        <BenchmarkCard
          label="Hybrid Engine (OOD Fallback)"
          value={evaluation.hybrid_heldout_top1}
          subtitle="Distribution-relative fallback"
        />
      </motion.div>

      <motion.div
        variants={item}
        className="hairline border-accent/30 bg-accent/5 px-4 py-3.5"
      >
        <p className="text-[14px] leading-relaxed text-text">
          A distribution-relative fallback localizes unseen failure classes when pure classifiers
          fail.
        </p>
        <p className="data mt-2 text-muted">
          Raw model gate: {evaluation.raw_gate_passed ? 'passed' : 'failed'} · Hybrid gate:{' '}
          {evaluation.hybrid_gate_passed ? 'passed' : 'failed'} · leave-one-class-out mean:{' '}
          {pct(evaluation.loco_mean)} (flat across classes, a caveat we report rather than hide)
        </p>
      </motion.div>

      <motion.div variants={item} className="hairline bg-panel p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <p className="data text-muted">Baselines vs. the hybrid engine</p>
          <div className="flex items-center gap-4">
            <span className="data inline-flex items-center gap-1.5 text-muted">
              <span className="block size-2 rounded-sm" style={{ background: CHART.pass }} />
              trained classes
            </span>
            <span className="data inline-flex items-center gap-1.5 text-muted">
              <span className="block size-2 rounded-sm" style={{ background: CHART.warn }} />
              held out
            </span>
          </div>
        </div>

        <div className="mt-4" style={{ height: baselineRows.length * 54 + 36 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={baselineRows}
              layout="vertical"
              barGap={3}
              margin={{ top: 4, right: 52, bottom: 4, left: 8 }}
            >
              <CartesianGrid stroke={CHART.border} horizontal={false} />
              <XAxis type="number" domain={[0, 100]} unit="%" tick={AXIS} tickLine={false} axisLine={false} />
              <YAxis
                type="category"
                dataKey="label"
                tick={AXIS}
                tickLine={false}
                axisLine={false}
                width={132}
              />
              {/* The gate: held-out accuracy has to clear the last-step
                  baseline to mean anything. Drawing it makes the one number
                  the whole claim rests on visible rather than asserted. */}
              <ReferenceLine
                x={(evaluation.baselines['last step']?.heldout ?? 0) * 100}
                stroke={CHART.critical}
                strokeDasharray="4 4"
                label={{
                  value: 'held-out gate',
                  position: 'top',
                  fill: CHART.critical,
                  fontSize: 10,
                  fontFamily: "'Geist Mono', monospace",
                }}
              />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(125,249,196,0.06)' }} />
              <Bar dataKey="seen" name="trained" radius={[0, 2, 2, 0]} isAnimationActive={!reduced}>
                {baselineRows.map((row) => (
                  <Cell
                    key={`${row.label}-seen`}
                    fill={row.isOurs ? CHART.accent : CHART.pass}
                    fillOpacity={row.isOurs ? 0.95 : 0.45}
                  />
                ))}
                <LabelList
                  dataKey="seen"
                  position="right"
                  formatter={pctLabel}
                  style={LABEL_STYLE}
                />
              </Bar>
              <Bar dataKey="heldout" name="held out" radius={[0, 2, 2, 0]} isAnimationActive={!reduced}>
                {baselineRows.map((row) => (
                  <Cell
                    key={`${row.label}-heldout`}
                    fill={row.isOurs ? CHART.warn : CHART.muted}
                    fillOpacity={row.isOurs ? 0.95 : 0.4}
                  />
                ))}
                <LabelList
                  dataKey="heldout"
                  position="right"
                  formatter={pctLabel}
                  style={LABEL_STYLE}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="data mt-2 text-muted">
          Solid bars are this system. The dashed line is the last-step baseline on held-out
          classes, the bar every held-out number has to clear.
        </p>
      </motion.div>

      <motion.div variants={item} className="hairline bg-panel p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <p className="data text-muted">Per-class top-1: classifier alone vs. hybrid engine</p>
          <div className="flex items-center gap-4">
            <span className="data inline-flex items-center gap-1.5 text-muted">
              <span className="block size-2 rounded-sm" style={{ background: CHART.critical }} />
              classifier alone
            </span>
            <span className="data inline-flex items-center gap-1.5 text-muted">
              <span className="block size-2 rounded-sm" style={{ background: CHART.accent }} />
              hybrid
            </span>
          </div>
        </div>
        <div className="mt-2 h-80">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarRows} outerRadius="72%">
              <PolarGrid stroke={CHART.border} />
              <PolarAngleAxis dataKey="cls" tick={{ ...AXIS, fontSize: 10 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ ...AXIS, fontSize: 9 }} stroke={CHART.border} />
              <Radar
                name="classifier"
                dataKey="raw"
                stroke={CHART.critical}
                fill={CHART.critical}
                fillOpacity={0.18}
                isAnimationActive={!reduced}
              />
              <Radar
                name="hybrid"
                dataKey="hybrid"
                stroke={CHART.accent}
                fill={CHART.accent}
                fillOpacity={0.22}
                isAnimationActive={!reduced}
              />
              <Tooltip content={<ChartTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <p className="data text-muted">
          The two held-out classes are where the shapes separate: the classifier collapses on
          them and the distribution-relative tier is what fills the gap.
        </p>
      </motion.div>

      <motion.div variants={item} className="hairline overflow-x-auto bg-panel">
        <div className="flex items-center justify-between px-4 pt-4">
          <p className="data text-muted">Per-fault-class detection breakdown</p>
          <p className="data text-muted">Click a column to sort</p>
        </div>
        <table className="mt-3 w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-border">
              {columns.map((col) => (
                <th key={col.key} className="px-4 py-2.5">
                  <button
                    type="button"
                    onClick={() => toggleSort(col.key)}
                    className="data inline-flex items-center gap-1 font-medium uppercase tracking-[0.08em] text-muted transition-colors hover:text-accent"
                  >
                    {col.label}
                    {sort.key === col.key && <span>{sort.dir === 1 ? '↑' : '↓'}</span>}
                  </button>
                </th>
              ))}
              <th className="data px-4 py-2.5 font-medium uppercase tracking-[0.08em] text-muted">
                Split
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.className} className="border-b border-border last:border-0">
                <td className="data px-4 py-2.5 text-text">{row.className}</td>
                <td className="data px-4 py-2.5 text-text">{pct(row.rawTop1)}</td>
                <td className="data px-4 py-2.5 text-text">{pct(row.hybridTop1)}</td>
                <td className="data px-4 py-2.5 text-muted">{pct(row.loco)}</td>
                <td className="px-4 py-2.5">
                  <span
                    className={`data rounded border px-1.5 py-0.5 ${
                      row.split === 'heldout'
                        ? 'border-warn/40 bg-warn/10 text-warn'
                        : 'border-border text-muted'
                    }`}
                  >
                    {row.split === 'heldout' ? 'held out' : 'seen'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>
    </motion.div>
  );
}

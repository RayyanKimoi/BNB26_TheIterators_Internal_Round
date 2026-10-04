/**
 * Insights tab: aggregate reliability, read live from GET /dashboard/reliability.
 *
 * Recharts is used here because PRD.md's tech stack names it for exactly this
 * ("Charts | Recharts | Eval dashboard and reliability trends"). Every series
 * is computed server side from real `agent_runs` rows; nothing on this page is
 * sampled, smoothed or estimated.
 *
 * Cost deliberately shows "not configured" rather than a number unless
 * TOKEN_COST_PER_1K_USD is set: there is no per-token price anywhere in this
 * project, and a plausible-looking dollar figure on a reliability dashboard is
 * exactly the kind of invented metric CLAUDE.md forbids.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { api } from '../api/client';
import { AnimatedNumber } from '../components/AnimatedNumber';
import { ErrorState } from '../components/AppShell';
import type { ReliabilityResponse, SimilarRunsResponse } from '../types/api';

/* Palette drawn from the design tokens, never ad-hoc hexes. */
const COLORS = {
  accent: '#7df9c4',
  warn: '#e8b14c',
  critical: '#e0574c',
  pass: '#5bc98c',
  muted: '#8a8a92',
  border: '#1f1f23',
  panel: '#111113',
  text: '#e8e8e8',
};

const CLASS_COLORS = [
  COLORS.critical,
  COLORS.warn,
  COLORS.accent,
  COLORS.pass,
  '#b48ead',
  '#7fa7d9',
  '#c98b5b',
];

const AXIS = { stroke: COLORS.muted, fontSize: 11, fontFamily: "'Geist Mono', monospace" };

function ChartTooltip({ active, payload, label }: {
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
          <span style={{ color: entry.color }}>{entry.name}</span>: {entry.value}
        </p>
      ))}
    </div>
  );
}

function StatCard({
  label,
  value,
  decimals = 0,
  suffix = '',
  subtitle,
  muted = false,
}: {
  label: string;
  value: number | null;
  decimals?: number;
  suffix?: string;
  subtitle: string;
  muted?: boolean;
}) {
  return (
    <div className="hairline bg-panel px-4 py-3.5 transition-[border-color,box-shadow] duration-200 hover:border-accent/60 hover:shadow-[0_0_28px_-10px_var(--color-accent)]">
      <p className="data uppercase tracking-[0.08em] text-muted">{label}</p>
      <p className={`mt-1.5 font-display text-[26px] leading-none ${muted ? 'text-muted' : 'text-text'}`}>
        {value === null ? (
          <span className="text-muted">&ndash;</span>
        ) : (
          <AnimatedNumber value={value} decimals={decimals} suffix={suffix} />
        )}
      </p>
      <p className="data mt-2 text-muted">{subtitle}</p>
    </div>
  );
}

function InsightsSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="hairline bg-panel px-4 py-3.5">
            <div className="h-2.5 w-24 animate-pulse rounded bg-border" />
            <div className="mt-3 h-7 w-16 animate-pulse rounded bg-border" />
            <div className="mt-2 h-2.5 w-28 animate-pulse rounded bg-border" />
          </div>
        ))}
      </div>
      {Array.from({ length: 2 }).map((_, i) => (
        <div key={i} className="hairline bg-panel p-4">
          <div className="h-2.5 w-40 animate-pulse rounded bg-border" />
          <div className="mt-4 h-56 animate-pulse rounded bg-border" />
        </div>
      ))}
    </div>
  );
}

export interface InsightsViewProps {
  /** Optional run to show diagnosis-memory matches for. */
  memoryRunId?: string | null;
}

export function InsightsView({ memoryRunId }: InsightsViewProps) {
  const reduced = useReducedMotion();
  const [data, setData] = useState<ReliabilityResponse | null>(null);
  const [memory, setMemory] = useState<SimilarRunsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    api
      .getReliability(controller.signal)
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

  useEffect(() => {
    if (!memoryRunId) return;
    const controller = new AbortController();
    api
      .getSimilarRuns(memoryRunId, 3, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setMemory(result);
      })
      .catch(() => {
        // Non-fatal: the rest of the dashboard still renders without it.
      });
    return () => controller.abort();
  }, [memoryRunId]);

  const trend = useMemo(
    () =>
      (data?.trend ?? []).map((point) => ({
        ...point,
        pass_rate_pct: Math.round(point.pass_rate * 1000) / 10,
      })),
    [data],
  );

  if (loading) return <InsightsSkeleton />;
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

  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.08 } },
  };
  const item: Variants = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
  };

  const mix = data.failure_class_breakdown.map((entry) => ({
    name: entry.injected_class,
    value: entry.count,
  }));

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-4">
      <motion.div variants={item} className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label="Overall Pass Rate"
          value={data.overall_pass_rate * 100}
          decimals={1}
          suffix="%"
          subtitle={`${data.success_count} of ${data.total_runs} runs`}
        />
        <StatCard
          label="Failed Runs"
          value={data.failure_count}
          subtitle="Across every ingested source"
        />
        <StatCard
          label="Total Tokens"
          value={data.total_tokens}
          subtitle="Summed over all runs"
        />
        <StatCard
          label="Estimated Cost"
          value={data.cost_rate_configured ? (data.estimated_cost_usd ?? 0) : null}
          decimals={2}
          suffix=" USD"
          muted={!data.cost_rate_configured}
          subtitle={
            data.cost_rate_configured
              ? 'From TOKEN_COST_PER_1K_USD'
              : 'No token price configured, so none is assumed'
          }
        />
      </motion.div>

      <motion.div variants={item} className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <div className="hairline bg-panel p-4">
          <p className="data text-muted">Reliability over time</p>
          <div className="mt-4 h-60">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trend} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
                <defs>
                  <linearGradient id="passRateFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={COLORS.accent} stopOpacity={0.35} />
                    <stop offset="100%" stopColor={COLORS.accent} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={COLORS.border} vertical={false} />
                <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} />
                <YAxis
                  tick={AXIS}
                  tickLine={false}
                  axisLine={false}
                  domain={[0, 100]}
                  unit="%"
                  width={46}
                />
                <Tooltip content={<ChartTooltip />} cursor={{ stroke: COLORS.border }} />
                <Area
                  type="monotone"
                  dataKey="pass_rate_pct"
                  name="pass rate"
                  stroke={COLORS.accent}
                  strokeWidth={2}
                  fill="url(#passRateFill)"
                  isAnimationActive={!reduced}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="hairline bg-panel p-4">
          <p className="data text-muted">Failure class mix</p>
          <div className="mt-4 h-60">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={mix}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={48}
                  outerRadius={78}
                  paddingAngle={2}
                  stroke={COLORS.panel}
                  isAnimationActive={!reduced}
                >
                  {mix.map((entry, index) => (
                    <Cell key={entry.name} fill={CLASS_COLORS[index % CLASS_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip content={<ChartTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
            {mix.map((entry, index) => (
              <span key={entry.name} className="data inline-flex items-center gap-1.5 text-muted">
                <span
                  className="block size-2 rounded-full"
                  style={{ background: CLASS_COLORS[index % CLASS_COLORS.length] }}
                />
                {entry.name} {entry.value}
              </span>
            ))}
          </div>
        </div>
      </motion.div>

      <motion.div variants={item} className="grid gap-4 lg:grid-cols-2">
        <div className="hairline bg-panel p-4">
          <p className="data text-muted">Token usage by day</p>
          <div className="mt-4 h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={trend} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                <CartesianGrid stroke={COLORS.border} vertical={false} />
                <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} />
                <YAxis tick={AXIS} tickLine={false} axisLine={false} width={56} />
                <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(125,249,196,0.06)' }} />
                <Bar
                  dataKey="total_tokens"
                  name="tokens"
                  fill={COLORS.accent}
                  fillOpacity={0.65}
                  isAnimationActive={!reduced}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="hairline bg-panel p-4">
          <p className="data text-muted">Latency z-score by day</p>
          <div className="mt-4 h-52">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trend} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                <CartesianGrid stroke={COLORS.border} vertical={false} />
                <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} />
                <YAxis tick={AXIS} tickLine={false} axisLine={false} width={46} />
                <Tooltip content={<ChartTooltip />} cursor={{ stroke: COLORS.border }} />
                <Line
                  type="monotone"
                  dataKey="duration_zscore"
                  name="duration z"
                  stroke={COLORS.warn}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={!reduced}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="data mt-2 text-muted">
            Each day&apos;s average run duration against the whole period&apos;s mean and standard
            deviation.
          </p>
        </div>
      </motion.div>

      <motion.div variants={item} className="hairline bg-panel p-4">
        <p className="data text-muted">Diagnosis memory</p>
        {!memoryRunId ? (
          <p className="data mt-3 text-muted">
            Open a run in the Trace tab, then return here to see the most similar historical
            failures by feature vector.
          </p>
        ) : !memory ? (
          <p className="data mt-3 text-muted">
            No diagnosis stored for the selected run yet, so there is nothing to match against.
          </p>
        ) : (
          <>
            <p className="data mt-1 text-muted">
              Closest matches to {memoryRunId.slice(0, 8)} across {memory.compared_against}{' '}
              diagnosed runs
            </p>
            <ul className="mt-3 flex flex-col gap-2">
              {memory.matches.map((match) => (
                <li key={match.run_id} className="flex flex-wrap items-center gap-3">
                  <span className="data text-text">{match.run_id.slice(0, 8)}</span>
                  <span className="hairline h-2 w-32 overflow-hidden rounded bg-bg">
                    <span
                      className="block h-full rounded bg-accent"
                      style={{ width: `${Math.max(2, Math.round(match.similarity * 100))}%` }}
                    />
                  </span>
                  <span className="data text-accent">{(match.similarity * 100).toFixed(1)}%</span>
                  <span className="data text-muted">predicted {match.predicted_class}</span>
                  {match.injected_class && (
                    <span className="data text-muted">actual {match.injected_class}</span>
                  )}
                </li>
              ))}
            </ul>
          </>
        )}
      </motion.div>
    </motion.div>
  );
}

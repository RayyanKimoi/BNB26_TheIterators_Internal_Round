/**
 * The four headline metrics above the runs table.
 *
 * Two of the numbers (trained-class and hybrid accuracy) are measured model
 * results passed in as props; the other two are derived from the loaded runs.
 * Nothing here invents a figure. When a value is still loading the card shows a
 * skeleton bar rather than a zero that would read as a real reading.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';

import { AnimatedNumber } from './AnimatedNumber';

export interface Metric {
  label: string;
  /** Numeric value; null renders as an em-free dash when not computable. */
  value: number | null;
  decimals?: number;
  suffix?: string;
  subtitle?: string;
}

export interface MetricsHeaderProps {
  totalRuns: number;
  /** Trained-class top-1, measured. Passed as a percentage, e.g. 95.0. */
  trainedAccuracy: number;
  /** Hybrid engine held-out top-1, measured. Passed as a percentage. */
  hybridAccuracy: number;
  /** Share of failed runs carrying a saved diagnosis, 0 to 100, or null. */
  detectionRate: number | null;
  loading?: boolean;
}

function MetricCard({ metric, loading }: { metric: Metric; loading?: boolean }) {
  const reduced = useReducedMotion();

  const card: Variants = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 8 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.32, ease: 'easeOut' },
    },
  };

  return (
    <motion.div
      variants={card}
      className="hairline bg-panel px-4 py-3.5 transition-[border-color,box-shadow] duration-200 hover:border-accent/60 hover:shadow-[0_0_28px_-10px_var(--color-accent)]"
    >
      <p className="data uppercase tracking-[0.08em] text-muted">{metric.label}</p>

      {loading ? (
        <div className="mt-2.5 h-7 w-20 animate-pulse rounded bg-border" aria-hidden="true" />
      ) : (
        <p className="mt-1.5 font-display text-[28px] leading-none text-text">
          {metric.value === null ? (
            <span className="text-muted">&ndash;</span>
          ) : (
            <AnimatedNumber
              value={metric.value}
              decimals={metric.decimals ?? 0}
              suffix={metric.suffix ?? ''}
            />
          )}
        </p>
      )}

      {metric.subtitle && <p className="data mt-2 text-muted">{metric.subtitle}</p>}
    </motion.div>
  );
}

export function MetricsHeader({
  totalRuns,
  trainedAccuracy,
  hybridAccuracy,
  detectionRate,
  loading = false,
}: MetricsHeaderProps) {
  const reduced = useReducedMotion();

  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.06 } },
  };

  const metrics: Metric[] = [
    { label: 'Total Runs', value: totalRuns },
    {
      label: 'Trained-Class Accuracy',
      value: trainedAccuracy,
      decimals: 1,
      suffix: '%',
      subtitle: 'Top-1 on seen classes',
    },
    {
      label: 'Hybrid Engine Accuracy',
      value: hybridAccuracy,
      decimals: 1,
      suffix: '%',
      subtitle: 'Distribution-relative fallback',
    },
    {
      label: 'Fault Detection Rate',
      value: detectionRate,
      decimals: detectionRate !== null && Number.isInteger(detectionRate) ? 0 : 1,
      suffix: '%',
      subtitle: 'Failed runs carrying a diagnosis',
    },
  ];

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="grid grid-cols-2 gap-3 lg:grid-cols-4"
    >
      {metrics.map((metric) => (
        <MetricCard key={metric.label} metric={metric} loading={loading} />
      ))}
    </motion.div>
  );
}

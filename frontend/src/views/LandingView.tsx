/**
 * Developer landing page.
 *
 * Dark terminal aesthetic on the project palette. The animated background,
 * particle wordmark and ASCII mark are self-contained effects (see
 * components/effects); nothing here pulls a third-party registry component or a
 * purple gradient. The two CTAs open the auth view or start a one-click demo.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import type { ReactNode } from 'react';

import { AsciiLogo } from '../components/effects/AsciiLogo';
import { ParticleText } from '../components/effects/ParticleText';
import { PixelField } from '../components/effects/PixelField';
import { LivePreviewCard } from '../components/LivePreviewCard';
import { useAuth } from '../context/authCore';

interface Feature {
  title: string;
  body: string;
  icon: ReactNode;
}

const FEATURES: Feature[] = [
  {
    title: '10-Feature Invariant Scanning',
    body: 'Every step scored on ten behavioural features, with a distribution-relative tier that flags anomalies no class covers.',
    icon: (
      <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M3 17l5-6 4 4 5-8 4 5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: 'Gemini Root Cause Explanations',
    body: 'A JSON-constrained Gemini pass turns the flagged step into a root cause, evidence summary and a proposed fix.',
    icon: (
      <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M12 3v3M12 18v3M4.2 7l2.1 2.1M17.7 14.9l2.1 2.1M3 12h3M18 12h3" strokeLinecap="round" />
        <circle cx="12" cy="12" r="3.2" />
      </svg>
    ),
  },
  {
    title: 'Deterministic Suffix Replay',
    body: 'Fork from any step with a fix, then replay the suffix from a checkpoint to see the run flip from fail to success.',
    icon: (
      <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M4 7h9a4 4 0 0 1 4 4v1" strokeLinecap="round" />
        <path d="M14 4l3 3-3 3" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M20 17h-9a4 4 0 0 1-4-4v-1" strokeLinecap="round" />
        <path d="M10 20l-3-3 3-3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: 'SHAP Feature Attribution',
    body: 'Model-level SHAP values show which features pushed a step toward root cause, so the call is never a black box.',
    icon: (
      <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="1.4">
        <rect x="4" y="12" width="3.4" height="7" rx="0.6" />
        <rect x="10.3" y="7" width="3.4" height="12" rx="0.6" />
        <rect x="16.6" y="9.5" width="3.4" height="9.5" rx="0.6" />
      </svg>
    ),
  },
];

export function LandingView({ onSignIn }: { onSignIn: () => void }) {
  const { loginAsGuest } = useAuth();
  const reduced = useReducedMotion();

  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.08, delayChildren: 0.05 } },
  };
  const item: Variants = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 14 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-bg text-text">
      <PixelField className="pointer-events-none absolute inset-0 h-full w-full opacity-70" />
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-bg/10 via-bg/40 to-bg" />

      <div className="relative mx-auto flex max-w-[1200px] flex-col px-4 sm:px-6">
        {/* top bar */}
        <header className="flex items-center justify-between py-5">
          <div className="flex items-center gap-2">
            <span className="dot-matrix hairline block size-5" aria-hidden="true" />
            <span className="font-display text-sm text-text">Black Box</span>
          </div>
          <button
            type="button"
            onClick={onSignIn}
            className="data hairline rounded px-3 py-1.5 text-text transition-colors hover:border-accent/60 hover:text-accent"
          >
            Sign In
          </button>
        </header>

        {/* hero */}
        <section className="grid items-center gap-10 py-10 lg:grid-cols-[1.1fr_0.9fr] lg:py-16">
          <motion.div variants={container} initial="hidden" animate="show">
            <motion.div variants={item} className="mb-5 flex items-center gap-4">
              <AsciiLogo />
              <span className="data rounded-full border border-accent/30 bg-accent/5 px-2.5 py-1 text-accent">
                Sentry for AI Agents
              </span>
            </motion.div>

            <motion.div variants={item} className="min-h-[72px]">
              <ParticleText text="BLACK BOX" fontSize={84} color="#e8e8e8" className="block" />
            </motion.div>

            <motion.h1
              variants={item}
              className="mt-2 font-display text-2xl leading-tight text-text sm:text-3xl"
            >
              Autonomous Agent Trace Anomaly Detection and Replay
            </motion.h1>

            <motion.p variants={item} className="mt-4 max-w-xl text-[15px] leading-relaxed text-muted">
              Distribution-relative fault localization, root cause analysis with Gemini, and suffix
              step replay.
            </motion.p>

            <motion.div variants={item} className="mt-7 flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={onSignIn}
                className="rounded border border-accent bg-accent/10 px-4 py-2.5 text-[14px] font-medium text-accent transition-colors hover:bg-accent/20"
              >
                Launch Dashboard / Sign In
              </button>
              <button
                type="button"
                onClick={loginAsGuest}
                className="hairline rounded px-4 py-2.5 text-[14px] text-text transition-colors hover:border-accent/60 hover:text-accent"
              >
                Instant Demo Access (1-Click)
              </button>
            </motion.div>
          </motion.div>

          <div className="flex justify-center lg:justify-end">
            <LivePreviewCard />
          </div>
        </section>

        {/* feature grid */}
        <motion.section
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
          className="grid gap-3 pb-16 sm:grid-cols-2 lg:grid-cols-4"
        >
          {FEATURES.map((feature) => (
            <motion.div
              key={feature.title}
              variants={item}
              className="hairline bg-panel/80 p-4 backdrop-blur transition-[border-color,box-shadow] duration-200 hover:border-accent/50 hover:shadow-[0_0_30px_-12px_var(--color-accent)]"
            >
              <span className="text-accent">{feature.icon}</span>
              <h3 className="mt-3 font-display text-[14px] text-text">{feature.title}</h3>
              <p className="mt-1.5 text-[13px] leading-relaxed text-muted">{feature.body}</p>
            </motion.div>
          ))}
        </motion.section>

        <footer className="border-t border-border py-5">
          <p className="data text-muted">
            A trained classifier that generalizes to held-out failure classes, not an LLM-as-judge.
          </p>
        </footer>
      </div>
    </div>
  );
}

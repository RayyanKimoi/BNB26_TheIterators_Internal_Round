/**
 * Deep-dive feature section: four cards with scroll-reveal, each with a small
 * live visual. Reveal uses opacity + translate (transform), bars grow via scaleX,
 * and the fork comparison is a user-driven before/after wipe (clip-path), so
 * nothing animates a layout property. Specific numbers shown in the JSON and the
 * SHAP chart are labelled as examples; the feature names are the real ten.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { useState } from 'react';

const FEATURES = [
  'duration_z',
  'token_z',
  'retry_count',
  'parse_failure',
  'tool_choice_entropy',
  'semantic_deviation',
  'arg_novelty',
  'state_hash_repeat',
  'downstream_error_count',
  'position_ratio',
];

const SHAP = [
  { name: 'semantic_deviation', value: 0.31 },
  { name: 'arg_novelty', value: 0.24 },
  { name: 'duration_z', value: 0.12 },
  { name: 'state_hash_repeat', value: 0.08 },
  { name: 'token_z', value: 0.05 },
];

const card: Variants = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
};

function InvariantScanner() {
  const reduced = useReducedMotion();
  return (
    <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2">
      {FEATURES.map((name, i) => {
        const fill = 0.35 + ((i * 7) % 5) * 0.12;
        return (
          <div key={name} className="flex flex-col gap-1">
            <span className="data truncate text-muted">{name}</span>
            <div className="hairline h-1.5 w-full overflow-hidden rounded-full bg-bg">
              <motion.div
                className="h-full origin-left rounded-full bg-accent/70"
                initial={{ scaleX: reduced ? fill : 0 }}
                whileInView={{ scaleX: fill }}
                viewport={{ once: true, amount: 0.6 }}
                transition={{ duration: 0.5, ease: 'easeOut', delay: i * 0.03 }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function GeminiRootCause() {
  return (
    <div className="hairline mt-4 bg-bg p-3">
      <div className="data flex items-center justify-between text-muted">
        <span>explanation.json</span>
        <span className="rounded border border-border px-1.5 py-0.5">example</span>
      </div>
      <pre className="data mt-2 overflow-x-auto leading-relaxed text-text">
        {'{\n'}
        {'  '}
        <span className="text-accent">&quot;root_cause&quot;</span>: &quot;stale currency field&quot;,{'\n'}
        {'  '}
        <span className="text-accent">&quot;evidence_summary&quot;</span>: [&quot;step 7 reused USD&quot;],{'\n'}
        {'  '}
        <span className="text-accent">&quot;proposed_fix&quot;</span>: &quot;set currency to EUR&quot;{'\n'}
        {'}'}
      </pre>
    </div>
  );
}

function ForkReplaySlider() {
  const [value, setValue] = useState(45);
  const steps = Array.from({ length: 12 });
  const reveal = `inset(0 ${100 - value}% 0 0)`;

  return (
    <div className="mt-4">
      <div className="relative">
        {/* parent: fail */}
        <div className="flex items-center gap-1">
          {steps.map((_, i) => (
            <span
              key={i}
              className={`h-6 flex-1 rounded-sm ${i === 7 ? 'bg-critical' : 'bg-pass/40'}`}
            />
          ))}
        </div>
        {/* forked: success, revealed by the slider */}
        <div className="absolute inset-0 flex items-center gap-1" style={{ clipPath: reveal }}>
          {steps.map((_, i) => (
            <span key={i} className="h-6 flex-1 rounded-sm bg-pass" />
          ))}
        </div>
      </div>

      <div className="mt-2 flex items-center justify-between">
        <span className="data text-critical">parent: FAIL</span>
        <span className="data text-pass">forked: SUCCESS</span>
      </div>

      <label htmlFor="fork-slider" className="sr-only">
        Compare parent and forked trace
      </label>
      <input
        id="fork-slider"
        type="range"
        min={0}
        max={100}
        value={value}
        onChange={(e) => setValue(Number(e.target.value))}
        className="mt-3 w-full accent-accent"
      />
    </div>
  );
}

function ShapBars() {
  const reduced = useReducedMotion();
  const max = Math.max(...SHAP.map((s) => s.value));
  return (
    <div className="mt-4 flex flex-col gap-2.5">
      {SHAP.map((item, i) => (
        <div key={item.name} className="flex items-center gap-3">
          <span className="data w-40 shrink-0 truncate text-muted">{item.name}</span>
          <div className="hairline h-3 flex-1 overflow-hidden rounded bg-bg">
            <motion.div
              className="h-full origin-left rounded bg-accent"
              initial={{ scaleX: reduced ? item.value / max : 0 }}
              whileInView={{ scaleX: item.value / max }}
              viewport={{ once: true, amount: 0.6 }}
              transition={{ duration: 0.6, ease: 'easeOut', delay: i * 0.05 }}
            />
          </div>
          <span className="data w-10 shrink-0 text-right text-text">{item.value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

interface DeepDiveCard {
  index: string;
  title: string;
  body: string;
  visual: React.ReactNode;
}

const CARDS: DeepDiveCard[] = [
  {
    index: '01',
    title: '10-Feature Invariant Scanner',
    body: 'Every step is scored on ten behavioural features. A distribution-relative tier flags steps that breach learned bounds even when no trained class fits.',
    visual: <InvariantScanner />,
  },
  {
    index: '02',
    title: 'Gemini Root Cause Explanations',
    body: 'The flagged step goes to Gemini under a strict JSON schema and comes back as a root cause, an evidence summary and a proposed fix.',
    visual: <GeminiRootCause />,
  },
  {
    index: '03',
    title: 'Deterministic Suffix Replay',
    body: 'Drag to compare the parent run against the fork. Patch the flagged step, replay the suffix from a checkpoint, watch it flip to success.',
    visual: <ForkReplaySlider />,
  },
  {
    index: '04',
    title: 'SHAP Feature Attribution',
    body: 'SHAP values rank which features pushed a step toward root cause, so the verdict is auditable rather than opaque.',
    visual: <ShapBars />,
  },
];

export function FeatureDeepDive() {
  return (
    <section className="py-24">
      <h2 className="font-display text-3xl text-text">How it localizes a fault</h2>
      <p className="mt-3 max-w-xl text-[16px] text-muted">
        Four layers turn a failed trace into a located, explained, and fixable root cause.
      </p>

      <div className="mt-8 grid gap-3 lg:grid-cols-2">
        {CARDS.map((item) => (
          <motion.div
            key={item.index}
            variants={card}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, amount: 0.3 }}
            className="hairline glow-accent bg-panel/70 p-5 backdrop-blur transition-[border-color,box-shadow] duration-200 hover:border-accent/70 hover:shadow-[0_0_40px_-10px_var(--color-accent)]"
          >
            <div className="flex items-baseline gap-3">
              <span className="data text-accent">{item.index}</span>
              <h3 className="font-display text-lg text-text">{item.title}</h3>
            </div>
            <p className="mt-2 text-[14px] leading-relaxed text-muted">{item.body}</p>
            {item.visual}
          </motion.div>
        ))}
      </div>
    </section>
  );
}

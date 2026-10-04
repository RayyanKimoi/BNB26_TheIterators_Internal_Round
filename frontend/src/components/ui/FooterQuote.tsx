/**
 * Closing terminal quote plus the landing footer.
 *
 * The footer shows a real backend liveness probe (same endpoint the shell polls)
 * and a final CTA. No em dash in the byline, per the UI copy rule; the
 * attribution sits on its own mono line instead.
 */

import { motion, useReducedMotion } from 'framer-motion';
import { useEffect, useState } from 'react';

import { api } from '../../api/client';

type Live = 'checking' | 'live' | 'offline';

const LIVE_COPY: Record<Live, { label: string; dot: string; text: string }> = {
  checking: { label: 'CHECKING', dot: 'bg-muted', text: 'text-muted' },
  live: { label: 'LIVE', dot: 'bg-pass', text: 'text-pass' },
  offline: { label: 'OFFLINE', dot: 'bg-critical', text: 'text-critical' },
};

export function FooterQuote({ onGetStarted }: { onGetStarted: () => void }) {
  const reduced = useReducedMotion();
  const [live, setLive] = useState<Live>('checking');

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const check = async () => {
      const ok = await api.health(controller.signal);
      if (!cancelled) setLive(ok ? 'live' : 'offline');
    };
    void check();
    const timer = setInterval(() => void check(), 30_000);
    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(timer);
    };
  }, []);

  const status = LIVE_COPY[live];

  return (
    <section className="py-24">
      <motion.figure
        initial={reduced ? { opacity: 0 } : { opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.4 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="hairline glow-accent mx-auto max-w-2xl bg-panel/70 p-6 backdrop-blur"
      >
        <div className="data mb-3 text-muted">
          <span className="text-accent">$</span> cat principles.txt
        </div>
        <blockquote className="font-display text-2xl leading-snug text-text sm:text-3xl">
          &quot;Simplicity is prerequisite for reliability.&quot;
        </blockquote>
        <figcaption className="data mt-3 text-muted">Edsger W. Dijkstra</figcaption>
      </motion.figure>

      <div className="mx-auto mt-10 flex max-w-2xl flex-col items-center gap-5 text-center">
        <h2 className="font-display text-3xl text-text">Get Started in 60 Seconds</h2>
        <button
          type="button"
          onClick={onGetStarted}
          className="rounded border border-accent bg-accent/10 px-5 py-2.5 text-[14px] font-medium text-accent transition-colors hover:bg-accent/20"
        >
          Instant Demo Access (1-Click)
        </button>
      </div>

      <div className="mt-12 flex items-center justify-between border-t border-border pt-5">
        <span className="data text-muted">Black Box / Sentry for AI Agents</span>
        <span className="inline-flex items-center gap-2">
          <span className={`block size-1.5 rounded-full ${status.dot}`} aria-hidden="true" />
          <span className={`data ${status.text}`}>BACKEND {status.label}</span>
        </span>
      </div>
    </section>
  );
}

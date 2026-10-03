/**
 * Scroll-driven 3D card stack, paired with an ASCII render of a marble bust that
 * fills what used to be empty space beside the carousel.
 *
 * The section pins a stage; as it scrolls the cards rotate along a depth
 * trajectory (all transforms, so the scrub stays on the compositor). Under
 * reduced motion it degrades to a readable two-column grid with the same ASCII
 * panel. Source image: MET Open Access, CC0.
 */

import { motion, useReducedMotion, useScroll, useTransform } from 'framer-motion';
import type { MotionValue } from 'framer-motion';
import { useRef } from 'react';

import { AsciiImage } from './AsciiImage';

export interface SpiralItem {
  tag: string;
  title: string;
  body: string;
}

const ITEMS: SpiralItem[] = [
  {
    tag: '01',
    title: 'Telemetry Ingestion',
    body: 'LangGraph checkpoints and OpenTelemetry spans normalize into the same step schema the detector scores.',
  },
  {
    tag: '02',
    title: 'Real-Time Scoring',
    body: 'Ten features per step, a trained classifier and a distribution-relative invariant tier, 0 to 100 per step.',
  },
  {
    tag: '03',
    title: 'One-Click Forking',
    body: 'Patch the flagged step and replay the suffix deterministically from a checkpoint. No live agent re-run.',
  },
  {
    tag: '04',
    title: 'Zero-Latency Caching',
    body: 'Explanations persist on the run, so a second view of a diagnosis costs nothing and never re-hits the LLM.',
  },
];

function AsciiPanel({ className = '' }: { className?: string }) {
  return (
    <div className={`hairline relative overflow-hidden bg-bg ${className}`}>
      <AsciiImage
        src="/statue.jpg"
        fontSize={8}
        cursorRadius={170}
        flowStrength={16}
        className="absolute inset-0 h-full w-full"
      />
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between p-3">
        <span className="data text-muted">herodotos.jpg</span>
        <span className="data text-accent">RENDER / LIVE</span>
      </div>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between p-3">
        <span className="data text-muted">image to ascii</span>
        <span className="data text-muted">MET / CC0</span>
      </div>
    </div>
  );
}

function SpiralCard({
  progress,
  index,
  total,
  item,
}: {
  progress: MotionValue<number>;
  index: number;
  total: number;
  item: SpiralItem;
}) {
  const active = useTransform(progress, (p) => p * (total - 1));
  const rotateY = useTransform(active, (a) => (index - a) * -32);
  const z = useTransform(active, (a) => -Math.abs(index - a) * 260);
  const x = useTransform(active, (a) => `${(index - a) * 46}%`);
  const opacity = useTransform(active, (a) => Math.max(0, 1 - Math.abs(index - a) * 0.6));
  const zIndex = useTransform(active, (a) => Math.round(100 - Math.abs(index - a) * 10));

  return (
    <motion.div style={{ rotateY, z, x, opacity, zIndex }} className="absolute w-[300px] sm:w-[400px]">
      <div className="hairline bg-panel/90 p-6 backdrop-blur">
        <span className="data text-[13px] text-accent">{item.tag}</span>
        <h3 className="mt-3 font-display text-2xl text-text">{item.title}</h3>
        <p className="mt-3 text-[15px] leading-relaxed text-muted">{item.body}</p>
      </div>
    </motion.div>
  );
}

export function SpiralGallery() {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start start', 'end end'],
  });

  if (reduced) {
    return (
      <section className="py-24">
        <h2 className="font-display text-3xl text-text">Platform</h2>
        <div className="mt-8 grid items-start gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <AsciiPanel className="h-[460px]" />
          <div className="grid gap-4 sm:grid-cols-2">
            {ITEMS.map((item) => (
              <div key={item.tag} className="hairline bg-panel p-6">
                <span className="data text-accent">{item.tag}</span>
                <h3 className="mt-3 font-display text-xl text-text">{item.title}</h3>
                <p className="mt-2 text-[14px] leading-relaxed text-muted">{item.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  }

  return (
    <section ref={ref} style={{ height: `${ITEMS.length * 95 + 70}vh` }} className="relative">
      <div className="sticky top-0 flex h-screen items-center overflow-hidden">
        <div className="mx-auto grid w-full max-w-[1280px] items-center gap-8 px-4 sm:px-6 lg:grid-cols-[0.9fr_1.1fr]">
          <AsciiPanel className="h-[42vh] max-h-[360px] lg:h-[68vh] lg:max-h-[600px]" />

          <div className="relative flex h-[440px] items-center justify-center" style={{ perspective: 1300 }}>
            <span className="data absolute -top-2 left-0 text-muted">
              Platform / scroll to explore
            </span>
            {ITEMS.map((item, i) => (
              <SpiralCard
                key={item.tag}
                progress={scrollYProgress}
                index={i}
                total={ITEMS.length}
                item={item}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/**
 * Horizontal ticker whose speed and direction bend to scroll velocity.
 *
 * Two rows drift in opposite directions; scrolling accelerates them and flips
 * direction with scroll direction. Translation only (the `x` transform), so it
 * stays off the layout path. Under reduced motion both rows hold still. The
 * accuracy figures shown are the measured model numbers, not decoration.
 */

import {
  motion,
  useAnimationFrame,
  useMotionValue,
  useReducedMotion,
  useScroll,
  useSpring,
  useTransform,
  useVelocity,
} from 'framer-motion';
import { useRef } from 'react';

const FAULTS = [
  'STALE_RETRIEVAL',
  'WRONG_TOOL_CHOSEN',
  'INFINITE_LOOP',
  'HALLUCINATED_ARGUMENT',
  'PREMATURE_TERMINATION',
  'SCHEMA_VIOLATION',
  'CONTEXT_TRUNCATION',
];

const STATS = [
  '95.0% TRAINED-CLASS ACCURACY',
  '52.5% OOD HYBRID ENGINE',
  '10-FEATURE INVARIANT SCAN',
  'DETERMINISTIC SUFFIX REPLAY',
];

function wrap(min: number, max: number, value: number): number {
  const range = max - min;
  return ((((value - min) % range) + range) % range) + min;
}

function Row({
  items,
  baseVelocity,
  accent,
}: {
  items: string[];
  baseVelocity: number;
  accent?: boolean;
}) {
  const reduced = useReducedMotion();
  const baseX = useMotionValue(0);
  const { scrollY } = useScroll();
  const scrollVelocity = useVelocity(scrollY);
  const smoothVelocity = useSpring(scrollVelocity, { damping: 50, stiffness: 400 });
  const velocityFactor = useTransform(smoothVelocity, [0, 1000], [0, 4], { clamp: false });
  const x = useTransform(baseX, (v) => `${wrap(-25, -50, v)}%`);
  const direction = useRef(1);

  useAnimationFrame((_t, delta) => {
    if (reduced) return;
    let moveBy = direction.current * baseVelocity * (delta / 1000);
    const factor = velocityFactor.get();
    if (factor < 0) direction.current = -1;
    else if (factor > 0) direction.current = 1;
    moveBy += direction.current * moveBy * Math.abs(factor);
    baseX.set(baseX.get() + moveBy);
  });

  const copies = [0, 1, 2, 3];
  return (
    <div className="overflow-hidden py-3">
      <motion.div
        style={reduced ? undefined : { x }}
        className="flex whitespace-nowrap will-change-transform"
      >
        {copies.map((copy) => (
          <div key={copy} className="flex shrink-0">
            {items.map((item) => (
              <span key={`${copy}-${item}`} className="flex items-center">
                <span className={`data text-[15px] ${accent ? 'text-accent' : 'text-muted'}`}>
                  {item}
                </span>
                <span className="px-6 text-border" aria-hidden="true">
                  /
                </span>
              </span>
            ))}
          </div>
        ))}
      </motion.div>
    </div>
  );
}

export function VelocityScroll() {
  return (
    <div className="border-y border-border bg-panel/40">
      <Row items={FAULTS} baseVelocity={-3} />
      <div className="h-px bg-border" />
      <Row items={STATS} baseVelocity={3} accent />
    </div>
  );
}

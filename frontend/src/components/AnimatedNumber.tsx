/**
 * A number that tweens to its target instead of snapping.
 *
 * Used for the metric cards and the "showing X of Y" counter. On first mount it
 * counts up from zero; on later changes it eases from the previous value. When
 * the viewer prefers reduced motion it renders the final value with no tween,
 * which is why the media query in index.css is not enough on its own: Framer's
 * imperative `animate` does not read that query for us.
 */

import { animate, useReducedMotion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

export interface AnimatedNumberProps {
  value: number;
  decimals?: number;
  suffix?: string;
  durationMs?: number;
}

export function AnimatedNumber({
  value,
  decimals = 0,
  suffix = '',
  durationMs = 600,
}: AnimatedNumberProps) {
  const reduced = useReducedMotion();
  const [display, setDisplay] = useState(reduced ? value : 0);
  const prev = useRef(reduced ? value : 0);

  useEffect(() => {
    if (reduced || prev.current === value) {
      setDisplay(value);
      prev.current = value;
      return;
    }
    const controls = animate(prev.current, value, {
      duration: durationMs / 1000,
      ease: 'easeOut',
      onUpdate: (v) => setDisplay(v),
    });
    prev.current = value;
    return () => controls.stop();
  }, [value, reduced, durationMs]);

  return (
    <>
      {display.toFixed(decimals)}
      {suffix}
    </>
  );
}

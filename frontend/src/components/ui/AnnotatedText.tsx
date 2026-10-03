/**
 * A word with a hand-drawn SVG accent that draws itself when scrolled into view.
 *
 * Two variants: an underline and a loose circle. The path is animated with
 * Framer's pathLength (a GPU-friendly stroke animation, not a layout change) and
 * stretches to the word via a non-uniform viewBox. Under reduced motion the
 * accent is simply shown fully drawn.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { ReactNode } from 'react';

export interface AnnotatedTextProps {
  children: ReactNode;
  variant?: 'underline' | 'circle';
  className?: string;
  /** Stroke colour; defaults to the mint accent token. */
  color?: string;
}

const UNDERLINE = 'M2,15 C 22,10 44,19 66,13 C 80,9 92,16 98,12';
const CIRCLE =
  'M50,4 C 80,4 97,12 97,20 C 97,30 74,36 50,36 C 26,36 3,30 3,20 C 3,12 22,4 50,5';

export function AnnotatedText({
  children,
  variant = 'underline',
  className = '',
  color = '#7df9c4',
}: AnnotatedTextProps) {
  const reduced = useReducedMotion();
  const circle = variant === 'circle';
  const path = circle ? CIRCLE : UNDERLINE;

  return (
    <span className={`relative inline-block ${className}`}>
      <span className="relative z-10">{children}</span>
      <svg
        viewBox="0 0 100 40"
        preserveAspectRatio="none"
        aria-hidden="true"
        className="pointer-events-none absolute left-0 w-full"
        style={
          circle
            ? { top: '-18%', height: '136%' }
            : { bottom: '-0.32em', height: '0.5em' }
        }
      >
        <motion.path
          d={path}
          fill="none"
          stroke={color}
          strokeWidth={circle ? 2 : 3}
          strokeLinecap="round"
          initial={{ pathLength: reduced ? 1 : 0, opacity: reduced ? 0.9 : 0 }}
          whileInView={{ pathLength: 1, opacity: 0.9 }}
          viewport={{ once: true, amount: 0.8 }}
          transition={{ duration: 0.7, ease: 'easeOut', delay: 0.15 }}
        />
      </svg>
    </span>
  );
}

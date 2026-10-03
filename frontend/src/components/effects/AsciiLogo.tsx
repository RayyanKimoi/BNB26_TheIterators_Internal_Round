/**
 * ASCII / dithered logo mark.
 *
 * A monospace cube drawn in characters. The base layer is muted; a mint copy
 * sits on top, revealed only within a soft circle that follows the pointer
 * (a CSS mask driven by custom properties), which gives the "dithered logo"
 * cursor-reveal feel with no canvas and no animation loop. Under reduced motion
 * the mint layer is dropped and only the static mark shows.
 */

import { useRef } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { useReducedMotion } from 'framer-motion';

const ART = [
  '    ______',
  '   /     /|',
  '  /_____/ |',
  '  |     | |',
  '  | ### | /',
  '  |_____|/',
].join('\n');

export interface AsciiLogoProps {
  className?: string;
}

export function AsciiLogo({ className = '' }: AsciiLogoProps) {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();

  const onMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (reduced) return;
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    el.style.setProperty('--mx', `${event.clientX - rect.left}px`);
    el.style.setProperty('--my', `${event.clientY - rect.top}px`);
  };

  const mask =
    'radial-gradient(70px at var(--mx, -999px) var(--my, -999px), #000 0%, #000 35%, transparent 72%)';

  return (
    <div
      ref={ref}
      onPointerMove={onMove}
      className={`relative select-none ${className}`}
      aria-hidden="true"
    >
      <pre className="font-mono text-[13px] leading-[1.15] text-border">{ART}</pre>
      {!reduced && (
        <pre
          className="pointer-events-none absolute inset-0 font-mono text-[13px] leading-[1.15] text-accent"
          style={{ WebkitMaskImage: mask, maskImage: mask }}
        >
          {ART}
        </pre>
      )}
    </div>
  );
}

/**
 * Interactive ASCII matrix background.
 *
 * A grid of monospace glyphs on near-black, visible on its own at a readable
 * baseline density and brightness — the matrix is the content, not a reward for
 * finding the cursor. The pointer adds a brighter mint highlight on top of that
 * baseline and leaves a short trail; it is additive, never the only light
 * source, so there is no dark halo or vignette where the cursor is not.
 * Self-contained canvas, project palette only, no purple. The loop pauses when
 * the canvas scrolls offscreen and never starts under reduced motion (a single
 * static frame is painted instead).
 */

import { useEffect, useRef } from 'react';
import { useReducedMotion } from 'framer-motion';

const RAMP = ' .:-=+*#%@';

export interface AsciiRevealProps {
  className?: string;
  /** Cell size in CSS pixels. */
  cell?: number;
  /** Pointer influence radius in CSS pixels. */
  radius?: number;
}

export function AsciiReveal({ className = '', cell = 17, radius = 300 }: AsciiRevealProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let width = 0;
    let height = 0;
    let raf = 0;
    let visible = true;
    const mouse = { x: -9999, y: -9999 };
    const heat = new Map<string, number>();

    const setFont = () => {
      ctx.font = `${cell - 5}px 'Geist Mono', ui-monospace, monospace`;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';
    };
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      setFont();
    };
    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);

    const frame = (time: number, animate: boolean) => {
      const t = time * 0.001;
      ctx.clearRect(0, 0, width, height);
      for (let y = cell; y < height; y += cell) {
        for (let x = cell; x < width; x += cell) {
          // Baseline is the readable matrix: visible on its own, no cursor
          // needed. The slow wave only varies it slightly so it feels alive.
          const wave = animate ? 0.3 + 0.1 * Math.sin(x * 0.03 + y * 0.025 + t) : 0.32;
          let boost = 0;
          if (animate) {
            const dx = x - mouse.x;
            const dy = y - mouse.y;
            const dist = Math.hypot(dx, dy);
            // Smooth falloff (eased) so the lit area reads larger and softer.
            const linear = dist < radius ? 1 - dist / radius : 0;
            const near = linear * linear * (3 - 2 * linear); // smoothstep
            const key = `${x},${y}`;
            const prev = heat.get(key) ?? 0;
            boost = Math.max(near, prev * 0.92);
            if (boost > 0.015) heat.set(key, boost);
            else heat.delete(key);
          }
          const level = Math.min(0.999, wave + boost * 0.7);
          const char = RAMP[Math.floor(level * (RAMP.length - 1))];
          if (char === ' ') continue;
          const r = Math.round(0x3a + (0x7d - 0x3a) * boost);
          const g = Math.round(0x3a + (0xf9 - 0x3a) * boost);
          const b = Math.round(0x40 + (0xc4 - 0x40) * boost);
          // Alpha floor keeps the whole field legible; the cursor only adds on
          // top of that, so nothing reads as a dark halo away from the pointer.
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.46 + boost * 0.5})`;
          ctx.fillText(char, x, y);
        }
      }
    };

    if (reduced) {
      frame(0, false);
      return () => resizeObserver.disconnect();
    }

    const onMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = event.clientX - rect.left;
      mouse.y = event.clientY - rect.top;
    };
    const onLeave = () => {
      mouse.x = -9999;
      mouse.y = -9999;
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerleave', onLeave);

    const loop = (time: number) => {
      frame(time, true);
      raf = requestAnimationFrame(loop);
    };

    const intersection = new IntersectionObserver(
      ([entry]) => {
        const nowVisible = entry.isIntersecting;
        if (nowVisible && !visible) raf = requestAnimationFrame(loop);
        if (!nowVisible && visible) cancelAnimationFrame(raf);
        visible = nowVisible;
      },
      { threshold: 0 },
    );
    intersection.observe(canvas);
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      intersection.disconnect();
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerleave', onLeave);
    };
  }, [cell, radius, reduced]);

  return <canvas ref={ref} className={className} aria-hidden="true" />;
}

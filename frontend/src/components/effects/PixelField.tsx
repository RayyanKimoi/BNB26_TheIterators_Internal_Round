/**
 * Ambient pixel field for the landing and auth backgrounds.
 *
 * A self-contained take on the "closing plasma" + "pixel canvas" idea, locked to
 * the project palette: a grid of cells shimmering on a slow plasma wave in the
 * border grey, lighting up toward mint under the pointer and trailing off. No
 * purple, no WebGL, no dependencies. Under prefers-reduced-motion it paints a
 * single static grid and starts no animation loop.
 */

import { useEffect, useRef } from 'react';
import { useReducedMotion } from 'framer-motion';

export interface PixelFieldProps {
  className?: string;
  /** Cell spacing in CSS pixels. Larger is sparser and cheaper. */
  gap?: number;
  /** Pointer glow radius in CSS pixels. */
  radius?: number;
}

export function PixelField({ className = '', gap = 26, radius = 130 }: PixelFieldProps) {
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
    const mouse = { x: -9999, y: -9999 };
    const heat = new Map<string, number>();

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    const paintStatic = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'rgba(31, 31, 35, 0.55)';
      for (let y = gap / 2; y < height; y += gap) {
        for (let x = gap / 2; x < width; x += gap) {
          ctx.fillRect(x - 1, y - 1, 2, 2);
        }
      }
    };

    if (reduced) {
      paintStatic();
      return () => observer.disconnect();
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

    const draw = (t: number) => {
      const time = t * 0.001;
      ctx.clearRect(0, 0, width, height);
      for (let y = gap / 2; y < height; y += gap) {
        for (let x = gap / 2; x < width; x += gap) {
          const ambient = 0.05 + 0.045 * Math.sin(x * 0.02 + y * 0.018 + time * 0.8);
          const dx = x - mouse.x;
          const dy = y - mouse.y;
          const dist = Math.hypot(dx, dy);
          const near = dist < radius ? 1 - dist / radius : 0;
          const key = `${x},${y}`;
          const prev = heat.get(key) ?? 0;
          const current = Math.max(near * near, prev * 0.92);
          if (current > 0.012) heat.set(key, current);
          else heat.delete(key);

          const intensity = Math.min(1, ambient + current);
          const size = 1.6 + current * 2.6;
          const r = Math.round(0x1f + (0x7d - 0x1f) * current);
          const g = Math.round(0x1f + (0xf9 - 0x1f) * current);
          const b = Math.round(0x23 + (0xc4 - 0x23) * current);
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${intensity})`;
          ctx.fillRect(x - size / 2, y - size / 2, size, size);
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerleave', onLeave);
    };
  }, [gap, radius, reduced]);

  return <canvas ref={ref} className={className} aria-hidden="true" />;
}

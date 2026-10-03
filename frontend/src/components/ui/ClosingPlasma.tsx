/**
 * Soft plasma backdrop for the auth screen.
 *
 * Classic layered-sine plasma computed on a tiny offscreen buffer, then upscaled
 * with smoothing so it reads as a blurred glow for near-zero cost. Palette is
 * locked to black and mint (intensity is capped low), so there is no purple and
 * no gradient banding toward it. Holds a single static frame under reduced
 * motion.
 */

import { useEffect, useRef } from 'react';
import { useReducedMotion } from 'framer-motion';

export interface ClosingPlasmaProps {
  className?: string;
}

export function ClosingPlasma({ className = '' }: ClosingPlasmaProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const buffer = document.createElement('canvas');
    const bctx = buffer.getContext('2d');
    if (!bctx) return;

    const DOWNSCALE = 10;
    let width = 0;
    let height = 0;
    let bw = 0;
    let bh = 0;
    let raf = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = Math.max(1, Math.floor(rect.width));
      height = Math.max(1, Math.floor(rect.height));
      canvas.width = width;
      canvas.height = height;
      bw = Math.max(1, Math.floor(width / DOWNSCALE));
      bh = Math.max(1, Math.floor(height / DOWNSCALE));
      buffer.width = bw;
      buffer.height = bh;
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    const render = (time: number, animate: boolean) => {
      const t = animate ? time * 0.0006 : 0;
      const image = bctx.createImageData(bw, bh);
      const data = image.data;
      for (let y = 0; y < bh; y++) {
        for (let x = 0; x < bw; x++) {
          const v =
            Math.sin(x * 0.28 + t) +
            Math.sin(y * 0.24 - t * 0.8) +
            Math.sin((x + y) * 0.18 + t * 0.5) +
            Math.sin(Math.sqrt(x * x + y * y) * 0.3 - t);
          const n = (v + 4) / 8;
          const m = Math.pow(n, 2.2) * 0.5;
          const i = (y * bw + x) * 4;
          data[i] = Math.round(0x0a + (0x7d - 0x0a) * m);
          data[i + 1] = Math.round(0x0a + (0xf9 - 0x0a) * m);
          data[i + 2] = Math.round(0x0b + (0xc4 - 0x0b) * m);
          data[i + 3] = 255;
        }
      }
      bctx.putImageData(image, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(buffer, 0, 0, bw, bh, 0, 0, width, height);
    };

    if (reduced) {
      render(0, false);
      return () => observer.disconnect();
    }

    const loop = (time: number) => {
      render(time, true);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, [reduced]);

  return <canvas ref={ref} className={className} aria-hidden="true" />;
}

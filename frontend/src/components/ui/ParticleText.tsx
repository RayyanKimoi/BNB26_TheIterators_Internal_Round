/**
 * Cursor-driven particle typography.
 *
 * Renders the given text to an offscreen canvas, samples the opaque pixels into
 * particles, then animates them: the pointer pushes nearby particles away and a
 * spring pulls every particle back to its home position. Self-contained, no
 * dependencies. Falls back to plain styled text under reduced motion or on
 * narrow screens, where a fixed-width canvas would overflow.
 */

import { useEffect, useRef, useState } from 'react';
import { useReducedMotion } from 'framer-motion';

export interface ParticleTextProps {
  text: string;
  className?: string;
  fontSize?: number;
  color?: string;
  weight?: number;
  fontFamily?: string;
}

interface Particle {
  hx: number;
  hy: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export function ParticleText({
  text,
  className = '',
  fontSize = 84,
  color = '#e8e8e8',
  weight = 700,
  fontFamily = "'Space Grotesk', ui-sans-serif, sans-serif",
}: ParticleTextProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();
  const [staticMode, setStaticMode] = useState(true);

  useEffect(() => {
    // Only run the fixed-width canvas where the container is wide enough for a
    // large wordmark; below that fall back to responsive text.
    const narrow = window.matchMedia('(max-width: 1023px)');
    const update = () => setStaticMode(reduced || narrow.matches);
    update();
    narrow.addEventListener('change', update);
    return () => narrow.removeEventListener('change', update);
  }, [reduced]);

  useEffect(() => {
    if (staticMode) return;
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const font = `${weight} ${fontSize}px ${fontFamily}`;

    ctx.font = font;
    const metrics = ctx.measureText(text);
    const pad = fontSize * 0.28;
    const cssW = Math.ceil(metrics.width + pad * 2);
    const cssH = Math.ceil(fontSize * 1.35);

    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    ctx.scale(dpr, dpr);

    ctx.font = font;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, pad, cssH / 2);

    const image = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const particles: Particle[] = [];
    const step = Math.max(3, Math.round(fontSize / 24));
    for (let y = 0; y < cssH; y += step) {
      for (let x = 0; x < cssW; x += step) {
        const alpha = image[(Math.floor(y * dpr) * canvas.width + Math.floor(x * dpr)) * 4 + 3];
        if (alpha > 128) {
          particles.push({
            hx: x,
            hy: y,
            x: x + (Math.random() - 0.5) * 60,
            y: y + (Math.random() - 0.5) * 60,
            vx: 0,
            vy: 0,
          });
        }
      }
    }

    const mouse = { x: -9999, y: -9999 };
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

    const pushRadius = 62;
    const disperse = 13;
    const returnPull = 0.07;
    const friction = 0.85;
    let raf = 0;

    const draw = () => {
      ctx.clearRect(0, 0, cssW, cssH);
      ctx.fillStyle = color;
      for (const p of particles) {
        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist = Math.hypot(dx, dy) || 0.001;
        if (dist < pushRadius) {
          const force = (1 - dist / pushRadius) * disperse;
          p.vx += (dx / dist) * force;
          p.vy += (dy / dist) * force;
        }
        p.vx += (p.hx - p.x) * returnPull;
        p.vy += (p.hy - p.y) * returnPull;
        p.vx *= friction;
        p.vy *= friction;
        p.x += p.vx;
        p.y += p.vy;
        ctx.fillRect(p.x, p.y, 1.7, 1.7);
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerleave', onLeave);
    };
  }, [text, fontSize, color, weight, fontFamily, staticMode]);

  if (staticMode) {
    return (
      <span
        className={className}
        style={{
          fontSize: `clamp(44px, 13vw, ${fontSize}px)`,
          fontWeight: weight,
          fontFamily,
          color,
          lineHeight: 1.02,
        }}
      >
        {text}
      </span>
    );
  }

  return <canvas ref={ref} className={className} role="img" aria-label={text} />;
}

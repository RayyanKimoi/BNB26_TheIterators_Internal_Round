/**
 * Image-to-ASCII renderer (Canvas 2D, no dependencies).
 *
 * Pipeline: cover-fit sample -> grayscale -> contrast/brightness ->
 * Floyd-Steinberg dithering -> luminance-to-character, drawn on a monospace grid
 * with aspect correction (character cells are taller than wide, so the portrait
 * is not stretched). The static dithered render is cached to an offscreen canvas
 * and blitted each frame.
 *
 * Hover interaction ("flow"): cells within a radius of the pointer sample the
 * source at a displaced, time-animated radial ripple, so the characters warp and
 * flow around the cursor. The animation loop runs only while the pointer is over
 * the canvas and the canvas is on screen; otherwise it paints the static render
 * and stops. Under prefers-reduced-motion there is no loop at all.
 *
 * Everything that affects the look is a prop: palette, density (fontSize),
 * contrast, brightness, dithering, flow strength/speed, colour and opacity.
 */

import { useEffect, useRef } from 'react';
import { useReducedMotion } from 'framer-motion';

export interface AsciiImageProps {
  src: string;
  className?: string;
  /** Character cell height in px. Smaller is denser / higher resolution. */
  fontSize?: number;
  /** Palette ordered darkest -> brightest. Brighter image -> denser glyph. */
  chars?: string;
  /** Contrast multiplier applied around mid-grey before mapping. */
  contrast?: number;
  /** Brightness multiplier applied after contrast. */
  brightness?: number;
  /** Floyd-Steinberg error diffusion for a photographic, dithered look. */
  dither?: boolean;
  /** Pointer influence radius in px. */
  cursorRadius?: number;
  /** Peak ripple displacement in px. 0 disables the flow entirely. */
  flowStrength?: number;
  /** Ripple animation speed. */
  flowSpeed?: number;
  /** Spatial frequency of the ripple. */
  flowFrequency?: number;
  /** Width-to-height ratio of a character cell. */
  characterAspect?: number;
  /** Canvas background; defaults to transparent so the panel shows through. */
  backgroundColor?: string;
  opacity?: number;
}

const DEFAULT_CHARS = ' .:-=+*#%@';

export function AsciiImage({
  src,
  className = '',
  fontSize = 8,
  chars = DEFAULT_CHARS,
  contrast = 1.32,
  brightness = 1.06,
  dither = true,
  cursorRadius = 170,
  flowStrength = 14,
  flowSpeed = 1.5,
  flowFrequency = 0.05,
  characterAspect = 0.6,
  backgroundColor = 'transparent',
  opacity = 1,
}: AsciiImageProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const base = document.createElement('canvas');
    const bctx = base.getContext('2d');
    if (!bctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const charW = Math.max(2, fontSize * characterAspect);
    const charH = fontSize;
    const font = `${fontSize}px 'Geist Mono', ui-monospace, monospace`;
    const levels = Math.max(2, chars.length);

    let width = 0;
    let height = 0;
    let cols = 0;
    let rows = 0;
    let lum = new Float32Array(0);
    // The dithered character index actually drawn per cell in the static
    // render. The flow overlay reads from this (not raw `lum`) so a dark area
    // keeps its Floyd-Steinberg speckle instead of going flat black under the
    // cursor, which is what produced the vignette.
    let ditherIdx = new Int16Array(0);
    let loaded = false;
    let running = false;
    let visible = true;
    let raf = 0;
    const mouse = { x: -9999, y: -9999, active: false };

    const sampleDitherIdx = (fx: number, fy: number): number => {
      const x = Math.min(cols - 1, Math.max(0, Math.round(fx)));
      const y = Math.min(rows - 1, Math.max(0, Math.round(fy)));
      return ditherIdx[y * cols + x];
    };

    const build = () => {
      if (!loaded) return;
      const rect = canvas.getBoundingClientRect();
      width = Math.max(1, Math.floor(rect.width));
      height = Math.max(1, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cols = Math.max(1, Math.floor(width / charW));
      rows = Math.max(1, Math.floor(height / charH));

      // Cover-fit the source into the panel's display aspect (width/height),
      // then sample into the cols x rows grid. This keeps the portrait correct
      // once each cell is drawn back at charW x charH.
      const sample = document.createElement('canvas');
      sample.width = cols;
      sample.height = rows;
      const sctx = sample.getContext('2d');
      if (!sctx) return;
      const displayRatio = width / height;
      const imgRatio = img.width / img.height;
      let sw: number;
      let sh: number;
      let sx: number;
      let sy: number;
      if (imgRatio > displayRatio) {
        sh = img.height;
        sw = img.height * displayRatio;
        sx = (img.width - sw) / 2;
        sy = 0;
      } else {
        sw = img.width;
        sh = img.width / displayRatio;
        sx = 0;
        sy = (img.height - sh) / 2;
      }
      sctx.drawImage(img, sx, sy, sw, sh, 0, 0, cols, rows);
      const data = sctx.getImageData(0, 0, cols, rows).data;

      lum = new Float32Array(cols * rows);
      for (let i = 0; i < cols * rows; i++) {
        let v = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255;
        v = (v - 0.5) * contrast + 0.5;
        v *= brightness;
        lum[i] = Math.min(1, Math.max(0, v));
      }

      // Static render, with optional Floyd-Steinberg dithering for smooth tone.
      base.width = Math.floor(width * dpr);
      base.height = Math.floor(height * dpr);
      bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      bctx.clearRect(0, 0, width, height);
      bctx.font = font;
      bctx.textBaseline = 'top';
      bctx.textAlign = 'left';

      ditherIdx = new Int16Array(cols * rows);
      const work = Float32Array.from(lum);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c;
          const value = Math.min(1, Math.max(0, work[i]));
          const idx = Math.round(value * (levels - 1));
          ditherIdx[i] = idx;
          if (dither) {
            const err = value - idx / (levels - 1);
            if (c + 1 < cols) work[i + 1] += (err * 7) / 16;
            if (r + 1 < rows) {
              if (c > 0) work[i + cols - 1] += (err * 3) / 16;
              work[i + cols] += (err * 5) / 16;
              if (c + 1 < cols) work[i + cols + 1] += (err * 1) / 16;
            }
          }
          const char = chars[idx];
          if (char === ' ') continue;
          const level = idx / (levels - 1);
          const g = Math.round(58 + level * 190);
          bctx.fillStyle = `rgb(${g}, ${g}, ${Math.min(255, Math.round(g * 1.04))})`;
          bctx.fillText(char, c * charW, r * charH);
        }
      }
    };

    const blitBase = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(base, 0, 0, base.width, base.height, 0, 0, width, height);
    };

    const frame = (time: number) => {
      if (!visible || !mouse.active || reduced) {
        blitBase();
        running = false;
        return;
      }
      blitBase();

      const t = time * 0.001 * flowSpeed;
      const rad = cursorRadius;
      const c0 = Math.max(0, Math.floor((mouse.x - rad) / charW));
      const c1 = Math.min(cols - 1, Math.ceil((mouse.x + rad) / charW));
      const r0 = Math.max(0, Math.floor((mouse.y - rad) / charH));
      const r1 = Math.min(rows - 1, Math.ceil((mouse.y + rad) / charH));

      // Clear the affected rect so the flowing glyphs replace the static ones.
      ctx.clearRect(c0 * charW, r0 * charH, (c1 - c0 + 1) * charW, (r1 - r0 + 1) * charH);
      ctx.font = font;
      ctx.textBaseline = 'top';
      ctx.textAlign = 'left';

      // EVERY cell in the cleared rect is repainted, including the corners
      // outside the ripple's radius. Skipping those (an earlier version did)
      // left them erased, which is what read as a dark square frame tracking
      // the cursor: the clear is a rectangle, so the repaint has to be one
      // too. Outside the radius the displacement is simply zero, so those
      // cells come back identical to the static render.
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const px = c * charW;
          const py = r * charH;
          const dx = px - mouse.x;
          const dy = py - mouse.y;
          const dist = Math.hypot(dx, dy);
          const inside = dist <= rad;

          let idx: number;
          let fall = 0;
          if (inside) {
            fall = 1 - dist / rad;
            const amp = fall * fall * flowStrength * Math.sin(dist * flowFrequency - t * 3);
            const ux = dx / (dist || 1);
            const uy = dy / (dist || 1);
            // Sample the already-dithered grid from a displaced position, so
            // the speckle texture flows with the cursor instead of vanishing.
            idx = sampleDitherIdx((px + ux * amp) / charW, (py + uy * amp) / charH);
          } else {
            idx = sampleDitherIdx(c, r);
          }

          const char = chars[idx];
          if (char === ' ') continue;
          const level = idx / (levels - 1);
          // Outside the radius this is exactly the static render's shade, so
          // the boundary is invisible; inside, the lift scales with falloff.
          const g = inside
            ? Math.min(255, Math.round(90 + level * 150 + fall * 40))
            : Math.round(58 + level * 190);
          ctx.fillStyle = `rgb(${g}, ${g}, ${Math.min(255, Math.round(g * 1.04))})`;
          ctx.fillText(char, px, py);
        }
      }
      raf = requestAnimationFrame(frame);
    };

    const start = () => {
      if (running || reduced || !visible || !mouse.active) return;
      running = true;
      raf = requestAnimationFrame(frame);
    };

    const img = new Image();
    img.onload = () => {
      loaded = true;
      build();
      blitBase();
    };
    img.src = src;

    const resizeObserver = new ResizeObserver(() => {
      build();
      if (!running) blitBase();
    });
    resizeObserver.observe(canvas);

    const onMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const inside = x >= 0 && y >= 0 && x <= rect.width && y <= rect.height;
      mouse.x = x;
      mouse.y = y;
      mouse.active = inside;
      if (inside) start();
    };
    const onLeave = () => {
      mouse.active = false;
    };

    let intersection: IntersectionObserver | null = null;
    if (!reduced) {
      window.addEventListener('pointermove', onMove, { passive: true });
      window.addEventListener('pointerleave', onLeave);
      intersection = new IntersectionObserver(
        ([entry]) => {
          visible = entry.isIntersecting;
          if (!visible) {
            running = false;
            cancelAnimationFrame(raf);
          }
        },
        { threshold: 0 },
      );
      intersection.observe(canvas);
    }

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      intersection?.disconnect();
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerleave', onLeave);
    };
  }, [
    src,
    fontSize,
    chars,
    contrast,
    brightness,
    dither,
    cursorRadius,
    flowStrength,
    flowSpeed,
    flowFrequency,
    characterAspect,
    reduced,
  ]);

  return (
    <canvas
      ref={ref}
      className={className}
      style={{ backgroundColor, opacity }}
      role="img"
      aria-label="Interactive ASCII rendering of a classical marble bust"
    />
  );
}

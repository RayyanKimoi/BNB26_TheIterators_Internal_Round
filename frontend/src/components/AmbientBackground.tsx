/**
 * Ambient plasma glow behind the whole app.
 *
 * Replaces the earlier ASCII grid. Decoration only: `fixed`,
 * `pointer-events-none` and `aria-hidden`, so it cannot intercept a click,
 * take focus, be announced, or affect layout.
 *
 * Built on the project's existing `ui/ClosingPlasma` rather than installing
 * componentry's closing-plasma package, for four reasons:
 *
 * 1. That package's palette is navy and slate (#0d0d14 / #1f2540 / #4a6191).
 *    CLAUDE.md bans purple and fixes the accent at #7DF9C4, and the request
 *    was to recolour it green anyway. ClosingPlasma is already locked to
 *    black and mint with its intensity capped, so there is nothing to undo.
 * 2. This repo is on npm, not pnpm, so the suggested `pnpm dlx` install would
 *    not have matched the lockfile.
 * 3. ClosingPlasma already holds a single static frame under
 *    prefers-reduced-motion, which CLAUDE.md requires.
 * 4. It renders onto a 10x downscaled offscreen buffer and upscales with
 *    smoothing, so a full-viewport glow costs roughly a 192x108 canvas. A
 *    full-resolution plasma behind every page would not be free.
 *
 * Opacity is deliberately far below the auth screen's 0.7: there it is the
 * feature, here it sits behind real data and must lose to it.
 */

import { ClosingPlasma } from './ui/ClosingPlasma';

/** Low enough to read as atmosphere, high enough to actually be visible. */
const LAYER_OPACITY = 0.18;

export function AmbientBackground() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden"
      style={{ opacity: LAYER_OPACITY }}
    >
      <ClosingPlasma className="h-full w-full" />
    </div>
  );
}

/**
 * Thin scroll progress bar pinned to the top of the viewport.
 *
 * The storytelling pattern asks for a progress indicator so a long page stays
 * legible. Driven by scaleX (a transform, GPU friendly) off a spring-smoothed
 * scroll position, so it never animates layout.
 */

import { motion, useScroll, useSpring } from 'framer-motion';

export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 30, mass: 0.3 });

  return (
    <motion.div
      style={{ scaleX }}
      className="fixed inset-x-0 top-0 z-50 h-0.5 origin-left bg-accent"
      aria-hidden="true"
    />
  );
}

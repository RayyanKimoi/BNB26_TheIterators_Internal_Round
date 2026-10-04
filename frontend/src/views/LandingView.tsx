/**
 * Long-form developer landing page.
 *
 * A scroll-triggered narrative: hero hook, a velocity ticker, the four-layer
 * deep dive, a scroll-driven 3D platform gallery, and a closing quote. Every
 * effect is a self-contained module in components/ui, locked to the project
 * palette (no purple), and each degrades cleanly under reduced motion. The copy
 * reads without any of the motion.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';

import { AnnotatedText } from '../components/ui/AnnotatedText';
import { AsciiReveal } from '../components/ui/AsciiReveal';
import { ClosingNotes } from '../components/ui/ClosingNotes';
import { FeatureDeepDive } from '../components/ui/FeatureDeepDive';
import { FooterQuote } from '../components/ui/FooterQuote';
import { ScrollProgress } from '../components/ui/ScrollProgress';
import { SpiralGallery } from '../components/ui/SpiralGallery';
import { TraceStatusCard } from '../components/ui/TraceStatusCard';
import { ParticleText } from '../components/ui/ParticleText';
import { VelocityScroll } from '../components/ui/VelocityScroll';
import { useAuth } from '../context/authCore';

export function LandingView({ onSignIn }: { onSignIn: () => void }) {
  const { loginAsGuest } = useAuth();
  const reduced = useReducedMotion();

  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.08, delayChildren: 0.05 } },
  };
  const item: Variants = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 14 },
    show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: 'easeOut' } },
  };

  return (
    <div className="relative bg-bg text-text">
      <ScrollProgress />

      {/* hero */}
      <section className="relative min-h-screen overflow-hidden">
        <AsciiReveal className="absolute inset-0 h-full w-full opacity-70" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-bg/20 via-bg/55 to-bg" />

        <div className="relative mx-auto max-w-[1280px] px-4 sm:px-6">
          <header className="flex items-center justify-between py-5">
            <div className="flex items-center gap-2">
              <span className="dot-matrix hairline block size-5" aria-hidden="true" />
              <span className="font-display text-sm text-text">Black Box</span>
            </div>
            <button
              type="button"
              onClick={onSignIn}
              className="data hairline rounded px-3 py-1.5 text-text transition-colors hover:border-accent/60 hover:text-accent"
            >
              Sign In
            </button>
          </header>

          <div className="grid items-center gap-12 py-14 lg:grid-cols-[1.15fr_0.85fr] lg:py-24">
            <motion.div variants={container} initial="hidden" animate="show">
              <motion.span
                variants={item}
                className="data inline-block rounded-full border border-accent/30 bg-accent/5 px-3 py-1.5 text-[13px] text-accent"
              >
                Sentry for AI Agents
              </motion.span>

              <motion.div variants={item} className="mt-6 min-h-[88px] lg:min-h-[120px]">
                <ParticleText text="BLACK BOX" fontSize={104} color="#e8e8e8" className="block" />
              </motion.div>

              <motion.h1
                variants={item}
                className="mt-4 font-display text-3xl leading-tight text-text sm:text-[40px] lg:text-[46px]"
              >
                Find the <AnnotatedText variant="underline">exact step</AnnotatedText> that broke your
                agent, then <AnnotatedText variant="circle">fork the fix</AnnotatedText>.
              </motion.h1>

              <motion.p
                variants={item}
                className="mt-6 max-w-xl text-[17px] leading-relaxed text-muted sm:text-lg"
              >
                Distribution-relative fault localization, root cause analysis with Gemini, and suffix
                step replay.
              </motion.p>

              <motion.div variants={item} className="mt-9 flex flex-wrap items-center gap-4">
                <button
                  type="button"
                  onClick={onSignIn}
                  className="rounded border border-accent bg-accent/10 px-6 py-3.5 text-[16px] font-medium text-accent transition-colors hover:bg-accent/20 hover:shadow-[0_0_32px_-10px_var(--color-accent)]"
                >
                  Launch Dashboard / Sign In
                </button>
                <button
                  type="button"
                  onClick={loginAsGuest}
                  className="hairline rounded px-6 py-3.5 text-[16px] text-text transition-colors hover:border-accent/60 hover:text-accent"
                >
                  Instant Demo Access (1-Click)
                </button>
              </motion.div>
            </motion.div>

            <motion.div
              initial={reduced ? { opacity: 0 } : { opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: 'easeOut', delay: 0.2 }}
              className="flex justify-center lg:justify-end"
            >
              <TraceStatusCard />
            </motion.div>
          </div>
        </div>
      </section>

      <VelocityScroll />

      <div className="mx-auto max-w-[1280px] px-4 sm:px-6">
        <FeatureDeepDive />
        <SpiralGallery />
        <ClosingNotes />
        <FooterQuote onGetStarted={loginAsGuest} />
      </div>
    </div>
  );
}

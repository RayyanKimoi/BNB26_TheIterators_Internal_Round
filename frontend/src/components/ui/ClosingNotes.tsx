/**
 * Closing section for the landing page: what the system actually claims, and
 * what it does not.
 *
 * The numbers quoted here are the measured ones from
 * model/artifacts/evaluation.json. The limitations are real and documented in
 * PROJECT_STATUS.md; stating them on the landing page rather than burying
 * them is the whole argument the project makes about itself.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';

interface Note {
  heading: string;
  body: string;
}

const CLAIMS: Note[] = [
  {
    heading: 'It localizes, it does not detect',
    body: 'Point it at a run you already know failed and it tells you which step broke it. Point it at a passing run and it will still flag something, because that is not the question it was trained to answer.',
  },
  {
    heading: 'The classifier alone does not generalize',
    body: 'On failure classes held out of training entirely, the trained model scores 7.5 percent, exactly matching a blame-the-last-step baseline. That number is in the product, not hidden behind it.',
  },
  {
    heading: 'The fallback is what carries the unseen cases',
    body: 'A distribution-relative tier flags steps that are statistically extreme against everything the model trained on, then declines to name a class it has never seen. That is what takes held-out localization to 52.5 percent.',
  },
  {
    heading: 'A fork is a replay, not a rerun',
    body: 'Patching a step replays the suffix deterministically from a stored checkpoint. No agent is re-executed and no tokens are spent, which also means a fork can never shorten a trace, so a runaway loop stays a runaway loop.',
  },
];

export function ClosingNotes() {
  const reduced = useReducedMotion();

  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.08 } },
  };
  const item: Variants = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 14 },
    show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: 'easeOut' } },
  };

  return (
    <motion.section
      variants={container}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.2 }}
      className="py-24"
    >
      <motion.div variants={item} className="max-w-2xl">
        <span className="data rounded-full border border-accent/30 bg-accent/5 px-2.5 py-1 text-accent">
          What this actually claims
        </span>
        <h2 className="mt-5 font-display text-3xl leading-tight text-text sm:text-4xl">
          Most agent tooling shows you what happened. This one commits to where it went wrong.
        </h2>
        <p className="mt-5 text-[17px] leading-relaxed text-muted">
          A trace tells you an agent failed. It rarely tells you which of its nineteen steps was
          the one that mattered, and the last error in the log is usually a symptom two steps
          downstream of the cause. Black Box scores every step, names the step it blames, and
          shows the evidence behind the call so you can disagree with it.
        </p>
      </motion.div>

      <div className="mt-10 grid gap-3 sm:grid-cols-2">
        {CLAIMS.map((note) => (
          <motion.div
            key={note.heading}
            variants={item}
            className="hairline glow-accent bg-panel/70 p-5 backdrop-blur"
          >
            <h3 className="font-display text-[15px] text-text">{note.heading}</h3>
            <p className="mt-2 text-[13px] leading-relaxed text-muted">{note.body}</p>
          </motion.div>
        ))}
      </div>

      <motion.p variants={item} className="data mt-8 max-w-2xl leading-relaxed text-muted">
        Every figure on this page and in the dashboard is read from a held-out evaluation that
        splits the corpus by failure class, never randomly. Two of the seven classes are withheld
        from training entirely and only ever appear at test time. Where a number is weak, it is
        printed next to the strong one.
      </motion.p>
    </motion.section>
  );
}

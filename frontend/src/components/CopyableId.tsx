/**
 * Click-to-copy identifier, shared by the runs table and the trace header.
 *
 * `displayLength` truncates the shown text (the table wants a short slice);
 * omit it to show the id in full. The clipboard write can fail silently over
 * plain http or in a locked-down context, which is fine — it is a convenience,
 * not a requirement for using the page.
 */

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useState } from 'react';

export interface CopyableIdProps {
  id: string;
  /** Characters to show before truncating. Omit to show the full id. */
  displayLength?: number;
  className?: string;
}

export function CopyableId({ id, displayLength, className = '' }: CopyableIdProps) {
  const [copied, setCopied] = useState(false);
  const reduced = useReducedMotion();

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(id);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // Clipboard can be unavailable over plain http; fail quietly.
    }
  };

  return (
    <button
      type="button"
      onClick={copy}
      title={`Copy ${id}`}
      className={`data group inline-flex items-center gap-1.5 text-text transition-colors hover:text-accent ${className}`}
    >
      <span>{displayLength ? id.slice(0, displayLength) : id}</span>
      <AnimatePresence mode="wait" initial={false}>
        {copied ? (
          <motion.span
            key="done"
            initial={reduced ? { opacity: 0 } : { opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16 }}
            className="inline-flex items-center gap-1 text-accent"
          >
            <svg viewBox="0 0 16 16" className="size-3" aria-hidden="true">
              <path
                d="M3.5 8.5l3 3 6-7"
                stroke="currentColor"
                strokeWidth="1.6"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            copied
          </motion.span>
        ) : (
          <motion.span
            key="icon"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16 }}
            className="text-muted opacity-0 transition-opacity group-hover:opacity-100"
          >
            <svg viewBox="0 0 16 16" className="size-3" aria-hidden="true">
              <rect x="5" y="5" width="8" height="8" rx="1" stroke="currentColor" strokeWidth="1.3" fill="none" />
              <path d="M3 11V3h8" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinecap="round" />
            </svg>
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );
}

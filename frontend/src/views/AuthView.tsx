/**
 * Login / create-account view.
 *
 * A local demo session (see AuthContext): there is no auth server, so this is
 * transparent about that rather than imitating a real sign-in. Both tabs call
 * the same local login; the guest button skips straight to the demo session. On
 * success the auth state flips and App swaps this view for the dashboard, so no
 * manual redirect is needed here.
 */

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useState } from 'react';
import type { FormEvent } from 'react';

import { ClosingPlasma } from '../components/ui/ClosingPlasma';
import { useAuth } from '../context/authCore';

type Mode = 'signin' | 'signup';

export function AuthView({ onBack }: { onBack: () => void }) {
  const { login, register, loginAsGuest } = useAuth();
  const reduced = useReducedMotion();

  const [mode, setMode] = useState<Mode>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === 'signup') await register(email, password);
      else await login(email, password);
      // Success unmounts this view; leave the button in its loading state.
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not sign in.');
      setLoading(false);
    }
  };

  const tabs: { id: Mode; label: string }[] = [
    { id: 'signin', label: 'Sign In' },
    { id: 'signup', label: 'Create Account' },
  ];

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-bg px-4">
      <ClosingPlasma className="pointer-events-none absolute inset-0 h-full w-full opacity-70" />
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-bg/40 via-bg/55 to-bg" />

      <button
        type="button"
        onClick={onBack}
        className="data hairline absolute left-4 top-4 z-10 inline-flex items-center gap-2 rounded bg-panel/80 px-3 py-2 text-text backdrop-blur transition-colors hover:border-accent/60 hover:text-accent sm:left-6 sm:top-6"
      >
        <svg viewBox="0 0 16 16" className="size-3.5" aria-hidden="true">
          <path
            d="M10 3l-5 5 5 5"
            stroke="currentColor"
            strokeWidth="1.6"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        Back to home
      </button>

      <motion.div
        initial={reduced ? { opacity: 0 } : { opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="hairline relative w-full max-w-md bg-panel/90 p-7 backdrop-blur"
      >
        <div className="flex items-center gap-2">
          <span className="dot-matrix hairline block size-5" aria-hidden="true" />
          <span className="font-display text-sm text-text">Black Box</span>
        </div>

        <div className="mt-5 flex gap-1" role="tablist" aria-label="Authentication mode">
          {tabs.map((tab) => {
            const active = tab.id === mode;
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => {
                  setMode(tab.id);
                  setError(null);
                }}
                className="relative flex-1 rounded px-3 py-2 text-[13px] transition-colors"
              >
                {active && (
                  <motion.span
                    layoutId="auth-tab"
                    className="absolute inset-0 rounded border border-accent/40 bg-accent/10"
                    transition={{ duration: reduced ? 0 : 0.2 }}
                  />
                )}
                <span className={`relative z-10 ${active ? 'text-accent' : 'text-muted'}`}>
                  {tab.label}
                </span>
              </button>
            );
          })}
        </div>

        <form onSubmit={submit} className="mt-5 flex flex-col gap-3">
          <label className="data flex flex-col gap-1.5 text-muted">
            Email
            <input
              type="email"
              value={email}
              autoComplete="email"
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@team.dev"
              className="hairline rounded bg-bg px-3 py-2 font-mono text-[13px] text-text placeholder:text-muted focus:border-accent/60 focus-visible:outline-none"
            />
          </label>

          <label className="data flex flex-col gap-1.5 text-muted">
            Password
            <input
              type="password"
              value={password}
              autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === 'signup' ? 'at least 6 characters' : 'your password'}
              className="hairline rounded bg-bg px-3 py-2 font-mono text-[13px] text-text placeholder:text-muted focus:border-accent/60 focus-visible:outline-none"
            />
          </label>

          {mode === 'signin' && (
            <p className="data -mt-1 text-muted">
              No account yet? Switch to Create Account above.
            </p>
          )}

          <AnimatePresence initial={false}>
            {error && (
              <motion.p
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="data text-critical"
                role="alert"
              >
                {error}
              </motion.p>
            )}
          </AnimatePresence>

          <button
            type="submit"
            disabled={loading}
            className="mt-1 rounded border border-accent bg-accent/10 px-4 py-2.5 text-[14px] font-medium text-accent transition-colors hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading
              ? mode === 'signup'
                ? 'Creating account...'
                : 'Signing in...'
              : mode === 'signin'
                ? 'Sign In'
                : 'Create Account'}
          </button>
        </form>

        <div className="my-4 flex items-center gap-3">
          <span className="h-px flex-1 bg-border" />
          <span className="data text-muted">or</span>
          <span className="h-px flex-1 bg-border" />
        </div>

        <button
          type="button"
          onClick={loginAsGuest}
          className="hairline w-full rounded px-4 py-2.5 text-[14px] text-text transition-colors hover:border-accent/60 hover:text-accent"
        >
          Instant Demo Access (Guest)
        </button>

        <p className="data mt-4 leading-relaxed text-muted">
          Local demo session: accounts are stored hashed in this browser, not on a server.
          Nothing you type leaves your device.
        </p>
      </motion.div>
    </div>
  );
}

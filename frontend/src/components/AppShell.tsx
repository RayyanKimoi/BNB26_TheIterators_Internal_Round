/**
 * Application shell: header, navigation, content frame, toast host.
 *
 * Layout, not content. Each tab renders its own screen inside `children`, so
 * this file should stay free of anything run-specific.
 */

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { API_BASE_URL, api } from '../api/client';
import { TopNav } from './TopNav';
import { TABS } from '../config/tabs';
import type { TabId } from '../config/tabs';
import { useAuth } from '../context/authCore';
import { ToastContext } from '../hooks/useToast';
import type { Toast, ToastApi, ToastKind } from '../hooks/useToast';

/* -------------------------------------------------------------------------
 * Toasts
 * ---------------------------------------------------------------------- */

const TOAST_TTL_MS = 6000;

const TOAST_STYLES: Record<ToastKind, string> = {
  info: 'border-border text-text',
  success: 'border-pass/40 text-pass',
  error: 'border-critical/50 text-critical',
};

function ToastHost({ toasts, dismiss }: { toasts: Toast[]; dismiss: (id: number) => void }) {
  const reduced = useReducedMotion();
  return (
    <div
      className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-full max-w-sm flex-col gap-2"
      role="status"
      aria-live="polite"
    >
      <AnimatePresence initial={false}>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            layout
            initial={reduced ? { opacity: 0 } : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduced ? { opacity: 0 } : { opacity: 0, y: 8 }}
            transition={{ duration: 0.24, ease: 'easeOut' }}
            className={`pointer-events-auto flex items-start gap-3 border bg-panel px-3 py-2 ${TOAST_STYLES[toast.kind]}`}
          >
            <span className="mt-[3px] block size-1.5 shrink-0 rounded-full bg-current" />
            <p className="flex-1 text-[13px] leading-snug text-text">{toast.message}</p>
            <button
              type="button"
              onClick={() => dismiss(toast.id)}
              aria-label="Dismiss notification"
              className="text-muted transition-colors hover:text-text"
            >
              <svg viewBox="0 0 16 16" className="size-3.5" aria-hidden="true">
                <path
                  d="M4 4l8 8M12 4l-8 8"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  fill="none"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Backend status
 * ---------------------------------------------------------------------- */

type Health = 'checking' | 'online' | 'offline';

const HEALTH_COPY: Record<Health, { label: string; dot: string; text: string }> = {
  checking: { label: 'Checking', dot: 'bg-muted', text: 'text-muted' },
  online: { label: 'Online', dot: 'bg-pass', text: 'text-pass' },
  offline: { label: 'Offline', dot: 'bg-critical', text: 'text-critical' },
};

function useBackendHealth(pollMs = 30_000): Health {
  const [health, setHealth] = useState<Health>('checking');

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const check = async () => {
      const ok = await api.health(controller.signal);
      if (!cancelled) setHealth(ok ? 'online' : 'offline');
    };

    void check();
    const timer = setInterval(() => void check(), pollMs);
    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(timer);
    };
  }, [pollMs]);

  return health;
}

function StatusIndicator({ health }: { health: Health }) {
  const { label, dot, text } = HEALTH_COPY[health];
  return (
    <div className="flex items-center gap-2" title={`API ${API_BASE_URL}`}>
      <span className={`block size-1.5 rounded-full ${dot}`} aria-hidden="true" />
      <span className={`data ${text}`}>{label}</span>
      <span className="data hidden text-muted lg:inline">{API_BASE_URL}</span>
    </div>
  );
}

function UserPill() {
  const { user, isDemoMode, logout } = useAuth();
  if (!user) return null;
  return (
    <div className="flex items-center gap-2">
      <span
        className="data hidden items-center gap-1.5 rounded border border-border px-2 py-1 text-muted sm:inline-flex"
        title={isDemoMode ? 'Demo session' : 'Signed in'}
      >
        <span className="block size-1.5 rounded-full bg-accent" aria-hidden="true" />
        {user.email}
        {isDemoMode ? ' · demo' : ''}
      </span>
      <button
        type="button"
        onClick={logout}
        className="data hairline rounded px-2.5 py-1 text-muted transition-colors hover:border-critical/50 hover:text-critical"
      >
        Logout
      </button>
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Shell
 * ---------------------------------------------------------------------- */

export interface AppShellProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  children: ReactNode;
}

export function AppShell({ activeTab, onTabChange, children }: AppShellProps) {
  const health = useBackendHealth();
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (kind: ToastKind, message: string) => {
      const id = nextId.current++;
      setToasts((current) => [...current, { id, kind, message }]);
      setTimeout(() => dismiss(id), TOAST_TTL_MS);
    },
    [dismiss],
  );

  const toastApi = useMemo<ToastApi>(() => ({ push, dismiss }), [push, dismiss]);
  const active = TABS.find((tab) => tab.id === activeTab) ?? TABS[0];

  // One warning when the API is unreachable, rather than one per failed call.
  const warned = useRef(false);
  useEffect(() => {
    if (health === 'offline' && !warned.current) {
      warned.current = true;
      push('error', `Cannot reach the API at ${API_BASE_URL}. Start the backend with: uvicorn backend.main:app --reload`);
    }
    if (health === 'online') warned.current = false;
  }, [health, push]);

  return (
    <ToastContext.Provider value={toastApi}>
      <div className="flex min-h-screen flex-col text-text">
        <TopNav
          activeTab={activeTab}
          onTabChange={onTabChange}
          trailing={
            <>
              <StatusIndicator health={health} />
              <span className="hidden h-5 w-px bg-border sm:block" aria-hidden="true" />
              <UserPill />
            </>
          }
        />

        <main className="w-full flex-1 px-4 py-6 sm:px-6">
          <div className="mb-5">
            <h1 className="font-display text-lg text-text">{active.label}</h1>
            <p className="text-[13px] text-muted">{active.blurb}</p>
          </div>
          {children}
        </main>

        <footer className="border-t border-border">
          <div className="flex w-full items-center justify-between px-4 py-3 sm:px-6">
            <span className="data text-muted">Sentry for AI Agents</span>
            <span className="data text-muted">
              Numbers on the Model tab are measured, never estimated
            </span>
          </div>
        </footer>

        <ToastHost toasts={toasts} dismiss={dismiss} />
      </div>
    </ToastContext.Provider>
  );
}

/* -------------------------------------------------------------------------
 * Shared states, used by every screen so they look the same everywhere
 * ---------------------------------------------------------------------- */

export function LoadingState({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="hairline flex items-center gap-3 bg-panel px-4 py-8">
      <span className="size-1.5 animate-pulse rounded-full bg-accent" aria-hidden="true" />
      <span className="data text-muted">{label}</span>
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  return (
    <div className="hairline border-critical/40 bg-panel px-4 py-6" role="alert">
      <p className="font-display text-[13px] text-critical">Request failed</p>
      <p className="data mt-2 whitespace-pre-wrap text-muted">{error.message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="hairline mt-4 px-3 py-1.5 text-[13px] text-text transition-colors hover:border-accent hover:text-accent"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="hairline bg-panel px-4 py-10 text-center">
      <p className="font-display text-[13px] text-text">{title}</p>
      {hint && <p className="data mx-auto mt-2 max-w-md text-muted">{hint}</p>}
    </div>
  );
}

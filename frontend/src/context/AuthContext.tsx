/**
 * AuthProvider: holds the local demo session and persists it to localStorage.
 *
 * Every storage access is wrapped in try/catch because localStorage can throw or
 * return null in private windows and when site data is blocked. The provider
 * renders correctly either way; a missing session simply means "logged out".
 */

import { useCallback, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

import { AuthContext, AUTH_STORAGE_KEY, DEMO_EMAIL } from './authCore';
import type { AuthContextValue } from './authCore';

interface StoredSession {
  email: string;
  role: string;
  demo: boolean;
}

function readSession(): StoredSession | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredSession>;
    if (typeof parsed.email === 'string' && typeof parsed.role === 'string') {
      return { email: parsed.email, role: parsed.role, demo: Boolean(parsed.demo) };
    }
  } catch {
    // Unreadable or blocked storage: treat as logged out.
  }
  return null;
}

function writeSession(session: StoredSession | null): void {
  try {
    if (session) localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
    else localStorage.removeItem(AUTH_STORAGE_KEY);
  } catch {
    // Storage unavailable: the in-memory state still drives the UI this session.
  }
}

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<StoredSession | null>(() => readSession());

  const login = useCallback(async (email: string, password: string) => {
    const trimmed = email.trim();
    if (!EMAIL_PATTERN.test(trimmed)) throw new Error('Enter a valid email address.');
    if (password.length < 6) throw new Error('Password must be at least 6 characters.');
    // Brief delay so the button's loading state is visible. No network call.
    await new Promise((resolve) => setTimeout(resolve, 450));
    const next: StoredSession = { email: trimmed, role: 'engineer', demo: false };
    writeSession(next);
    setSession(next);
  }, []);

  const loginAsGuest = useCallback(() => {
    const next: StoredSession = { email: DEMO_EMAIL, role: 'guest', demo: true };
    writeSession(next);
    setSession(next);
  }, []);

  const logout = useCallback(() => {
    writeSession(null);
    setSession(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user: session ? { email: session.email, role: session.role } : null,
      isAuthenticated: session !== null,
      isDemoMode: session?.demo ?? false,
      login,
      loginAsGuest,
      logout,
    }),
    [session, login, loginAsGuest, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/**
 * AuthProvider: holds the local demo session and validates credentials against
 * a local registry, both persisted to localStorage.
 *
 * Fixes the bug where any 6+ character password logged a user in: `login` no
 * longer accepts input on shape alone, it looks the email up in the registry
 * written by `register` and compares a SHA-256 hash of the password. Unknown
 * email or wrong password both fail with the same generic message, same as a
 * real auth system would, so a failed attempt never reveals which part was
 * wrong. This is still a client-side demo (no server, no salt, the hash lives
 * in this browser only) and the UI discloses that; it is not production auth.
 *
 * Every storage access is wrapped in try/catch because localStorage can throw or
 * return null in private windows and when site data is blocked.
 */

import { useCallback, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

import { AuthContext, AUTH_STORAGE_KEY, AUTH_USERS_KEY, DEMO_EMAIL } from './authCore';
import type { AuthContextValue } from './authCore';

interface StoredSession {
  email: string;
  role: string;
  demo: boolean;
}

type UserRegistry = Record<string, string>; // lowercased email -> sha-256 hex hash

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

function readUsers(): UserRegistry {
  try {
    const raw = localStorage.getItem(AUTH_USERS_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') return parsed as UserRegistry;
  } catch {
    // Unreadable or blocked storage: start from an empty registry.
  }
  return {};
}

function writeUsers(users: UserRegistry): void {
  try {
    localStorage.setItem(AUTH_USERS_KEY, JSON.stringify(users));
  } catch {
    // Storage unavailable: registration still succeeds for this tab's session,
    // it just will not survive a reload.
  }
}

/** SHA-256 hex digest via the browser's native Web Crypto. No dependency. */
async function hashPassword(password: string): Promise<string> {
  const bytes = new TextEncoder().encode(password);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
// Cosmetic delay so the button's loading state reads as a real request.
const AUTH_DELAY_MS = 400;

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<StoredSession | null>(() => readSession());

  const register = useCallback(async (email: string, password: string) => {
    const trimmed = email.trim();
    const key = trimmed.toLowerCase();
    if (!EMAIL_PATTERN.test(trimmed)) throw new Error('Enter a valid email address.');
    if (password.length < 6) throw new Error('Password must be at least 6 characters.');

    const users = readUsers();
    if (key in users) {
      throw new Error('An account with this email already exists. Sign in instead.');
    }

    const hash = await hashPassword(password);
    await new Promise((resolve) => setTimeout(resolve, AUTH_DELAY_MS));

    writeUsers({ ...users, [key]: hash });
    const next: StoredSession = { email: trimmed, role: 'engineer', demo: false };
    writeSession(next);
    setSession(next);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const trimmed = email.trim();
    const key = trimmed.toLowerCase();
    if (!trimmed || !password) throw new Error('Enter your email and password.');

    const users = readUsers();
    const stored = users[key];
    const hash = await hashPassword(password);
    await new Promise((resolve) => setTimeout(resolve, AUTH_DELAY_MS));

    // Same error whether the account does not exist or the password is wrong,
    // so a failed attempt never discloses which one it was.
    if (!stored || stored !== hash) {
      throw new Error('Invalid email or password.');
    }

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
      register,
      loginAsGuest,
      logout,
    }),
    [session, login, register, loginAsGuest, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

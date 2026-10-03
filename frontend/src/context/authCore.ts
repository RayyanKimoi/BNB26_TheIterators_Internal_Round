/**
 * Auth context primitives, kept separate from the provider component so that
 * AuthContext.tsx only exports a component (fast refresh stays happy), mirroring
 * the ToastContext / AppShell split.
 *
 * This is a LOCAL demo session, not a real auth server: there is no backend
 * users table, so `register` and `login` validate against a small credential
 * registry held in this browser's localStorage (password hashed with
 * Web Crypto, never stored in the clear). That is enough to fix "any password
 * works" without pretending to be production authentication, and the UI says
 * as much. It exists to gate the dashboard behind a landing page and to
 * support one-click demo access.
 */

import { createContext, useContext } from 'react';

export interface AuthUser {
  email: string;
  role: string;
}

export interface AuthContextValue {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isDemoMode: boolean;
  /** Verifies the password hash against the local registry. Rejects on mismatch. */
  login: (email: string, password: string) => Promise<void>;
  /** Creates a local account (hashed password), then signs in as it. */
  register: (email: string, password: string) => Promise<void>;
  /** One-click session as the shared demo user. */
  loginAsGuest: () => void;
  logout: () => void;
}

export const AuthContext = createContext<AuthContextValue | null>(null);

/** Read the auth session from anywhere below <AuthProvider>. */
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used inside <AuthProvider>');
  return context;
}

export const AUTH_STORAGE_KEY = 'blackbox.auth';
/** Local credential registry: { [lowercased email]: sha-256 hex hash }. */
export const AUTH_USERS_KEY = 'blackbox.auth.users';
export const DEMO_EMAIL = 'demo@blackbox.ai';

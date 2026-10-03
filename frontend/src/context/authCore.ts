/**
 * Auth context primitives, kept separate from the provider component so that
 * AuthContext.tsx only exports a component (fast refresh stays happy), mirroring
 * the ToastContext / AppShell split.
 *
 * This is a LOCAL demo session, not real authentication. There is no auth
 * server: credentials are validated for shape only and the session lives in
 * localStorage. The UI says as much. It exists to gate the dashboard behind a
 * landing page and to support one-click demo access, nothing more.
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
  /** Validates shape only, then stores a local session. Rejects on bad input. */
  login: (email: string, password: string) => Promise<void>;
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
export const DEMO_EMAIL = 'demo@blackbox.ai';

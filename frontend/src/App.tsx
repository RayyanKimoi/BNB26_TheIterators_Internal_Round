import { useState } from 'react';

import { AppShell, EmptyState } from './components/AppShell';
import type { TabId } from './config/tabs';
import { AuthProvider } from './context/AuthContext';
import { useAuth } from './context/authCore';
import { AuthView } from './views/AuthView';
import { LandingView } from './views/LandingView';
import { RunsView } from './views/RunsView';

/**
 * Unauthenticated visitors see the landing page, with the auth view one click
 * away. Once a session exists (real or demo) the dashboard shell takes over. The
 * runs dashboard is live (Step 2); later tabs still show a placeholder.
 */
function Root() {
  const { isAuthenticated } = useAuth();
  const [tab, setTab] = useState<TabId>('runs');
  const [screen, setScreen] = useState<'landing' | 'auth'>('landing');

  if (!isAuthenticated) {
    return screen === 'auth' ? (
      <AuthView onBack={() => setScreen('landing')} />
    ) : (
      <LandingView onSignIn={() => setScreen('auth')} />
    );
  }

  return (
    <AppShell activeTab={tab} onTabChange={setTab}>
      {tab === 'runs' ? (
        <RunsView />
      ) : (
        <EmptyState
          title="Model benchmarks not built yet"
          hint="Reads GET /model/evaluation and shows held-out accuracy against all four baselines."
        />
      )}
    </AppShell>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Root />
    </AuthProvider>
  );
}

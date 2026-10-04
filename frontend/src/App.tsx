import { Suspense, lazy, useCallback, useState } from 'react';

import { AppShell, EmptyState, LoadingState } from './components/AppShell';
import { AmbientBackground } from './components/AmbientBackground';
import type { TabId } from './config/tabs';
import { AuthProvider } from './context/AuthContext';
import { useAuth } from './context/authCore';
import { useRoute } from './lib/router';
import { AuthView } from './views/AuthView';
import { ForksView } from './views/ForksView';
import { LandingView } from './views/LandingView';
import { RunsView } from './views/RunsView';
import { SettingsView } from './views/SettingsView';
import { TraceView } from './views/TraceView';

/**
 * Insights and Model are the two views that pull in Recharts, which is
 * roughly half the bundle on its own. Splitting them out keeps the first
 * paint of the Runs tab cheap and loads the charting code only for the tabs
 * that actually draw charts.
 */
const InsightsView = lazy(() =>
  import('./views/InsightsView').then((m) => ({ default: m.InsightsView })),
);
const ModelBenchmarks = lazy(() =>
  import('./views/ModelBenchmarks').then((m) => ({ default: m.ModelBenchmarks })),
);

/**
 * Unauthenticated visitors see the landing page, with the auth view one click
 * away. Once a session exists (real or demo) the dashboard shell takes over,
 * with the six PRD tabs in the left sidebar.
 *
 * Routing is the tiny path sync in lib/router.ts rather than react-router:
 * the only real requirements are addressable tabs and a working
 * /trace/{run_id} deep link, which is what the Slack alert posts.
 */
function Root() {
  const { isAuthenticated } = useAuth();
  const [screen, setScreen] = useState<'landing' | 'auth'>('landing');
  const [route, navigate] = useRoute();

  const openTrace = useCallback(
    (runId: string) => navigate({ tab: 'trace', runId }),
    [navigate],
  );

  const onTabChange = useCallback(
    (tab: TabId) => {
      // Keep the open run when returning to Trace; drop it otherwise so the
      // address bar never claims to be showing a run that is not on screen.
      navigate(tab === 'trace' ? { tab, runId: route.runId } : { tab });
    },
    [navigate, route.runId],
  );

  if (!isAuthenticated) {
    return screen === 'auth' ? (
      <AuthView onBack={() => setScreen('landing')} />
    ) : (
      <LandingView onSignIn={() => setScreen('auth')} />
    );
  }

  return (
    <AppShell activeTab={route.tab} onTabChange={onTabChange}>
      {route.tab === 'runs' && <RunsView onOpenTrace={openTrace} />}

      {route.tab === 'trace' &&
        (route.runId ? (
          <TraceView
            key={route.runId}
            runId={route.runId}
            onBack={() => navigate({ tab: 'runs' })}
            onNavigateToRun={openTrace}
          />
        ) : (
          <EmptyState
            title="No run selected"
            hint="Pick a run from the Runs tab to see its blame heatmap, timeline and step inspector."
          />
        ))}

      {route.tab === 'forks' && <ForksView onOpenTrace={openTrace} />}
      {route.tab === 'model' && (
        <Suspense fallback={<LoadingState label="Loading charts" />}>
          <ModelBenchmarks />
        </Suspense>
      )}
      {route.tab === 'insights' && (
        <Suspense fallback={<LoadingState label="Loading charts" />}>
          <InsightsView memoryRunId={route.runId ?? null} />
        </Suspense>
      )}
      {route.tab === 'settings' && <SettingsView />}
    </AppShell>
  );
}

export default function App() {
  return (
    <AuthProvider>
      {/* Decorative, fixed and pointer-events-none, so it sits outside every
          screen rather than inside one. Mounted here so the plasma keeps a
          continuous animation across landing, auth and the dashboard instead
          of restarting on every navigation. */}
      <AmbientBackground />
      <Root />
    </AuthProvider>
  );
}

/**
 * Minimal path router, no dependency.
 *
 * The app needs real URLs for exactly two reasons: the six sidebar tabs are
 * addressable, and `backend/alerts.py` posts Slack links of the form
 * `/trace/{run_id}` that have to resolve when somebody clicks them. That is a
 * parse, a build and a history listener — not a reason to add react-router,
 * which `PRD.md` never asks for and CLAUDE.md's dependency rule discourages.
 *
 * Vite's dev server and any static host with SPA fallback serve index.html
 * for these paths, so a cold load of /trace/abc123 lands here correctly.
 */

import { useCallback, useEffect, useState } from 'react';

import { TAB_IDS } from '../config/tabs';
import type { TabId } from '../config/tabs';

export interface Route {
  tab: TabId;
  /** Only meaningful on the trace tab: the run being inspected. */
  runId?: string;
}

export const DEFAULT_ROUTE: Route = { tab: 'runs' };

function isTabId(value: string): value is TabId {
  return (TAB_IDS as readonly string[]).includes(value);
}

export function parsePath(pathname: string): Route {
  const [head, second] = pathname.replace(/^\/+|\/+$/g, '').split('/');
  if (!head) return DEFAULT_ROUTE;
  if (!isTabId(head)) return DEFAULT_ROUTE;
  if (head === 'trace' && second) return { tab: 'trace', runId: decodeURIComponent(second) };
  return { tab: head };
}

export function buildPath(route: Route): string {
  if (route.tab === 'trace' && route.runId) {
    return `/trace/${encodeURIComponent(route.runId)}`;
  }
  return `/${route.tab}`;
}

/**
 * Current route, kept in sync with the address bar in both directions:
 * `navigate` pushes, and the browser's back/forward buttons update state.
 */
export function useRoute(): [Route, (next: Route, replace?: boolean) => void] {
  const [route, setRoute] = useState<Route>(() =>
    typeof window === 'undefined' ? DEFAULT_ROUTE : parsePath(window.location.pathname),
  );

  useEffect(() => {
    const onPop = () => setRoute(parsePath(window.location.pathname));
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const navigate = useCallback((next: Route, replace = false) => {
    const path = buildPath(next);
    if (path !== window.location.pathname) {
      if (replace) window.history.replaceState(null, '', path);
      else window.history.pushState(null, '', path);
    }
    setRoute(next);
  }, []);

  return [route, navigate];
}

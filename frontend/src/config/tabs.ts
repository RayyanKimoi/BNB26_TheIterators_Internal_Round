/** Navigation config. Separate from the shell so adding a tab touches one file. */

export type TabId = 'runs' | 'model';

export interface Tab {
  id: TabId;
  label: string;
  /** Shown under the header as orientation for a first-time viewer. */
  blurb: string;
}

/**
 * The PRD specifies six tabs in a left sidebar: Runs, Trace, Forks, Model,
 * Insights, Settings. This build starts with the two that have screens behind
 * them, in a top bar. Trace and Forks are reached from inside Runs rather
 * than as peers, which keeps the navigation honest about what exists.
 */
export const TABS: Tab[] = [
  { id: 'runs', label: 'Runs & Traces', blurb: 'Every run, scored step by step' },
  { id: 'model', label: 'Model Benchmarks', blurb: 'Held-out accuracy against four baselines' },
];

/**
 * Navigation config: the six tabs from PRD.md's dashboard spec.
 *
 * PRD.md places them in a left sidebar; the product owner asked for a top
 * bar instead, so `components/TopNav.tsx` renders them horizontally. The tab
 * set and order are unchanged.
 *
 * Separate from the shell so adding a tab touches one file, and icon-free so
 * this stays a plain data module (the SVG paths live in TopNav.tsx, keyed by
 * id). CLAUDE.md bans emoji icons, so those are stroked SVG paths rather than
 * the emoji glyphs a nav like this often uses.
 */

export type TabId = 'runs' | 'trace' | 'forks' | 'model' | 'insights' | 'settings';

export const TAB_IDS = ['runs', 'trace', 'forks', 'model', 'insights', 'settings'] as const;

export interface Tab {
  id: TabId;
  label: string;
  /** Shown under the header as orientation for a first-time viewer. */
  blurb: string;
}

export const TABS: Tab[] = [
  { id: 'runs', label: 'Runs', blurb: 'Every run, scored step by step' },
  { id: 'trace', label: 'Trace', blurb: 'Per-step blame heatmap and the step inspector' },
  { id: 'forks', label: 'Forks', blurb: 'Fork lineage and side-by-side run comparison' },
  { id: 'model', label: 'Model', blurb: 'Held-out accuracy against four baselines' },
  { id: 'insights', label: 'Insights', blurb: 'Reliability, failure mix and token trends' },
  { id: 'settings', label: 'Settings', blurb: 'Webhooks, ingestion and model configuration' },
];

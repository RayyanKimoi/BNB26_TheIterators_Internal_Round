/**
 * Shared filter state for the runs dashboard.
 *
 * Kept out of FilterBar.tsx so that file only exports components, which is what
 * fast refresh needs. The table filters live entirely client side; see
 * RunsView for how they narrow the loaded run list.
 */

import type { RunStatus } from '../types/api';

/** Sentinel for "no class injected", i.e. a clean or successful run. */
export const CLEAN_CLASS = '__clean__';

export interface RunTableFilters {
  status: 'all' | RunStatus;
  taskType: string;
  faultClass: string;
  search: string;
}

export const EMPTY_FILTERS: RunTableFilters = {
  status: 'all',
  taskType: 'all',
  faultClass: 'all',
  search: '',
};

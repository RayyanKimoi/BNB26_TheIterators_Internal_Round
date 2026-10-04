/**
 * Top navigation bar: the logo, the six tabs, and the session controls.
 *
 * PRD.md specifies these six tabs in a *left sidebar*; this is a deliberate
 * deviation at the product owner's request. The tab set and their order are
 * unchanged, only the axis.
 *
 * Icons are stroked SVG, not emoji: CLAUDE.md bans emoji icons, and they also
 * render inconsistently across platforms, which is the practical reason the
 * rule exists. Under `lg` the tabs become their own horizontally scrolling
 * row so a narrow window never crushes them into the session controls.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { ReactNode } from 'react';

import { TABS } from '../config/tabs';
import type { TabId } from '../config/tabs';

const ICONS: Record<TabId, ReactNode> = {
  runs: <path d="M3 5h18M3 12h18M3 19h18" strokeLinecap="round" />,
  trace: <path d="M4 18V9M9 18V5M14 18v-6M19 18v-9" strokeLinecap="round" />,
  forks: (
    <>
      <circle cx="6" cy="5" r="2" />
      <circle cx="6" cy="19" r="2" />
      <circle cx="18" cy="12" r="2" />
      <path d="M6 7v10M8 5h4a4 4 0 0 1 4 4v1M8 19h4a4 4 0 0 0 4-4v-1" strokeLinecap="round" />
    </>
  ),
  model: (
    <>
      <path d="M4 19V5M4 19h16" strokeLinecap="round" />
      <path d="M8 16V11M12 16V7M16 16v-3" strokeLinecap="round" />
    </>
  ),
  insights: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 4v8l5 3" strokeLinecap="round" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path
        d="M19.4 14a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V20a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 9 18.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 9a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </>
  ),
};

export interface TopNavProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  /** Status dot and user pill, rendered by the shell that owns that state. */
  trailing?: ReactNode;
}

export function TopNav({ activeTab, onTabChange, trailing }: TopNavProps) {
  const reduced = useReducedMotion();

  const tabs = (
    <div className="flex items-center gap-1">
      {TABS.map((tab) => {
        const isActive = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onTabChange(tab.id)}
            aria-current={isActive ? 'page' : undefined}
            title={tab.blurb}
            className={`relative flex shrink-0 items-center gap-2 rounded px-3 py-1.5 text-[13px] transition-colors ${
              isActive ? 'text-accent' : 'text-muted hover:text-text'
            }`}
          >
            {isActive && (
              <motion.span
                layoutId="topnav-active"
                className="absolute inset-0 rounded border border-accent/40 bg-accent/10"
                transition={{ duration: reduced ? 0 : 0.2, ease: 'easeOut' }}
              />
            )}
            <svg
              viewBox="0 0 24 24"
              className="relative z-10 size-4 shrink-0"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              aria-hidden="true"
            >
              {ICONS[tab.id]}
            </svg>
            <span className="relative z-10 whitespace-nowrap">{tab.label}</span>
          </button>
        );
      })}
    </div>
  );

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-bg/95 backdrop-blur">
      <div className="flex h-14 w-full items-center gap-4 px-4 sm:px-6">
        <div className="flex shrink-0 items-center gap-2.5">
          <img
            src="/logo.png"
            alt=""
            aria-hidden="true"
            className="size-8 shrink-0 object-contain"
          />
          <div className="leading-tight">
            <p className="font-display text-[15px] text-text">Black Box</p>
            <p className="data hidden text-muted sm:block">Sentry for AI Agents</p>
          </div>
        </div>

        <nav aria-label="Primary" className="hidden lg:block">
          {tabs}
        </nav>

        <div className="ml-auto flex shrink-0 items-center gap-3 sm:gap-4">{trailing}</div>
      </div>

      {/* Narrow screens: the tabs get their own scrolling row rather than
          competing with the logo and session controls for one line. */}
      <nav
        aria-label="Primary"
        className="flex gap-1 overflow-x-auto border-t border-border px-2 py-2 lg:hidden"
      >
        {tabs}
      </nav>
    </header>
  );
}

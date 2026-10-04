/**
 * Diagnosis engine provider: a browser-local preference for which backend
 * should serve diagnoses.
 *
 * Self-contained on purpose. Nothing here touches the API client, the router
 * or any existing hook, so adding it cannot change how the app already
 * behaves. The only persisted value is the provider id, under
 * `bb_engine_provider`.
 *
 * Credentials are deliberately NOT persisted. An API key in localStorage is
 * readable by any script that reaches this origin, and a key that survives a
 * refresh is a key that survives an XSS. Callers hold them in component state
 * for the lifetime of the page and no longer, and the UI says so.
 *
 * `useSyncExternalStore` rather than a context: the Settings panel and the
 * Step Inspector badge both read this, they never share a parent, and wrapping
 * the tree in a new provider would mean editing App.tsx. A tiny store keeps
 * them in sync within the tab, and the `storage` event keeps them in sync
 * across tabs.
 */

import { useCallback, useSyncExternalStore } from 'react';

export const ENGINE_STORAGE_KEY = 'bb_engine_provider';

export type EngineProviderId =
  | 'default'
  | 'groq'
  | 'byo_llm'
  | 'enterprise_webhook'
  | 'air_gapped';

export interface EngineCredentialField {
  label: string;
  placeholder: string;
  /** `key` masks the input; `url` does not. */
  kind: 'key' | 'url';
}

export interface EngineOption {
  id: EngineProviderId;
  label: string;
  /** Compact form for the Step Inspector header, where space is tight. */
  short: string;
  /** One line under the selector. No invented numbers. */
  detail: string;
  /**
   * Whether a diagnosis actually routes to this engine today. Only the hybrid
   * engine does. The others are a stated preference with no transport behind
   * them yet, and the UI has to say that rather than imply a live switch.
   */
  routed: boolean;
  credential?: EngineCredentialField;
}

export const ENGINE_OPTIONS: EngineOption[] = [
  {
    id: 'default',
    label: 'Black Box hybrid engine',
    short: 'Black Box Hybrid',
    detail:
      'Default. 95.0% top-1 on trained classes, 52.5% on held-out classes. Runs locally, no API call.',
    routed: true,
  },
  {
    id: 'groq',
    label: 'Groq (LLM as judge)',
    short: 'Groq',
    detail:
      'Hosted Llama on Groq, running the same judge prompt the Model tab benchmarks. Scored 32.5% on held-out classes against the hybrid engine 52.5%.',
    routed: true,
    credential: {
      label: 'Groq API key',
      placeholder: 'gsk_...',
      kind: 'key',
    },
  },
  {
    id: 'byo_llm',
    label: 'Custom OpenAI / Anthropic (BYO key)',
    short: 'OpenAI / Anthropic',
    detail: 'Route diagnosis to a hosted frontier model using your own key.',
    routed: false,
    credential: {
      label: 'API key',
      placeholder: 'sk-...',
      kind: 'key',
    },
  },
  {
    id: 'enterprise_webhook',
    label: 'Custom enterprise webhook / on-prem',
    short: 'Enterprise Webhook',
    detail: 'POST the trace to your own scoring service and read the diagnosis back.',
    routed: false,
    credential: {
      label: 'Endpoint URL',
      placeholder: 'https://scoring.internal/diagnose',
      kind: 'url',
    },
  },
  {
    id: 'air_gapped',
    label: 'Local air-gapped model (vLLM / Ollama)',
    short: 'Air-Gapped Local',
    detail: 'Point at a model on your own network. Nothing leaves the perimeter.',
    routed: false,
    credential: {
      label: 'Base URL',
      placeholder: 'http://localhost:11434',
      kind: 'url',
    },
  },
];

export const DEFAULT_ENGINE: EngineProviderId = 'default';

function isEngineProviderId(value: unknown): value is EngineProviderId {
  return ENGINE_OPTIONS.some((option) => option.id === value);
}

export function getEngineOption(id: EngineProviderId): EngineOption {
  return ENGINE_OPTIONS.find((option) => option.id === id) ?? ENGINE_OPTIONS[0];
}

/* -------------------------------------------------------------------------
 * Store
 * ---------------------------------------------------------------------- */

const listeners = new Set<() => void>();

/**
 * useSyncExternalStore compares snapshots by identity and re-renders in a loop
 * if getSnapshot builds a new value each call, so the current id is cached and
 * only recomputed when something actually writes.
 */
let cached: EngineProviderId | null = null;

function readFromStorage(): EngineProviderId {
  try {
    const raw = window.localStorage.getItem(ENGINE_STORAGE_KEY);
    return isEngineProviderId(raw) ? raw : DEFAULT_ENGINE;
  } catch {
    // Private mode, blocked site data, or a sandboxed iframe. The preference
    // simply does not persist; it must never take the page down.
    return DEFAULT_ENGINE;
  }
}

function emit() {
  for (const listener of listeners) listener();
}

export function getEngineProvider(): EngineProviderId {
  if (cached === null) cached = readFromStorage();
  return cached;
}

export function setEngineProvider(id: EngineProviderId): void {
  if (cached === id) return;
  cached = id;
  try {
    window.localStorage.setItem(ENGINE_STORAGE_KEY, id);
  } catch {
    // Selection still applies for this page, it just will not survive a reload.
  }
  emit();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);

  // Another tab changed the preference. `key === null` means storage was
  // cleared wholesale, which should also resync.
  const onStorage = (event: StorageEvent) => {
    if (event.key !== null && event.key !== ENGINE_STORAGE_KEY) return;
    cached = readFromStorage();
    emit();
  };
  window.addEventListener('storage', onStorage);

  return () => {
    listeners.delete(listener);
    window.removeEventListener('storage', onStorage);
  };
}

/** Server snapshot, and the value before hydration. */
function getServerSnapshot(): EngineProviderId {
  return DEFAULT_ENGINE;
}

export function useEngineProvider(): [EngineProviderId, (id: EngineProviderId) => void] {
  const provider = useSyncExternalStore(subscribe, getEngineProvider, getServerSnapshot);
  const select = useCallback((id: EngineProviderId) => setEngineProvider(id), []);
  return [provider, select];
}

/* -------------------------------------------------------------------------
 * Credential, in memory only
 * ---------------------------------------------------------------------- */

/**
 * The BYO key for the selected provider.
 *
 * A module variable, deliberately not localStorage and not React state: it has
 * to outlive navigating from Settings to Trace, and it must NOT outlive a
 * reload. A key in browser storage is readable by anything that reaches this
 * origin, so this one is gone the moment the tab is refreshed, and the user is
 * told that where they type it.
 */
let credential = '';

export function getEngineCredential(): string {
  return credential;
}

export function setEngineCredential(value: string): void {
  credential = value;
}

/** Headers for a diagnosis request. Omits anything that is not set. */
export function engineHeaders(): Record<string, string> {
  const provider = getEngineProvider();
  if (provider === DEFAULT_ENGINE) return {};
  const headers: Record<string, string> = { 'X-Engine-Provider': provider };
  if (credential) headers['X-Engine-Key'] = credential;
  return headers;
}

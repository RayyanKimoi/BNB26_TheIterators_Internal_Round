/**
 * Settings tab: what is configured, and how to change it.
 *
 * Deliberately not an editable form. Every setting here is a server-side
 * secret or a training-time decision: `SLACK_WEBHOOK_URL` and
 * `GEMINI_API_KEY` live in the backend's `.env`, and the held-out class split
 * is baked into the model artifact at train time. A Save button that posted
 * them from the browser would either need the server to accept secrets over
 * an unauthenticated endpoint, or would silently do nothing — and a control
 * that looks live but changes nothing is exactly the kind of fake surface
 * CLAUDE.md rules out.
 *
 * So this shows real state from GET /settings (booleans for secrets, never
 * values), and gives copyable snippets for the things you actually do change.
 */

import { motion, useReducedMotion } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { useEffect, useState } from 'react';

import { API_BASE_URL, api } from '../api/client';
import { ErrorState } from '../components/AppShell';
import type { SettingsResponse } from '../types/api';

function StatusDot({ on, label }: { on: boolean; label: string }) {
  return (
    <span
      className={`data inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 ${
        on ? 'border-pass/40 bg-pass/10 text-pass' : 'border-border text-muted'
      }`}
    >
      <span
        className={`block size-1.5 rounded-full ${on ? 'bg-pass shadow-[0_0_6px_currentColor]' : 'bg-muted'}`}
      />
      {label}
    </span>
  );
}

function CopyBlock({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // Clipboard is unavailable over plain http in some browsers; the text
      // is still selectable, so this is a convenience, not a requirement.
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between">
        <p className="data text-muted">{label}</p>
        <button
          type="button"
          onClick={copy}
          className="data text-muted transition-colors hover:text-accent"
        >
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre className="data hairline mt-1.5 overflow-x-auto bg-bg p-3 leading-relaxed text-text">
        {value}
      </pre>
    </div>
  );
}

function SettingsSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="hairline bg-panel p-4">
          <div className="h-2.5 w-36 animate-pulse rounded bg-border" />
          <div className="mt-4 h-24 animate-pulse rounded bg-border" />
        </div>
      ))}
    </div>
  );
}

export function SettingsView() {
  const reduced = useReducedMotion();
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    api
      .getSettings(controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setSettings(result);
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err : new Error(String(err)));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [reloadKey]);

  if (loading) return <SettingsSkeleton />;
  if (error) {
    return (
      <ErrorState
        error={error}
        onRetry={() => {
          setError(null);
          setLoading(true);
          setReloadKey((k) => k + 1);
        }}
      />
    );
  }
  if (!settings) return null;

  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.08 } },
  };
  const item: Variants = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
  };

  const envSnippet = [
    '# backend/.env',
    'DATABASE_URL=postgresql+psycopg://user:password@host:5432/postgres',
    'GEMINI_API_KEY=your-key-here',
    `GEMINI_MODEL=${settings.gemini_model}`,
    'SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...',
    'DASHBOARD_BASE_URL=http://localhost:5173',
    '# Optional. Without it, the Insights cost card reads "not configured"',
    '# rather than showing a guessed number.',
    'TOKEN_COST_PER_1K_USD=0.002',
  ].join('\n');

  const otelSnippet = `curl -X POST ${API_BASE_URL}/ingest/otel \\
  -H "Content-Type: application/json" \\
  -d '{
    "trace_id": "your-trace-id",
    "task_type": "support_triage",
    "spans": [
      {
        "span_id": "s1",
        "name": "llm.chat.completion",
        "start_time_unix_nano": 0,
        "end_time_unix_nano": 500000000,
        "attributes": { "llm.usage.total_tokens": 42 },
        "status_code": "OK"
      },
      {
        "span_id": "s2",
        "name": "tool.fetch_ticket",
        "start_time_unix_nano": 500000000,
        "end_time_unix_nano": 700000000,
        "attributes": { "tool.name": "fetch_ticket", "output.ticket_id": 4821 },
        "status_code": "ERROR"
      }
    ]
  }'`;

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-4">
      <motion.div variants={item} className="hairline bg-panel p-4">
        <p className="data text-muted">Configuration status</p>
        <p className="data mt-1 text-muted">
          Read from the running backend. Secrets are reported as present or absent and never sent
          to the browser.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <StatusDot on={settings.gemini_configured} label="GEMINI_API_KEY" />
          <StatusDot on={settings.slack_configured} label="SLACK_WEBHOOK_URL" />
          <StatusDot on={settings.token_cost_configured} label="TOKEN_COST_PER_1K_USD" />
          <StatusDot on={settings.model_artifact_present} label="model artifact" />
        </div>
        <dl className="mt-4 grid gap-x-8 gap-y-2 sm:grid-cols-2">
          {[
            ['API base URL', API_BASE_URL],
            ['Dashboard base URL', settings.dashboard_base_url],
            ['Gemini model', settings.gemini_model],
            ['Database', settings.database_dialect],
          ].map(([label, value]) => (
            <div key={label} className="flex items-baseline justify-between gap-4">
              <dt className="data text-muted">{label}</dt>
              <dd className="data truncate text-text">{value}</dd>
            </div>
          ))}
        </dl>
      </motion.div>

      <motion.div variants={item} className="hairline bg-panel p-4">
        <p className="data text-muted">Alerts and keys</p>
        <p className="data mt-1 leading-relaxed text-muted">
          These are server-side secrets, so they are set in the backend&apos;s .env file and the
          server is restarted, not edited from this page. A Slack alert fires at the end of every
          diagnosis that names a class; an unnamed (unknown) diagnosis never pages anyone.
        </p>
        <div className="mt-3">
          <CopyBlock label=".env" value={envSnippet} />
        </div>
      </motion.div>

      <motion.div variants={item} className="hairline bg-panel p-4">
        <p className="data text-muted">OpenTelemetry ingestion</p>
        <p className="data mt-1 leading-relaxed text-muted">
          POST a simplified OTel-shaped trace and it becomes a run with scored steps, exactly like
          a synthetic one. Spans are ordered by start time; `input.` and `output.` prefixed
          attributes become the step payloads, and a span with status_code ERROR sets the step&apos;s
          error flag.
        </p>
        <div className="mt-3">
          <CopyBlock label="curl" value={otelSnippet} />
        </div>
      </motion.div>

      <motion.div variants={item} className="hairline bg-panel p-4">
        <p className="data text-muted">Failure class split</p>
        <p className="data mt-1 leading-relaxed text-muted">
          Fixed at training time and baked into the model artifact. Changing it means editing
          model/dataset.py and rerunning python -m model.train, which is why it is shown here
          rather than toggled: a switch that did not retrain the model would misreport what the
          model actually knows.
        </p>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <p className="data text-muted">Trained on</p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {settings.trained_classes.map((cls) => (
                <span
                  key={cls}
                  className="data rounded border border-border px-1.5 py-0.5 text-text"
                >
                  {cls}
                </span>
              ))}
            </div>
          </div>
          <div>
            <p className="data text-muted">Held out, never seen in training</p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {settings.held_out_classes.map((cls) => (
                <span
                  key={cls}
                  className="data rounded border border-warn/40 bg-warn/10 px-1.5 py-0.5 text-warn"
                >
                  {cls}
                </span>
              ))}
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

/**
 * Diagnosis engine provider: the Settings section, and a compact badge for
 * the Step Inspector drawer.
 *
 * Both read the same store in lib/engineProvider.ts. Nothing here calls the
 * API or touches existing state, so it cannot change how diagnosis behaves.
 *
 * On honesty: only the hybrid engine is wired to a transport today. This panel
 * therefore separates "selected" from "serving" instead of implying the switch
 * is live. SettingsView's own docstring makes the same argument about why it
 * is not an editable form, and a selector that silently changed nothing would
 * be the fake surface that reasoning rules out. The selection is real and it
 * persists; what it does not yet do is reroute traffic, and the panel says so.
 */

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

import {
  ENGINE_OPTIONS,
  getEngineOption,
  useEngineProvider,
  type EngineProviderId,
} from "../lib/engineProvider";

/* -------------------------------------------------------------------------
 * Settings section
 * ---------------------------------------------------------------------- */

export function EngineProviderPanel() {
  const reduced = useReducedMotion();
  const [provider, setProvider] = useEngineProvider();

  // Held in memory only, never written to storage. See lib/engineProvider.ts.
  const [credential, setCredential] = useState("");

  const option = getEngineOption(provider);
  const isDefault = option.routed;

  return (
    <div className="hairline bg-panel p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="card-title">Diagnosis engine provider</h2>
        <span
          className={`data inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 ${
            isDefault
              ? "border-pass/40 bg-pass/10 text-pass"
              : "border-warn/40 bg-warn/10 text-warn"
          }`}
        >
          <span
            className={`block size-1.5 rounded-full ${isDefault ? "bg-pass shadow-[0_0_6px_currentColor]" : "bg-warn"}`}
            aria-hidden="true"
          />
          {isDefault ? "serving diagnoses" : "selected, not routed"}
        </span>
      </div>

      <p className="data mt-1 leading-relaxed text-muted">
        Which engine should score steps. The choice is stored in this browser
        only, under bb_engine_provider.
      </p>

      <div className="mt-4">
        <label htmlFor="engine-provider" className="data block text-muted">
          Engine
        </label>
        <select
          id="engine-provider"
          value={provider}
          onChange={(event) => {
            setProvider(event.target.value as EngineProviderId);
            setCredential("");
          }}
          className="data mt-1.5 w-full rounded border border-border bg-bg px-3 py-2 text-text outline-none"
        >
          {ENGINE_OPTIONS.map((entry) => (
            <option key={entry.id} value={entry.id} className="bg-bg text-text">
              {entry.label}
              {entry.routed ? "" : " (not wired yet)"}
            </option>
          ))}
        </select>
        <p className="data mt-2 leading-relaxed text-muted">{option.detail}</p>
      </div>

      <AnimatePresence initial={false}>
        {option.credential && (
          <motion.div
            key={option.id}
            initial={reduced ? { opacity: 0 } : { opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={reduced ? { opacity: 0 } : { opacity: 0, height: 0 }}
            transition={{ duration: reduced ? 0 : 0.22, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="mt-4">
              <label
                htmlFor="engine-credential"
                className="data block text-muted"
              >
                {option.credential.label}{" "}
                <span className="text-muted">(optional)</span>
              </label>
              <input
                id="engine-credential"
                type={option.credential.kind === "key" ? "password" : "url"}
                value={credential}
                onChange={(event) => setCredential(event.target.value)}
                placeholder={option.credential.placeholder}
                autoComplete="off"
                spellCheck={false}
                className="data mt-1.5 w-full rounded border border-border bg-bg px-3 py-2 text-text outline-none placeholder:text-muted"
              />
              <p className="data mt-2 leading-relaxed text-muted">
                Not saved. This value lives in the page for as long as the tab
                is open and is cleared on reload, because a key kept in browser
                storage is readable by anything that reaches this origin.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!isDefault && (
        <p className="data mt-4 border-l-2 border-warn/50 pl-3 leading-relaxed text-muted">
          Diagnoses are still served by the hybrid engine. Routing to an
          external provider needs a backend transport that does not exist yet,
          so this records the preference rather than switching traffic. The
          Model tab numbers describe the hybrid engine and nothing else.
        </p>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Step Inspector header badge
 * ---------------------------------------------------------------------- */

/**
 * Compact, clickable badge for the Step Inspector drawer header. Names the
 * selected engine and, on click, explains where it is configured.
 *
 * Escape handling is the one subtle part. The drawer already listens for
 * Escape on `window` to close itself, and an open popover should swallow the
 * first Escape rather than dismissing the whole drawer underneath it. This
 * listens in the CAPTURE phase, which runs before the drawer's bubble-phase
 * listener on the same target, and calls stopPropagation so the drawer never
 * sees that keystroke. The drawer's own handler is left untouched.
 */
export function EngineBadge() {
  const reduced = useReducedMotion();
  const [provider] = useEngineProvider();
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const option = getEngineOption(provider);

  useEffect(() => {
    if (!open) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      setOpen(false);
    };
    const onPointerDown = (event: PointerEvent) => {
      if (wrapRef.current?.contains(event.target as Node)) return;
      setOpen(false);
    };

    window.addEventListener("keydown", onKeyDown, true);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open]);

  return (
    <div ref={wrapRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Where the diagnosis engine is configured"
        className={`data inline-flex items-center gap-1.5 rounded border px-2 py-1 transition-colors ${
          option.routed
            ? "border-border text-muted hover:border-accent/60 hover:text-accent"
            : "border-warn/40 bg-warn/10 text-warn hover:border-warn"
        }`}
      >
        <span
          className={`block size-1.5 rounded-full ${option.routed ? "bg-accent" : "bg-warn"}`}
          aria-hidden="true"
        />
        <span className="text-muted">Engine:</span>
        <span className={option.routed ? "text-text" : "text-warn"}>
          {option.short}
        </span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            role="dialog"
            aria-label="Diagnosis engine configuration"
            initial={reduced ? { opacity: 0 } : { opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduced ? { opacity: 0 } : { opacity: 0, y: -4 }}
            transition={{ duration: reduced ? 0 : 0.16, ease: "easeOut" }}
            className="hairline absolute right-0 top-full z-10 mt-2 w-72 bg-bg p-3 shadow-xl"
          >
            <p className="data leading-relaxed text-text">
              Diagnosis engine can be configured per tenant in{" "}
              <span className="text-accent">/settings</span>.
            </p>
            {!option.routed && (
              <p className="data mt-2 border-l-2 border-warn/50 pl-2 leading-relaxed text-muted">
                {option.label} is selected but not wired yet, so this diagnosis
                was scored by the Black Box hybrid engine.
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

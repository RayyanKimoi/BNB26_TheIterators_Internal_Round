# Black Box — Agent Context

Full spec is in PRD.md. Read the relevant section before building. This file is the always-on summary.

## What this is
"Sentry for AI Agents." Finds the exact step that broke an agent run, explains why, lets the user fork from that step with a fix. Core differentiator: a TRAINED classifier that generalizes to held-out failure classes, not an LLM-as-judge.

## Stack
- Backend: FastAPI + SQLModel, Postgres via Supabase
- Model: scikit-learn HistGradientBoosting, joblib. Embeddings: sentence-transformers all-MiniLM-L6-v2 (local)
- LLM layer: Gemini free tier, model `gemini-3.5-flash-lite`, JSON-constrained output
- Frontend: React + Vite + TypeScript, Tailwind + shadcn/ui (heavily restyled), Framer Motion
- Agent: LangGraph + SqliteSaver

## Directory layout
- `generator/` synthetic trace generator, 7 injected fault classes
- `model/` feature extraction, training, evaluation, SHAP. Artifacts in `model/artifacts/`
- `backend/` FastAPI app, SQLModel schema, diagnosis contract, fork and replay
- `ingest/` LangGraph checkpointer adapter, OpenTelemetry span endpoint
- `replay/` checkpointed replay, deterministic suffix re-execution for forks
- `frontend/` React + Vite + TypeScript dashboard
- `.venv/` Python 3.13. Deps in `requirements.txt`, env template in `.env.example`

## Diagnosis contract — every consumer reads this exact shape
Source of truth is `backend/models.py::DiagnosisResponse`. Mirror it, never fork it.
{
  "run_id": "uuid",
  "flagged_step_index": 14,
  "confidence": 0.87,                  // flagged step's share of step_scores, 0-1
  "predicted_class": "stale_retrieval", // one of the 7 classes, or "unknown"
  "evidence_path": "step[14].output.currency",  // best-effort JSON path, real
  "evidence": {"semantic_deviation": 0.71, "duration_z": 2.4, "arg_novelty": 0.0},
  "shap": {"semantic_deviation": 0.31, "arg_novelty": 0.24, "duration_z": 0.12},  // TreeExplainer, real
  "step_scores": [2, 4, 1, 88, 11],    // one per step, 0-100, sums to ~100
  "explanation": "...",                // Gemini, null until /explain is called
  "suggested_fixes": [{"rank": 1, "patch": {}, "rationale": "..."}],  // P1 Gemini
  "class_confidence": 0.92,            // class head probability, 0.0 when unknown
  "unknown_reason": null,              // why no class was named, null when named
  "anomaly_signal": null               // "token_collapse" | "state_repetition" | "{feature}_low" | "{feature}_high" | null
}
`shap` and `evidence` keys are always real feature names from the ten columns.
`anomaly_signal` set means the distribution-relative invariant tier flagged the
step, not the classifier. It names an observation, never a class, so whenever it
is set `predicted_class` is "unknown". The tier is class-agnostic: beyond the two
original named signals (`token_collapse`, `state_repetition`), every other
feature gets a generic two-sided training-percentile check, named
`{feature}_low` / `{feature}_high` (e.g. `semantic_deviation_high`).

## Design tokens
bg #0A0A0B · panel #111113 · border #1F1F23 · text #E8E8E8 · muted #8A8A92 · accent #7DF9C4 · warn #E8B14C · critical #E0574C · pass #5BC98C
Fonts: Space Grotesk (headings/metrics), Geist (body), Geist Mono (all data/JSON/ids)

## Hard rules — never violate
- No purple gradients, no emoji icons, no fake metrics/counters/reviews, no em dashes in UI copy
- Split the dataset BY failure class, never randomly (random split destroys the generalization claim)
- Geist Mono for all data: step ids, JSON, field paths, timestamps
- Report real numbers only. Never invent a metric
- Honour prefers-reduced-motion
- Don't add LangChain or heavy frameworks not called for in PRD.md
# Black Box — PRD

*Sentry for AI Agents*

Oct 3, 2026

## Overview

Black Box finds the exact step that broke an AI agent run, explains why in plain English, and lets a developer fork the run from that step to test a fix. It is a web dashboard for developers, positioned as **Sentry for AI Agents**.

AI agents chain steps together: an LLM decides an action, calls a tool, reads the result, decides again. When a run fails, the cause is usually one bad step buried in the middle, not the last one. Logs show everything that happened but not which step is responsible, so developers read long traces by hand.

**What makes this different from existing tools.** AgentReplay, OrcaReplay and tracefork solve reproducibility through bit-exact replay. Other hackathon projects in this space use an LLM-as-judge reading one run in isolation. Black Box trains a classifier on hundreds of historical runs, learns the statistical signature of each failure class, and proves it generalizes to failure classes held out of training. That is what the problem statement actually asks for, and it is the claim none of the existing tools can make.

## Goals and Non-Goals

**Goals**

- Localize the failing step in a multi-step agent run with measurable accuracy, including on failure classes held out of training.
- Report a structured JSON diagnosis that names the failure class and points at the exact field that went wrong, not just the step number.
- Let a developer fork a run at any step with a fix applied, replaying only the steps after it.
- Prove the model beats the naive baseline of always blaming the last step.
- Ship a deployed, live product with a distinctive terminal-grade UI, not a localhost demo.

**Non-Goals**

- Deep learning or GPU training. Gradient-boosted trees on engineered features train in seconds and are easier to defend to a judge.
- A production-scale ingestion pipeline. Hundreds of runs is enough to train and evaluate; we are not building for 10,000 spans a minute.
- Multi-tenant billing, teams, or role-based access beyond a single login.
- Competing with LangSmith on general-purpose tracing. We ingest traces; we do not try to replace the tracing layer.

## Target User and Positioning

The target user is a developer or team that has built an AI agent (a coding agent, support bot, or automation workflow) and needs to debug why it occasionally fails. This is a developer tool, not a consumer app — a web dashboard the user plugs into their agent's execution logs, the same way they'd plug Sentry into a web app.

Positioning line: **"Sentry for AI Agents"** — instantly legible to any judge or developer who has used an error-tracking tool, and distinct from generic "AI observability" framing because it leads with a familiar, trusted analogy.

## Problem Statement Coverage

Every one of the seven required key features maps to a specific built component. Nothing in this table is optional.

| PS key feature | What we build | Where it lives |
| --- | --- | --- |
| Execution Data | Synthetic trace generator with injected faults, plus a LangGraph checkpointer adapter and an OpenTelemetry ingest endpoint | `generator/`, `ingest/` |
| Failure Diagnosis | HistGradientBoosting classifier scoring every step 0 to 100 for suspicion | `model/` |
| Failure Explanation | Structured JSON diagnosis with field-level evidence, plus a Groq plain-English explanation | `POST /runs/{id}/diagnose` and `/explain` |
| Checkpointed Replay | LangGraph `get_state_history` and `update_state` for real runs; stored step snapshots for synthetic runs | `replay/` |
| Alternative Execution | Fork: deterministic suffix replay with the fix applied, original run immutable | `POST /runs/{id}/fork` |
| Model Evaluation | Held-out failure-class split, per-class accuracy, baseline comparison | Model tab |
| Trace Comparison | Side-by-side original versus fork timeline with changed steps highlighted | Forks tab |

## Failure Taxonomy

Seven named failure classes. The generator injects each one at a known step, so ground truth is exact. Train on five, hold out two for the generalization claim.

| Class | How it is injected | Signature the model can learn |
| --- | --- | --- |
| `wrong_tool_chosen` | Swap the correct tool for a plausible wrong one | Tool-choice entropy spike, semantic drift from task |
| `hallucinated_argument` | Insert a field value absent from prior context | Argument unseen in any upstream step output |
| `stale_retrieval` | Serve a cached result from an earlier, different query | Timestamp gap, low similarity to current query |
| `premature_termination` | End the run before the goal condition is met | Step count below class median, no terminal tool call |
| `infinite_loop` | Repeat a node cycle until budget exhaustion | Repeated state hash, retry count climbing |
| `schema_violation` | Return output that breaks the declared tool schema | Parse-failure flag, type mismatch |
| `context_truncation` | Drop earlier messages from the prompt window | Token count drop, reference to absent content |

**Held-out pair for evaluation:** `infinite_loop` and `context_truncation`. Never present in training. This is what the generalization number is measured on.

**Dataset target:** 200 to 300 runs across 3 task types. Roughly 60 percent failed, 40 percent successful. Successful runs matter: without them the model learns "any anomaly equals failure" rather than what separates a surviving anomaly from a fatal one.

## Model Specification

**Task framing.** Per-step binary classification: is this step the root cause? At inference, score every step in the run and rank them. The top-ranked step is the flag; all scores feed the heatmap.

**Features per step.** Each becomes one column.

| Feature | Type | Why it separates classes |
| --- | --- | --- |
| `duration_z` | float | Latency outlier versus the same action type across the corpus |
| `token_z` | float | Context truncation and loops both distort token counts |
| `retry_count` | int | Loop and transient-failure signal |
| `parse_failure` | bool | Direct schema-violation signal |
| `tool_choice_entropy` | float | Model uncertainty at the decision point |
| `semantic_deviation` | float | Cosine distance between this step's output embedding and the run's task embedding |
| `arg_novelty` | float | Fraction of argument tokens absent from all upstream outputs |
| `state_hash_repeat` | int | Times this state hash was already seen in the run |
| `downstream_error_count` | int | Errors occurring after this step |
| `position_ratio` | float | step\_index divided by total steps, so the model can learn position priors without memorizing them |

Embeddings come from `sentence-transformers` (`all-MiniLM-L6-v2`), local and free. No embedding API dependency.

**Model.** `HistGradientBoostingClassifier` from scikit-learn. Persist with `joblib`. Training takes seconds on this dataset size.

**Split discipline.** Split by failure class, never randomly. A random split leaks held-out classes into training and destroys the generalization claim, which is the single most important number in the pitch.

**Metrics reported.**

- Top-1 localization accuracy on trained classes
- Top-1 localization accuracy on held-out classes, reported separately and prominently
- Top-3 accuracy, since a shortlist of three is still useful to a developer
- Baseline comparison against always blaming the last step and against always blaming the first errored step

**Honesty rule.** Report whatever the real numbers are. A model that scores 70 percent on held-out classes with a clear baseline comparison is far more credible than a suspiciously perfect number, and judges who build models will know the difference.

## Tech Stack and Architecture

The FastAPI backend sits in the middle, tying together the Postgres database, the scikit-learn model, the Groq LLM, and Slack alerts behind one API.

&#91;embedded content: architecture · frontend, backend, model, LLM, alerts\]

| Layer | Tool | Why |
| --- | --- | --- |
| Demo agent | LangGraph + SqliteSaver checkpointer | Gives checkpoint history, `get_state_history`, `update_state` and resume-from-checkpoint for free. Fork is built on this, not hand-rolled |
| Trace generation | Python script with fault injection | Controlled ground truth, which the model needs and a live agent cannot give you |
| Ingestion | FastAPI endpoint accepting OpenTelemetry spans | Ecosystem standard rather than a bespoke format |
| Database | Postgres via Supabase | Relational fit for runs, steps, diagnoses, forks. Hosted free, auth included |
| Backend | FastAPI + SQLModel | Typed, async, auto Swagger docs |
| Embeddings | sentence-transformers, `all-MiniLM-L6-v2` | Local, free, no API dependency in the hot path |
| Model | scikit-learn HistGradientBoosting, joblib | Trains in seconds, defensible, no GPU |
| LLM layer | Groq free tier, JSON-constrained prompts | Near-instant inference matters on stage |
| Frontend | React + Vite + TypeScript | Typed, fast dev loop |
| Styling | Tailwind + shadcn/ui, heavily restyled | Polished base, overridden so it does not look like default shadcn |
| Motion | Framer Motion | Scroll reveals, timeline transitions, fork animations |
| Visual components | Componentry: closing-plasma, cursor-driven-particle-typography, dithered-logo, ascii-effect, pixel-canvas | Landing page only, installed via shadcn CLI |
| Charts | Recharts | Eval dashboard and reliability trends |
| Diff | react-diff-viewer-continued | Fix diffs inside the terminal frame |
| Alerts | Slack or Discord incoming webhook | One HTTP POST, no SDK |
| Hosting | Vercel frontend, Render or Railway backend | Live public URL |

## Feature Set

Three tiers. P0 must be complete and working before any P1 work starts. P2 is only touched if P0 and P1 are both solid.

| # | Feature | Tier | Notes |
| --- | --- | --- | --- |
| 1 | Synthetic trace generator, 7 fault classes | P0 | Everything downstream depends on this. Build first |
| 2 | Feature extraction pipeline | P0 | 10 columns per step as specified above |
| 3 | HistGradientBoosting localization model | P0 | The PS's central requirement |
| 4 | Held-out class evaluation with baselines | P0 | The number the whole pitch rests on |
| 5 | Structured JSON diagnosis API | P0 | One schema consumed by UI, alerts and tests alike |
| 6 | Blame heatmap timeline UI | P0 | Every step scored 0 to 100, colour graded |
| 7 | Fork with deterministic suffix replay | P0 | Outcome flip is the demo's peak moment |
| 8 | Trace comparison, original versus fork | P0 | Required by the PS |
| 9 | Groq plain-English root cause | P1 | Explains why, not just where |
| 10 | Field-level fault localization | P1 | Points at `step[14].output.currency`, not just step 14 |
| 11 | Suggested-fix diff view | P1 | Red and green inside the terminal frame |
| 12 | LangGraph real-agent demo | P1 | Proves it works beyond synthetic data |
| 13 | Slack or Discord structured alert | P1 | 30 minutes of work, large demo payoff |
| 14 | Multiple fork candidates ranked | P2 | Groq proposes 2 to 3 fixes, fork all, show which passes |
| 15 | Diagnosis memory, similar past failures | P2 | Cosine similarity over stored diagnosis vectors |
| 16 | Auto-generated regression test | P2 | Turns a confirmed fix into a permanent check |
| 17 | Reliability dashboard, latency, tokens, cost | P2 | Rounds out the observability story |
| 18 | OpenTelemetry ingest endpoint | P2 | Interoperability claim |

## Data Model

Six tables. The two additions over a naive schema that matter most: `parent_run_id` makes forks a lineage rather than a mutation, and `step_scores` stores every step's suspicion score so the heatmap is reproducible rather than recomputed on each view.

**agent\_runs**

| Field | Type | Description |
| --- | --- | --- |
| id | uuid | Primary key |
| user\_id | uuid | Owner |
| source | enum | `synthetic` / `langgraph` / `otel` |
| task\_type | text | Which demo task this run attempted |
| status | enum | `success` / `failed` |
| parent\_run\_id | uuid, null | Set when this run is a fork |
| forked\_at\_step | int, null | Step index the fork diverged from |
| fix\_applied | jsonb, null | What was changed in the fork |
| injected\_class | text, null | Ground-truth fault class, synthetic only |
| true\_failure\_step | int, null | Ground-truth faulty step index |
| total\_tokens | int | Cost and truncation signal |
| duration\_ms | int | Wall-clock run time |
| created\_at | timestamp | Run start |

**steps**

| Field | Type | Description |
| --- | --- | --- |
| id | uuid | Primary key |
| run\_id | uuid | FK to agent\_runs |
| step\_index | int | Order within the run |
| action\_type | text | `call_llm` / `call_tool` / `decide` |
| tool\_name | text, null | Which tool, when applicable |
| input | jsonb | Step input payload |
| output | jsonb | Step output payload |
| state\_snapshot | jsonb | Agent state after this step, the fork point |
| state\_hash | text | For loop detection |
| duration\_ms | int | Step latency |
| tokens | int | Token count for this step |
| error\_flag | bool | Whether the step errored |

**diagnoses**

| Field | Type | Description |
| --- | --- | --- |
| id | uuid | Primary key |
| run\_id | uuid | FK to agent\_runs |
| flagged\_step\_index | int | Top-ranked step |
| confidence | float | Model probability for the flagged step |
| predicted\_class | text | One of the seven failure classes |
| evidence\_path | text | Exact field, e.g. `step[14].output.currency` |
| evidence | jsonb | Feature values that drove the score |
| feature\_vector | jsonb | Stored for similarity search in diagnosis memory |
| explanation | text, null | Groq plain-English root cause |
| suggested\_fixes | jsonb, null | Ranked candidate fixes |
| created\_at | timestamp | When diagnosed |

**step\_scores**

| Field | Type | Description |
| --- | --- | --- |
| id | uuid | Primary key |
| diagnosis\_id | uuid | FK to diagnoses |
| step\_index | int | Which step this score belongs to |
| suspicion\_score | float | 0 to 100, drives the heatmap colour |

One row per step per diagnosis. This is what makes the heatmap show near-misses rather than a single red flag.

**regression\_tests**

| Field | Type | Description |
| --- | --- | --- |
| id | uuid | Primary key |
| diagnosis\_id | uuid | FK to diagnoses |
| assertion | jsonb | Saved mini-trace and expected outcome |
| created\_at | timestamp | When the test was generated |

## API Design

| Endpoint | Method | Description |
| --- | --- | --- |
| `/runs` | GET | List runs, filterable by status, class, source |
| `/runs/{id}` | GET | Full trace with steps and stored suspicion scores |
| `/runs/{id}/diagnose` | POST | Score every step, persist diagnosis and step\_scores, return the structured JSON |
| `/runs/{id}/explain` | POST | Groq call on the flagged step, returns explanation and ranked suggested fixes |
| `/runs/{id}/fork` | POST | Body: `{from_step, fix}`. Creates a child run, replays the suffix, returns the new run id and outcome |
| `/runs/{id}/compare/{other_id}` | GET | Aligned step-by-step diff between two runs |
| `/runs/{id}/similar` | GET | Past diagnoses with nearby feature vectors |
| `/runs/{id}/regression-test` | POST | Persist the confirmed fix as an assertion |
| `/model/evaluation` | GET | Per-class accuracy, held-out results, baseline comparison |
| `/dashboard/reliability` | GET | Aggregate pass rate, failure mix, latency, tokens, cost |
| `/ingest/otel` | POST | Accept OpenTelemetry spans and map them into the trace schema |

**The diagnosis contract.** Every consumer reads this same object. Define it once as a Pydantic model and reuse it in the UI, the Slack alert and the regression-test generator.

```json
{
  "run_id": "uuid",
  "flagged_step_index": 14,
  "confidence": 0.87,
  "predicted_class": "stale_retrieval",
  "evidence_path": "step[14].output.currency",
  "evidence": {
    "semantic_deviation": 0.71,
    "duration_z": 2.4,
    "arg_novelty": 0.0
  },
  "step_scores": [2, 4, 1, 88, 11],
  "explanation": "Step 14 returned a cached rate quote from an earlier query...",
  "suggested_fixes": [
    {"rank": 1, "patch": {"...": "..."}, "rationale": "..."}
  ]
}
```

## Design System

Minimal, modern, dark, terminal-adjacent. Dot-matrix and ASCII texture carry the character; the data stays legible.

**Typography**

| Role | Font | Usage |
| --- | --- | --- |
| Headings, metrics | Space Grotesk | Section titles, large numbers, confidence scores |
| Body | Geist | Paragraphs, labels, descriptions |
| Data | Geist Mono | Step ids, JSON, field paths, timestamps, terminal output |

**Colour tokens**

| Token | Value | Use |
| --- | --- | --- |
| `--bg` | `#0A0A0B` | Page background |
| `--panel` | `#111113` | Cards, sidebars, inspector |
| `--border` | `#1F1F23` | 1px hairlines, the main source of terminal feel |
| `--text` | `#E8E8E8` | Primary text |
| `--muted` | `#8A8A92` | Secondary text, labels |
| `--accent` | `#7DF9C4` | Interactive affordances only, never decoration |
| `--warn` | `#E8B14C` | Mid suspicion |
| `--critical` | `#E0574C` | Flagged step |
| `--pass` | `#5BC98C` | Passing fork |

No purple gradients. No emoji icons. No fake metrics, counters or testimonials. No em dashes in UI copy. Status colour appears only where it carries meaning.

**Motion, Framer Motion throughout**

- Scroll reveals: opacity 0 to 1 plus 8px upward translate, 240ms, `easeOut`. Never spring, never bounce
- Stagger timeline rows by 20ms on mount, capped so long traces do not crawl
- Heatmap bars fill once on load, then hold. No pulsing or looping on data
- Fork animation: the suffix steps visibly re-run top to bottom, which makes the mechanism legible rather than instant and magical
- Honour `prefers-reduced-motion` everywhere

**Terminal frame component.** One reusable component, used in at least four places: the JSON diagnosis, the fix diff, the landing page how-it-works blocks and the Settings ingestion snippet. Thin border, a label bar carrying context such as `diagnosis.json` or `fork --from 14`, Geist Mono inside, line numbers, syntax highlighting limited to three colours, and a dot-matrix texture at very low opacity on the frame background.

**Componentry usage, landing page only.** `closing-plasma` for hero and footer at low turbulence, `cursor-driven-particle-typography` for the hero headline, `dithered-logo` in the nav, `ascii-effect` for one logo moment, `pixel-canvas` for hero cursor interaction. These stay off the app screens, where they would fight legibility.

## UI/UX Flow

**Landing page**, single dark scroll, five sections.

1. Hero. `closing-plasma` at low opacity and turbulence as atmosphere, headline in particle typography, `dithered-logo` in the nav. Copy is concrete: find the step that broke your agent. No vague hero language.
2. The problem. A wall of raw log lines, deliberately hard to scan, which on scroll transforms into the blame heatmap with one step lit. This transition sells the product without copy.
3. How it works. Three terminal frames showing real commands and real JSON output. Capture, diagnose, fork.
4. The model. Real held-out numbers against the baseline. Never invented figures.
5. Footer CTA with `closing-plasma`, which is what the component was built for.

**App, six tabs in a left sidebar.**

| Tab | Purpose | Key elements |
| --- | --- | --- |
| Runs | Landing after login | Run cards with dot-matrix run id, progress track showing how far the run got, step count, status. Filters by status, class, source |
| Trace | The core screen | Vertical step timeline, every row scored 0 to 100 with a graded bar. Flagged step dominant, near-misses still visible. Click opens the inspector |
| Forks | Lineage and comparison | Tree of the original and every fork, what changed, outcome. Side-by-side compare of any two |
| Model | Satisfies PS Model Evaluation visibly | Per-class accuracy table with held-out classes separated, baseline comparison chart |
| Insights | Aggregate view | Reliability over time, failure mix, latency, token and cost trends, diagnosis memory matches |
| Settings | Configuration | Webhook URL, API key, ingest endpoint, held-out class config |

**The inspector panel**, opened by clicking any step in Trace, stacks three blocks: evidence as structured JSON in a terminal frame with the broken field path highlighted; the Groq explanation in plain English; and the suggested fix as a red and green diff with a Fork from here button.

**Primary flow.** Login, Runs, open a failed run, read the heatmap, click the red step, read evidence and explanation, review the fix diff, Fork from here, watch the suffix replay in the Forks tab, outcome flips to pass, save as regression test. That path exercises all seven PS key features in one continuous motion, which is exactly what the demo should be.

## 32-Hour Build Timeline

Four gated phases. Each gate is a hard stop: if the gate is not met, you do not advance to the next phase's scope, you fix the gate.

&#91;embedded content: 32-hour roadmap · 4 phases, 3 gates\]

| Hours | Phase | Work | Gate |
| --- | --- | --- | --- |
| 0 to 3 | Setup | Repo, Supabase schema, env config, trace schema frozen, LangGraph demo agent skeleton, design tokens in Tailwind | Schema frozen and agreed. Changing it later is the most expensive mistake available |
| 3 to 12 | Core pipeline | Generator with 7 fault classes, 200+ runs, feature extraction, model trained, held-out evaluation with baselines | Model beats the last-step baseline on held-out classes. If it does not, the pitch has no centre |
| 12 to 22 | API and UI | FastAPI endpoints, diagnosis contract, Trace heatmap, inspector, fork with suffix replay, comparison view | Full primary flow works end to end locally |
| 22 to 32 | Novelty, polish, deploy | Groq explanation and fixes, diff view, Slack alert, landing page, deploy, rehearse | Deployed by hour 26. Rehearsed three times by hour 30 |

**Parallelization.** With a team, split along the API contract: one person owns generator plus model, one owns backend plus fork logic, one owns frontend. Freeze the diagnosis JSON contract in hour 3 so the frontend can build against mock data immediately rather than waiting on the model.

**The last two hours are not build time.** They are for rehearsal, the backup video, and fixing whatever the rehearsal exposes.

## Handing This To Claude Code

**Do not paste the whole PRD as a build instruction.** Handing a coding agent 3,000 words and saying build this produces a shallow version of everything and a working version of nothing. It burns context, and by the time it reaches the fork logic it has forgotten the diagnosis contract.

**Do this instead.**

1. **Save the PRD as `PRD.md` in the repo root.** Export this doc to Markdown and commit it. It becomes reference material the agent can read on demand rather than context it must hold.
2. **Create a `CLAUDE.md` in the repo root.** This is the file Claude Code reads automatically on every session. Keep it under roughly 50 lines: stack, directory layout, the diagnosis JSON contract verbatim, design tokens, and the rules that must never be violated (no purple gradients, no invented metrics, split by failure class never randomly, Geist Mono for all data). Everything else lives in `PRD.md`.
3. **Work phase by phase, one prompt per component.** Each prompt names the section of `PRD.md` to read first. For example: *Read the Model Specification section of PRD.md. Implement the feature extraction pipeline in `model/features.py`. Input is a run dict matching the steps schema in the Data Model section. Output is a pandas DataFrame, one row per step, with exactly the ten columns listed. Do not train anything yet.*
4. **Build in dependency order,** which is the order the phases are written in. Generator, then features, then model, then evaluation, then API, then UI. Each layer is testable before the next one exists.
5. **After each component, ask for a test before moving on.** A generator that produces malformed traces discovered at hour 20 is fatal; discovered at hour 5 it costs ten minutes.

**Prompting notes that matter for this project.**

- Paste the diagnosis JSON contract into any prompt that touches it, every time. It is short, and consistency across the UI, alerts and tests depends on it.
- For UI work, give the design tokens and the specific Componentry component name rather than describing a look. Vague aesthetic direction is what produces generic output.
- When asking for the Trace screen, specify that every step renders a score, not just the flagged one. Agents default to highlighting one thing.
- Ask for `prefers-reduced-motion` handling explicitly in the first motion prompt; retrofitting it later is tedious.
- If the agent proposes LangChain or a heavy framework where the PRD does not call for one, decline. Scope creep from a coding agent is still scope creep.

## Demo Script and Success Metrics

**Live demo script (under 4 minutes)**

1. One line of framing: agents fail at one step buried in the middle, and logs do not tell you which.
2. Open the deployed dashboard. Runs list, one failed run.
3. Open it. The heatmap renders, step 14 red, a couple of amber near-misses. Say out loud that every step is scored, not just one flagged.
4. Click step 14. Evidence JSON with the exact broken field path. This is the field-level claim, say it.
5. Explain. Groq's plain-English root cause appears live.
6. Suggest Fix. Red and green diff.
7. Fork from here. The suffix replays visibly. Outcome flips FAIL to SUCCESS, 3 of 12 steps re-executed. This is the peak; pause here.
8. Model tab. Held-out class accuracy against the baseline. This is the slide that separates the project from an LLM wrapper.
9. Slack channel showing the alert that fired with the full diagnosis.

Four minutes, rehearsed three times minimum. The Model tab is non-negotiable even if the clock is tight, because it is the only part no competing project can show.

**Success metrics to report to judges**

- Top-1 localization accuracy on trained classes, and separately on the two held-out classes
- Lift over the last-step baseline, stated as a number
- Seven failure classes covered, with dataset size
- End-to-end latency from run failure to explanation and suggested fix shown
- A live public URL

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Model does not beat the baseline on held-out classes | Medium | Make the two held-out classes structurally distinct (`infinite_loop` has a repeat-hash signature, `context_truncation` a token-drop signature) so generalization is plausible rather than hoped for. Check this at hour 12, not hour 30 |
| Trace schema changes after the UI is built | Medium | Freeze the schema and the diagnosis contract at hour 3. This is the single most expensive thing to change late |
| Scope creep into P2 before P0 is solid | High | The tier gates exist for this. No P1 work until all eight P0 items pass their gate |
| Groq rate limits during rehearsal or demo | Medium | Cache explanations for the demo run. Keep a pre-generated fallback in the database |
| Deploy or network failure on stage | Medium | Deploy by hour 26. Record a full backup video by hour 30. Rehearse once against the live URL on venue wifi |
| LangGraph integration eats time it does not deserve | Medium | It is P1, not P0. Synthetic traces alone satisfy every PS requirement. Cut it without regret if hour 22 arrives and the core is shaky |
| UI looks like default shadcn | Medium | Design tokens applied first, in hour 3, before any component is built. Retrofitting a visual identity never works |
| Team unfamiliar with the stack | Medium | Every layer chosen for shallow learning curves. Assign by comfort, not by interest, for the first twelve hours |

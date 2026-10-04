/**
 * API types, mirrored field for field from `backend/models.py`.
 *
 * That Pydantic module is the single source of truth. When it changes, change
 * this file in the same commit. `scripts/check-contract.mjs` reads the Python
 * source and fails the build if the two drift, so a mismatch is caught here
 * rather than as a runtime `undefined` in the inspector.
 *
 * Nullable Python fields (`str | None`) are modelled as `T | null`, not
 * `T | undefined`: the API sends explicit JSON nulls, and a field that is
 * present-and-null means something different from a field that is absent.
 */

/** The seven failure classes, plus the value used when none can be named. */
export type FailureClass =
  | 'wrong_tool_chosen'
  | 'hallucinated_argument'
  | 'stale_retrieval'
  | 'premature_termination'
  | 'infinite_loop'
  | 'schema_violation'
  | 'context_truncation';

export type PredictedClass = FailureClass | 'unknown';

/**
 * Set when the distribution-relative invariant tier flagged the step instead
 * of the classifier. These name an OBSERVATION, never a failure class.
 * Whenever this is non-null, `predicted_class` is `'unknown'`.
 *
 * `token_collapse` and `state_repetition` are the two original, specially
 * handled signals. Every other feature gets a generic two-sided check named
 * `{feature}_low` / `{feature}_high` (the class-agnostic invariant sweep —
 * see model/predict.py::Localizer._invariant_check), which is why this is a
 * template literal type over `FeatureName` rather than a fixed enum.
 */
export type AnomalySignal = 'token_collapse' | 'state_repetition' | `${FeatureName}_low` | `${FeatureName}_high`;

export type RunStatus = 'success' | 'failed';
export type RunSource = 'synthetic' | 'langgraph' | 'otel';
export type ActionType = 'call_llm' | 'call_tool' | 'decide';

/** The ten feature columns. Keys of `evidence` and `shap`. */
export type FeatureName =
  | 'duration_z'
  | 'token_z'
  | 'retry_count'
  | 'parse_failure'
  | 'tool_choice_entropy'
  | 'semantic_deviation'
  | 'arg_novelty'
  | 'state_hash_repeat'
  | 'downstream_error_count'
  | 'position_ratio';

export interface SuggestedFix {
  rank: number;
  patch: Record<string, unknown>;
  rationale: string;
}

/**
 * The 13-key diagnosis contract. Every consumer reads this exact shape.
 *
 * `evidence_path`, `shap`, `explanation` and `suggested_fixes` are nullable or
 * empty until the P1 layers that populate them have run, so render them
 * conditionally rather than assuming presence.
 */
export interface DiagnosisResponse {
  run_id: string | null;
  flagged_step_index: number | null;
  /** The flagged step's share of `step_scores`, 0 to 1. */
  confidence: number;
  predicted_class: PredictedClass;
  /** e.g. `step[14].output.currency`. Null until field-level localization ships. */
  evidence_path: string | null;
  /** Raw feature values on the flagged step. Partial: NaN columns are omitted as null. */
  evidence: Partial<Record<FeatureName, number | boolean | null>> & Record<string, unknown>;
  /** SHAP attributions. Positive pushed the step toward "root cause". */
  shap: Partial<Record<FeatureName, number>> | null;
  /** One score per step, 0 to 100, summing to about 100. Drives the heatmap. */
  step_scores: number[];
  explanation: string | null;
  suggested_fixes: SuggestedFix[];
  /** Class head probability. 0.0 whenever `predicted_class` is `'unknown'`. */
  class_confidence: number;
  /** Why no class was named. Null when one was. */
  unknown_reason: string | null;
  anomaly_signal: AnomalySignal | null;
}

export interface RunSummary {
  id: string;
  source: RunSource;
  task_type: string;
  status: RunStatus;
  injected_class: FailureClass | null;
  total_tokens: number;
  duration_ms: number;
  /** ISO 8601 string on the wire. */
  created_at: string;
  parent_run_id: string | null;
  forked_at_step: number | null;
  has_diagnosis: boolean;
  /** Number of steps in the run. Counted server side. */
  total_steps: number;
}

export interface StepDetail {
  id: string;
  run_id: string;
  step_index: number;
  action_type: ActionType;
  tool_name: string | null;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  state_snapshot: Record<string, unknown>;
  state_hash: string;
  duration_ms: number;
  tokens: number;
  error_flag: boolean;
}

export interface RunDetail {
  id: string;
  user_id: string | null;
  source: RunSource;
  task_type: string;
  status: RunStatus;
  parent_run_id: string | null;
  forked_at_step: number | null;
  fix_applied: Record<string, unknown> | null;
  /** Ground truth, synthetic runs only. Always null on a fork. */
  injected_class: FailureClass | null;
  true_failure_step: number | null;
  total_tokens: number;
  duration_ms: number;
  created_at: string;
  steps: StepDetail[];
  diagnosis: DiagnosisResponse | null;
}

export interface ExplainRequest {
  step_index: number;
}

/** The explainer's enforced output shape. */
export interface ExplanationPayload {
  root_cause: string;
  evidence_summary: string[];
  proposed_fix: string;
}

export interface ExplanationResponse extends ExplanationPayload {
  run_id: string;
  step_index: number;
  predicted_class: PredictedClass;
  shap: Partial<Record<FeatureName, number>> | null;
  /** True when served from `agent_runs.explanation` with no new LLM call. */
  cached: boolean;
}

export interface ForkRequest {
  from_step: number;
  fix_payload?: Record<string, unknown> | null;
  /** Recorded on the fork for audit. Never executed server side. */
  override_code?: string | null;
}

export interface ForkResponse {
  child_run_id: string;
  parent_run_id: string;
  /** Always `'completed'` once the suffix has been replayed. */
  status: string;
  /** The child's outcome after re-diagnosis. This is the FAIL to SUCCESS flip. */
  outcome: RunStatus;
  parent_outcome: RunStatus;
  forked_at_step: number;
  steps_replayed: number;
  steps_total: number;
  fix_applied: Record<string, unknown> | null;
  steps: StepDetail[];
  diagnosis: DiagnosisResponse | null;
}

/**
 * The evaluation artifact itself (model/artifacts/evaluation.json), nested one
 * level under the `evaluation` key of GET /model/evaluation's response. There
 * is no backend Pydantic model for this: the endpoint serves the JSON file
 * verbatim (`dict[str, Any]`), so `scripts/check-contract.mjs` cannot verify
 * this shape against source the way it does the diagnosis contract. Keep this
 * in sync by hand against `model/artifacts/evaluation.json` if that artifact's
 * shape ever changes.
 */
export interface EvaluationArtifact {
  test_seen_top1: number;
  test_seen_top3: number;
  test_heldout_top1: number;
  test_heldout_top3: number;
  test_seen_runs: number;
  test_heldout_runs: number;
  /** e.g. `{ "last step": { seen: 0.1, heldout: 0.075 } }` */
  baselines: Record<string, Record<string, number>>;
  loco_mean: number | null;
  /** The RAW model gate, before the hybrid fallback. False on this corpus. */
  raw_gate_passed: boolean;
  hybrid_seen_top1: number | null;
  hybrid_heldout_top1: number | null;
  /** Keys are `{class}_seen` or `{class}_heldout`, each a top-1 rate. */
  hybrid_per_class: Record<string, number> | null;
  hybrid_gate_passed: boolean | null;
  /** True once the hybrid engine clears its gate; the headline pass/fail. */
  gate_passed: boolean;
}

/**
 * GET /model/evaluation serves two artifacts verbatim, nested under these two
 * keys. `metrics` is model/artifacts/metrics.json: a flat, open-ended set of
 * numeric readings (per-class top-1, per-class LOCO, baseline comparisons,
 * named-rate and false-positive-rate figures) whose exact keys depend on which
 * failure classes exist in the corpus, so it is typed as a dictionary rather
 * than an exhaustive interface.
 */
export interface EvaluationResponse {
  evaluation: EvaluationArtifact;
  metrics: Record<string, number>;
}

/**
 * Query filters for `GET /runs`. These three are exactly what the endpoint
 * supports today; it does not paginate, so there is deliberately no `limit`.
 */
export interface RunFilters {
  status?: RunStatus;
  source?: RunSource;
  injected_class?: FailureClass;
}

// ---------------------------------------------------------------------------
// Completion pass: comparison, similarity search, regression tests, the
// reliability dashboard, and OTel ingestion.
// ---------------------------------------------------------------------------

/** One aligned position in two runs' step sequences, by step_index. */
export interface StepDiff {
  step_index: number;
  a_present: boolean;
  b_present: boolean;
  a_tool_name: string | null;
  b_tool_name: string | null;
  a_error_flag: boolean | null;
  b_error_flag: boolean | null;
  tool_changed: boolean;
  error_flag_changed: boolean;
  output_changed: boolean;
  changed_output_keys: string[];
}

/** GET /runs/{id}/compare/{other_id} */
export interface CompareResponse {
  run_a_id: string;
  run_b_id: string;
  run_a_status: string;
  run_b_status: string;
  steps_compared: number;
  diffs: StepDiff[];
}

/** One historical match from GET /runs/{id}/similar. */
export interface SimilarRun {
  run_id: string;
  /** Cosine similarity, 1.0 = identical. */
  similarity: number;
  predicted_class: PredictedClass;
  flagged_step_index: number;
  injected_class: FailureClass | null;
}

export interface SimilarRunsResponse {
  run_id: string;
  /** How many other diagnosed runs were searched. */
  compared_against: number;
  matches: SimilarRun[];
}

/** Body for POST /runs/{id}/regression-test. Omit diagnosis_id to assert
 * against the run's current (most recent) diagnosis. */
export interface RegressionTestRequest {
  diagnosis_id?: string | null;
  assertion: Record<string, unknown>;
}

export interface RegressionTestResponse {
  id: string;
  diagnosis_id: string;
  assertion: Record<string, unknown>;
  /** ISO 8601 string on the wire. */
  created_at: string;
}

export interface FailureClassCount {
  injected_class: string;
  count: number;
}

/** One daily bucket of the reliability trend. */
export interface ReliabilityTrendPoint {
  /** ISO date (UTC), e.g. 2026-10-04. */
  date: string;
  total_runs: number;
  success_count: number;
  pass_rate: number;
  total_tokens: number;
  avg_duration_ms: number;
  /** This day's average duration vs. the whole period's mean/std. */
  duration_zscore: number;
}

/**
 * GET /dashboard/reliability. `estimated_cost_usd` is null unless the backend
 * has `TOKEN_COST_PER_1K_USD` configured — there is no default rate, so a
 * null here means "not configured," never a silently-assumed price.
 */
export interface ReliabilityResponse {
  total_runs: number;
  success_count: number;
  failure_count: number;
  overall_pass_rate: number;
  failure_class_breakdown: FailureClassCount[];
  trend: ReliabilityTrendPoint[];
  total_tokens: number;
  estimated_cost_usd: number | null;
  cost_rate_configured: boolean;
}

/**
 * One span in a simplified OTel-like ingestion request. Not full OTLP — see
 * backend/models.py::OtelSpan for why.
 */
export interface OtelSpan {
  span_id: string;
  name: string;
  start_time_unix_nano: number;
  end_time_unix_nano: number;
  attributes: Record<string, unknown>;
  status_code: string;
}

export interface OtelIngestRequest {
  trace_id: string;
  task_type: string;
  spans: OtelSpan[];
}

export interface OtelIngestResponse {
  run_id: string;
  source: string;
  steps_created: number;
  status: string;
}

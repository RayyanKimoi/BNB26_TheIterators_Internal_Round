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
 */
export type AnomalySignal = 'token_collapse' | 'state_repetition';

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

export interface EvaluationResponse {
  test_seen_top1: number;
  test_seen_top3: number;
  test_heldout_top1: number;
  test_heldout_top3: number;
  test_seen_runs: number;
  test_heldout_runs: number;
  /** e.g. `{ "last step": { seen: 0.1, heldout: 0.075 } }` */
  baselines: Record<string, Record<string, number>>;
  loco_mean: number | null;
  /** The RAW model gate. False on the current corpus. Show it alongside the hybrid. */
  gate_passed: boolean;
  hybrid_seen_top1: number | null;
  hybrid_heldout_top1: number | null;
  hybrid_per_class: Record<string, number> | null;
  hybrid_gate_passed: boolean | null;
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

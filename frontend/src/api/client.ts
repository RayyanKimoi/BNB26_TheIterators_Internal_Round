/**
 * Typed API client.
 *
 * Plain `fetch`, no axios: every call here is a JSON request against one
 * origin, which fetch does natively, and the PRD is explicit about not adding
 * dependencies it does not call for.
 *
 * Errors are normalized into `ApiError` so a caller never has to inspect a
 * Response. FastAPI reports failures as `{"detail": ...}`, where detail is a
 * string for our own `HTTPException`s and an array of validation objects for
 * a 422, and both are flattened to a readable message here.
 */

import type {
  CompareResponse,
  DiagnosisResponse,
  EvaluationResponse,
  ExplainRequest,
  ExplanationResponse,
  ForkRequest,
  ForkResponse,
  OtelIngestRequest,
  OtelIngestResponse,
  RegressionTestRequest,
  RegressionTestResponse,
  ReliabilityResponse,
  RunDetail,
  RunFilters,
  RunSummary,
  SimilarRunsResponse,
} from '../types/api';

/** Falls back to the local backend so a fresh clone runs with no .env. */
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';

/** Requests that invoke the model or the LLM are slow; the default is not. */
const DEFAULT_TIMEOUT_MS = 15_000;
const SLOW_TIMEOUT_MS = 60_000;

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }

  /** True when the backend is unreachable rather than returning an error. */
  get isNetworkError(): boolean {
    return this.status === 0;
  }

  /** True when the explainer's upstream LLM failed, typically a rate limit. */
  get isUpstreamFailure(): boolean {
    return this.status === 502;
  }
}

function messageFromDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) =>
        item && typeof item === 'object' && 'msg' in item
          ? String((item as { msg: unknown }).msg)
          : null,
      )
      .filter(Boolean);
    if (parts.length) return parts.join('; ');
  }
  return fallback;
}

interface RequestOptions {
  method?: 'GET' | 'POST';
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, timeoutMs = DEFAULT_TIMEOUT_MS } = options;

  // An inner controller enforces the timeout; the caller's signal still wins.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error; // the caller cancelled; let it propagate
    const timedOut = controller.signal.aborted;
    throw new ApiError(
      timedOut
        ? `Request to ${path} timed out after ${timeoutMs / 1000}s`
        : `Cannot reach the API at ${API_BASE_URL}. Is the backend running?`,
      0,
      error,
    );
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }

  if (!response.ok) {
    let detail: unknown = null;
    try {
      detail = (await response.json())?.detail ?? null;
    } catch {
      detail = await response.text().catch(() => null);
    }
    throw new ApiError(
      messageFromDetail(detail, `${method} ${path} failed with ${response.status}`),
      response.status,
      detail,
    );
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function queryString(filters: RunFilters = {}): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== '') {
      params.append(key, String(value));
    }
  }
  const query = params.toString();
  return query ? `?${query}` : '';
}

export const api = {
  /** GET /runs */
  listRuns(filters: RunFilters = {}, signal?: AbortSignal): Promise<RunSummary[]> {
    return request<RunSummary[]>(`/runs${queryString(filters)}`, { signal });
  },

  /** GET /runs/{id} — the full trace with steps and the latest diagnosis. */
  getRun(runId: string, signal?: AbortSignal): Promise<RunDetail> {
    return request<RunDetail>(`/runs/${encodeURIComponent(runId)}`, { signal });
  },

  /** POST /runs/{id}/diagnose — scores every step. Loads the model on first call. */
  diagnoseRun(runId: string, signal?: AbortSignal): Promise<DiagnosisResponse> {
    return request<DiagnosisResponse>(`/runs/${encodeURIComponent(runId)}/diagnose`, {
      method: 'POST',
      signal,
      timeoutMs: SLOW_TIMEOUT_MS,
    });
  },

  /** POST /runs/{id}/explain — cached server side, so repeat calls are fast. */
  explainRun(
    runId: string,
    body: ExplainRequest,
    signal?: AbortSignal,
  ): Promise<ExplanationResponse> {
    return request<ExplanationResponse>(`/runs/${encodeURIComponent(runId)}/explain`, {
      method: 'POST',
      body,
      signal,
      timeoutMs: SLOW_TIMEOUT_MS,
    });
  },

  /** POST /runs/{id}/fork — replays the suffix and re-diagnoses the child. */
  forkRun(runId: string, body: ForkRequest, signal?: AbortSignal): Promise<ForkResponse> {
    return request<ForkResponse>(`/runs/${encodeURIComponent(runId)}/fork`, {
      method: 'POST',
      body,
      signal,
      timeoutMs: SLOW_TIMEOUT_MS,
    });
  },

  /** GET /model/evaluation — the numbers behind the Model tab. */
  getEvaluation(signal?: AbortSignal): Promise<EvaluationResponse> {
    return request<EvaluationResponse>('/model/evaluation', { signal });
  },

  /** GET /runs/{id}/compare/{other_id} — aligned step-by-step diff. */
  compareRuns(runId: string, otherId: string, signal?: AbortSignal): Promise<CompareResponse> {
    return request<CompareResponse>(
      `/runs/${encodeURIComponent(runId)}/compare/${encodeURIComponent(otherId)}`,
      { signal },
    );
  },

  /** GET /runs/{id}/similar — cosine similarity over stored feature vectors. */
  getSimilarRuns(
    runId: string,
    limit?: number,
    signal?: AbortSignal,
  ): Promise<SimilarRunsResponse> {
    const query = limit ? `?limit=${limit}` : '';
    return request<SimilarRunsResponse>(`/runs/${encodeURIComponent(runId)}/similar${query}`, {
      signal,
    });
  },

  /** POST /runs/{id}/regression-test — persist a confirmed fix assertion. */
  createRegressionTest(
    runId: string,
    body: RegressionTestRequest,
    signal?: AbortSignal,
  ): Promise<RegressionTestResponse> {
    return request<RegressionTestResponse>(
      `/runs/${encodeURIComponent(runId)}/regression-test`,
      { method: 'POST', body, signal },
    );
  },

  /** GET /dashboard/reliability — aggregate pass rate, failure mix, trend. */
  getReliability(signal?: AbortSignal): Promise<ReliabilityResponse> {
    return request<ReliabilityResponse>('/dashboard/reliability', { signal });
  },

  /** POST /ingest/otel — map a simplified OTel-like trace into the schema. */
  ingestOtel(body: OtelIngestRequest, signal?: AbortSignal): Promise<OtelIngestResponse> {
    return request<OtelIngestResponse>('/ingest/otel', { method: 'POST', body, signal });
  },

  /** Cheap liveness probe for the header status dot. */
  async health(signal?: AbortSignal): Promise<boolean> {
    try {
      await request<unknown>('/runs', { signal, timeoutMs: 5_000 });
      return true;
    } catch {
      return false;
    }
  },
};

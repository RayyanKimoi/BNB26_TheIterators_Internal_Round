/**
 * Fail the build if the TypeScript types drift from `backend/models.py`.
 *
 * The contract is shared by the UI, the Slack alert and the regression-test
 * generator, so a silent mismatch surfaces as an `undefined` deep inside the
 * inspector rather than as an error. This compares field names directly
 * against the Python source, with no Python runtime required.
 *
 *   node scripts/check-contract.mjs
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const MODELS_PY = resolve(here, '../../backend/models.py');
const TYPES_TS = resolve(here, '../src/types/api.ts');

/**
 * Pydantic classes to compare, mapped to their TypeScript interface.
 *
 * EvaluationResponse is deliberately absent: GET /model/evaluation serves
 * model/artifacts/evaluation.json and metrics.json verbatim as a raw dict, not
 * through a Pydantic response_model, so there is no source-of-truth class to
 * check `EvaluationArtifact`/`EvaluationResponse` against here. A Pydantic
 * class by that name used to exist and this checker passed against it for
 * months while the live endpoint quietly returned a different shape — exactly
 * the silent drift this file exists to catch, missed because the class was
 * never wired to the route. Keep EvaluationArtifact in sync with the real
 * artifact file by hand instead.
 */
const PAIRS = [
  ['DiagnosisResponse', 'DiagnosisResponse'],
  ['SuggestedFix', 'SuggestedFix'],
  ['RunSummary', 'RunSummary'],
  ['StepDetail', 'StepDetail'],
  ['RunDetail', 'RunDetail'],
  ['ExplainRequest', 'ExplainRequest'],
  ['ExplanationPayload', 'ExplanationPayload'],
  ['ForkRequest', 'ForkRequest'],
  ['ForkResponse', 'ForkResponse'],
  ['StepDiff', 'StepDiff'],
  ['CompareResponse', 'CompareResponse'],
  ['SimilarRun', 'SimilarRun'],
  ['SimilarRunsResponse', 'SimilarRunsResponse'],
  ['RegressionTestRequest', 'RegressionTestRequest'],
  ['RegressionTestResponse', 'RegressionTestResponse'],
  ['FailureClassCount', 'FailureClassCount'],
  ['ReliabilityTrendPoint', 'ReliabilityTrendPoint'],
  ['ReliabilityResponse', 'ReliabilityResponse'],
  ['OtelSpan', 'OtelSpan'],
  ['OtelIngestRequest', 'OtelIngestRequest'],
  ['OtelIngestResponse', 'OtelIngestResponse'],
  ['SettingsResponse', 'SettingsResponse'],
];

/** Interfaces that extend another, whose inherited fields are declared there. */
const INHERITS = { ExplanationResponse: 'ExplanationPayload' };

function pythonFields(source, className) {
  const start = source.indexOf(`class ${className}(`);
  if (start === -1) throw new Error(`class ${className} not found in models.py`);
  const rest = source.slice(start);
  const nextClass = rest.indexOf('\nclass ', 1);
  const body = nextClass === -1 ? rest : rest.slice(0, nextClass);
  const fields = new Set();
  // `name: type = Field(...)` or `name: type` at one indent level.
  for (const match of body.matchAll(/^\s{4}([a-z_][a-z0-9_]*)\s*:/gim)) {
    fields.add(match[1]);
  }
  return fields;
}

function tsFields(source, interfaceName) {
  const pattern = new RegExp(`export interface ${interfaceName}[^{]*\\{`);
  const match = pattern.exec(source);
  if (!match) throw new Error(`interface ${interfaceName} not found in api.ts`);
  let depth = 0;
  let index = match.index + match[0].length - 1;
  const start = index;
  do {
    const char = source[index];
    if (char === '{') depth++;
    else if (char === '}') depth--;
    index++;
  } while (depth > 0 && index < source.length);
  const body = source.slice(start, index);
  const fields = new Set();
  for (const m of body.matchAll(/^\s{2}([a-z_][a-z0-9_]*)\??\s*:/gim)) {
    fields.add(m[1]);
  }
  return fields;
}

const py = readFileSync(MODELS_PY, 'utf8');
const ts = readFileSync(TYPES_TS, 'utf8');

let failures = 0;
for (const [pyName, tsName] of PAIRS) {
  const expected = pythonFields(py, pyName);
  const actual = tsFields(ts, tsName);
  for (const [child, parent] of Object.entries(INHERITS)) {
    if (tsName === parent) continue;
    if (child === tsName) for (const f of tsFields(ts, parent)) actual.add(f);
  }

  const missing = [...expected].filter((f) => !actual.has(f));
  const extra = [...actual].filter((f) => !expected.has(f));

  if (missing.length || extra.length) {
    failures++;
    console.error(`\n  ${pyName} -> ${tsName}`);
    if (missing.length) console.error(`    missing in TypeScript: ${missing.join(', ')}`);
    if (extra.length) console.error(`    not in Pydantic:        ${extra.join(', ')}`);
  } else {
    console.log(`  ok  ${tsName.padEnd(22)} ${expected.size} fields`);
  }
}

// ExplanationResponse inherits its three payload fields, so check it combined.
{
  const expected = pythonFields(py, 'ExplanationResponse');
  const actual = new Set([
    ...tsFields(ts, 'ExplanationResponse'),
    ...tsFields(ts, 'ExplanationPayload'),
  ]);
  const missing = [...expected].filter((f) => !actual.has(f));
  if (missing.length) {
    failures++;
    console.error(`\n  ExplanationResponse missing in TypeScript: ${missing.join(', ')}`);
  } else {
    console.log(`  ok  ${'ExplanationResponse'.padEnd(22)} ${expected.size} fields`);
  }
}

const diagnosisCount = pythonFields(py, 'DiagnosisResponse').size;
if (diagnosisCount !== 13) {
  failures++;
  console.error(`\n  DiagnosisResponse has ${diagnosisCount} keys, expected the documented 13`);
}

if (failures) {
  console.error(`\ncontract check FAILED (${failures} mismatch${failures === 1 ? '' : 'es'})`);
  process.exit(1);
}
console.log(`\ncontract check passed. DiagnosisResponse carries ${diagnosisCount} keys.`);

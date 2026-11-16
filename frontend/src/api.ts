export type ModelId = 'primary' | 'naive_bayes' | 'logistic_regression'

export const MODEL_IDS: ModelId[] = ['primary', 'naive_bayes', 'logistic_regression']

export const MODEL_LABELS: Record<ModelId, string> = {
  primary: 'Primary release',
  naive_bayes: 'Naive Bayes',
  logistic_regression: 'Logistic regression',
}

export type Prediction = {
  request_id: string
  status: 'classified' | 'unable_to_assess'
  label: 'spam' | 'ham' | null
  spam_score: number | null
  threshold: number
  model_version: string
  reason_code: string | null
}

export type ComparisonResult = Prediction & { model_id?: string }

export type Comparison = {
  request_id: string
  status: string
  results: ComparisonResult[]
}

export type ModelInfo = {
  model_id: string | null
  model_version: string | null
  release_id: string | null
  domain: string | null
  language: string | null
  threshold: number | null
  feature_family: string | null
  label_meanings: { label: string; meaning: string }[]
  limitations: string[]
  source_name: string | null
  source_url: string | null
  source_license: string | null
  source_citation: string | null
}

export type Bins = { bin_edges: number[]; counts: number[] }

export type ScoreHistograms = {
  model_id: string | null
  model_version: string | null
  bin_edges: number[]
  all_counts: number[]
  ham_counts: number[]
  spam_counts: number[]
}

export type DatasetSummary = {
  schema_version: string | null
  dataset_id: string | null
  split_id: string | null
  sample_count: number | null
  excluded_count: number | null
  abstained_count: number | null
  source_counts: { key: string; count: number }[]
  class_counts: { key: string; count: number }[]
  length_bins: Bins | null
  score_histograms: ScoreHistograms | null
  artifact_versions: { key: string; value: string }[]
}

export type MetricEntry = { key: string; label: string; definition: string; value: number | null; note: string | null }

export type ConfusionMatrix = { labels: string[]; matrix: number[][] }

export type EvaluationBlock = {
  sample_count: number | null
  total_count: number | null
  classified_count: number | null
  abstained_count: number | null
  coverage: number | null
  support: { key: string; count: number }[]
  abstention_support: { key: string; count: number }[]
  metrics: MetricEntry[]
  confusion_matrix: ConfusionMatrix | null
}

export type Evaluation = {
  model_id: string | null
  model_version: string | null
  partition: string | null
  dataset_sha256: string | null
  split_manifest_sha256: string | null
  decision_threshold: number | null
  positive_label: string | null
  evaluated_at: string | null
  support: { key: string; count: number }[]
  headline: EvaluationBlock
  operational: EvaluationBlock | null
  notes: string[]
}

export type ClusterPoint = { id: string; x: number; y: number; cluster_id: number }

export type ClusterSummary = { cluster_id: number; size: number | null; description: string | null }

export type Clusters = {
  clustering_version: string | null
  projection_version: string | null
  source_id: string | null
  split_id: string | null
  partition: string | null
  sampling: string | null
  sample_count: number | null
  total_count: number | null
  points: ClusterPoint[]
  clusters: ClusterSummary[]
}

export type NumberLookupResult = {
  input: string
  status: 'looked_up' | 'not_a_number' | 'unavailable'
  valid: boolean | null
  e164: string | null
  country_name: string | null
  country_code: string | null
  carrier: string | null
  line_type: string | null
  reason_code: string | null
}

export type NumberLookup = {
  request_id: string
  provider: string | null
  results: NumberLookupResult[]
}

export class ApiError extends Error {
  constructor(message: string, public code: string, public requestId?: string) {
    super(message)
    this.name = 'ApiError'
  }
}

const messages: Record<string, string> = {
  MODEL_UNAVAILABLE: 'The model is not connected yet. Analysis will become available after a validated model is added.',
  COMPARISON_UNAVAILABLE: 'Classifier comparison is not connected yet. It becomes available once both frozen model releases are installed.',
  DATA_UNAVAILABLE: 'Dataset summaries have not been published yet.',
  EVALUATION_UNAVAILABLE: 'Evaluation results have not been published yet.',
  CLUSTERS_UNAVAILABLE: 'The clustering projection has not been published yet.',
  INVALID_TEXT: 'Enter a message between 1 and 2,000 characters.',
  INVALID_REQUEST: 'The request could not be accepted. Check the message and try again.',
  UNSUPPORTED_MEDIA_TYPE: 'The request format was rejected. Reload the page and try again.',
  BODY_TOO_LARGE: 'This message is too large to send. Shorten it and try again.',
  INFERENCE_FAILED: 'The analysis could not be completed. Try again later.',
  NUMBER_LOOKUP_UNAVAILABLE: 'Number lookup is not configured. It becomes available once a validation provider is connected on the server.',
  NUMBER_LOOKUP_FAILED: 'The number lookup could not be completed. Try again later.',
  TOO_MANY_NUMBERS: 'Too many numbers were requested at once. Try a shorter message.',
}

export async function requestJson(path: string, signal: AbortSignal, text?: string): Promise<unknown> {
  const response = await fetch(`/api/v1/${path}`, {
    method: text === undefined ? 'GET' : 'POST',
    ...(text === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) }),
    signal,
    cache: 'no-store',
  })
  let data: unknown
  try { data = await response.json() } catch { throw new ApiError('The server returned an unreadable response.', 'INVALID_RESPONSE') }
  if (!response.ok) {
    const error = (data as { error?: { code?: unknown; request_id?: unknown } })?.error
    const code = typeof error?.code === 'string' ? error.code : 'REQUEST_FAILED'
    throw new ApiError(messages[code] || 'The service could not complete this request. Try again later.', code,
      typeof error?.request_id === 'string' ? error.request_id : undefined)
  }
  return data
}

export function parsePrediction(data: unknown): Prediction {
  if (!data || typeof data !== 'object') throw new ApiError('The analysis response is invalid.', 'INVALID_RESPONSE')
  const result = data as Prediction
  const validNumber = (value: unknown) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
  const common = typeof result.request_id === 'string' && result.request_id.length > 0 &&
    typeof result.model_version === 'string' && result.model_version.length > 0 && validNumber(result.threshold)
  const classified = result.status === 'classified' && validNumber(result.spam_score) &&
    result.label === ((result.spam_score as number) >= result.threshold ? 'spam' : 'ham') && result.reason_code === null
  const abstained = result.status === 'unable_to_assess' && result.label === null && result.spam_score === null &&
    typeof result.reason_code === 'string' && result.reason_code.length > 0
  if (!common || (!classified && !abstained)) throw new ApiError('The analysis response is invalid.', 'INVALID_RESPONSE')
  return result
}

export function parseNumberLookup(data: unknown): NumberLookup {
  const record = data as { request_id?: unknown; provider?: unknown; results?: unknown }
  if (!record || typeof record !== 'object' || !Array.isArray(record.results)) {
    throw new ApiError('The number lookup response is invalid.', 'INVALID_RESPONSE')
  }
  const statuses = ['looked_up', 'not_a_number', 'unavailable']
  return {
    request_id: typeof record.request_id === 'string' ? record.request_id : '',
    provider: typeof record.provider === 'string' ? record.provider : null,
    results: record.results.map(entry => {
      const item = (entry && typeof entry === 'object' ? entry : {}) as Record<string, unknown>
      const status = typeof item.status === 'string' && statuses.includes(item.status)
        ? (item.status as NumberLookupResult['status'])
        : 'unavailable'
      return {
        input: typeof item.input === 'string' ? item.input : '',
        status,
        valid: typeof item.valid === 'boolean' ? item.valid : null,
        e164: typeof item.e164 === 'string' && item.e164 ? item.e164 : null,
        country_name: typeof item.country_name === 'string' && item.country_name ? item.country_name : null,
        country_code: typeof item.country_code === 'string' && item.country_code ? item.country_code : null,
        carrier: typeof item.carrier === 'string' && item.carrier ? item.carrier : null,
        line_type: typeof item.line_type === 'string' && item.line_type ? item.line_type : null,
        reason_code: typeof item.reason_code === 'string' && item.reason_code ? item.reason_code : null,
      }
    }),
  }
}

export async function postJson(path: string, signal: AbortSignal, body: unknown): Promise<unknown> {
  const response = await fetch(`/api/v1/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
    cache: 'no-store',
  })
  let data: unknown
  try { data = await response.json() } catch { throw new ApiError('The server returned an unreadable response.', 'INVALID_RESPONSE') }
  if (!response.ok) {
    const error = (data as { error?: { code?: unknown; request_id?: unknown } })?.error
    const code = typeof error?.code === 'string' ? error.code : 'REQUEST_FAILED'
    throw new ApiError(messages[code] || 'The service could not complete this request. Try again later.', code,
      typeof error?.request_id === 'string' ? error.request_id : undefined)
  }
  return data
}

export function parseComparison(data: unknown): Comparison {
  const record = data as { request_id?: unknown; status?: unknown; results?: unknown }
  if (!record || typeof record !== 'object' || !Array.isArray(record.results) || record.results.length === 0) {
    throw new ApiError('The comparison response is invalid.', 'INVALID_RESPONSE')
  }
  const results = record.results.map(entry => {
    const parsed = parsePrediction({ ...(entry as object), request_id: (entry as { request_id?: string }).request_id ?? 'comparison' })
    return { ...parsed, model_id: typeof (entry as { model_id?: unknown }).model_id === 'string' ? (entry as { model_id: string }).model_id : undefined }
  })
  return {
    request_id: typeof record.request_id === 'string' ? record.request_id : '',
    status: typeof record.status === 'string' ? record.status : 'compared',
    results,
  }
}

const str = (value: unknown): string | null =>
  typeof value === 'string' && value.trim().length > 0 ? value.trim() : null

const num = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null

const count = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null

const strList = (value: unknown): string[] =>
  Array.isArray(value) ? value.map(str).filter((entry): entry is string => entry !== null) : []

function numberList(value: unknown): number[] {
  if (!Array.isArray(value)) return []
  const parsed = value.map(num)
  return parsed.some(entry => entry === null) ? [] : (parsed as number[])
}

function countEntries(value: unknown): { key: string; count: number }[] {
  if (Array.isArray(value)) {
    return value
      .map(entry => {
        const record = entry as { key?: unknown; name?: unknown; label?: unknown; count?: unknown; value?: unknown }
        const key = str(record?.key) ?? str(record?.name) ?? str(record?.label)
        const total = count(record?.count) ?? count(record?.value)
        return key !== null && total !== null ? { key, count: total } : null
      })
      .filter((entry): entry is { key: string; count: number } => entry !== null)
  }
  if (value && typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, raw]) => {
        const total = count(raw)
        return total === null ? null : { key, count: total }
      })
      .filter((entry): entry is { key: string; count: number } => entry !== null)
  }
  return []
}

function parseBins(value: unknown): Bins | null {
  const record = value as { bin_edges?: unknown; counts?: unknown } | null
  if (!record || typeof record !== 'object') return null
  const bin_edges = numberList(record.bin_edges)
  const counts = numberList(record.counts)
  if (bin_edges.length < 2 || counts.length !== bin_edges.length - 1) return null
  return { bin_edges, counts }
}

function parseScoreHistograms(value: unknown): ScoreHistograms | null {
  const record = value as Record<string, unknown> | null
  if (!record || typeof record !== 'object') return null
  const bin_edges = numberList(record.bin_edges)
  if (bin_edges.length < 2) return null
  const size = bin_edges.length - 1
  const all_counts = numberList(record.all_counts)
  const ham_counts = numberList(record.ham_counts)
  const spam_counts = numberList(record.spam_counts)
  if (all_counts.length !== size || ham_counts.length !== size || spam_counts.length !== size) return null
  return {
    model_id: str(record.model_id),
    model_version: str(record.model_version),
    bin_edges,
    all_counts,
    ham_counts,
    spam_counts,
  }
}

export function parseDatasetSummary(data: unknown): DatasetSummary {
  const record = (data && typeof data === 'object' ? data : {}) as Record<string, unknown>
  const schema = record.schema_version
  return {
    schema_version: typeof schema === 'number' && Number.isFinite(schema) ? String(schema) : str(schema),
    dataset_id: str(record.dataset_id),
    split_id: str(record.split_id),
    sample_count: count(record.sample_count),
    excluded_count: count(record.excluded_count),
    abstained_count: count(record.abstained_count),
    source_counts: countEntries(record.source_counts),
    class_counts: countEntries(record.class_counts),
    length_bins: parseBins(record.length_bins),
    score_histograms: parseScoreHistograms(record.score_histograms),
    artifact_versions: Object.entries((record.artifact_versions && typeof record.artifact_versions === 'object'
      ? record.artifact_versions
      : {}) as Record<string, unknown>)
      .map(([key, raw]) => ({ key, value: str(raw) }))
      .filter((entry): entry is { key: string; value: string } => entry.value !== null),
  }
}

export const METRIC_DEFINITIONS: { key: string; label: string; definition: string }[] = [
  { key: 'accuracy', label: 'Accuracy', definition: 'Share of all evaluated messages the model labelled correctly.' },
  { key: 'precision', label: 'Precision', definition: 'Of the messages flagged as spam, the share that really were spam.' },
  { key: 'recall', label: 'Recall', definition: 'Of the real spam messages, the share the model flagged.' },
  { key: 'f1', label: 'F1', definition: 'Harmonic mean of precision and recall, as a single balance figure.' },
  { key: 'false_positive_rate', label: 'False positive rate', definition: 'Of the legitimate messages, the share wrongly flagged as spam.' },
]

function parseConfusion(value: unknown): ConfusionMatrix | null {
  if (Array.isArray(value)) {
    const matrix = value.map(row => numberList(row))
    if (matrix.length > 0 && matrix.every(row => row.length === matrix.length)) {
      return { labels: matrix.map((_, index) => `Class ${index}`), matrix }
    }
    return null
  }
  const record = value as { labels?: unknown; matrix?: unknown } | null
  if (!record || typeof record !== 'object') return null
  const labels = strList(record.labels)
  const matrix = Array.isArray(record.matrix) ? record.matrix.map(row => numberList(row)) : []
  if (matrix.length === 0 || !matrix.every(row => row.length === matrix.length)) return null
  return { labels: labels.length === matrix.length ? labels : matrix.map((_, index) => `Class ${index}`), matrix }
}

function parseBlock(source: Record<string, unknown> | null): EvaluationBlock {
  const record = source ?? {}
  const metricSource = (record.metrics && typeof record.metrics === 'object' ? record.metrics : record) as Record<string, unknown>
  const notes = (record.metric_notes && typeof record.metric_notes === 'object' ? record.metric_notes : {}) as Record<string, unknown>
  return {
    sample_count: count(record.sample_count),
    total_count: count(record.total_count),
    classified_count: count(record.classified_count),
    abstained_count: count(record.abstained_count),
    coverage: num(record.coverage),
    support: countEntries(record.support),
    abstention_support: countEntries(record.abstention_support),
    metrics: METRIC_DEFINITIONS.map(entry => ({
      ...entry,
      value: num(metricSource?.[entry.key]),
      note: str(notes?.[entry.key]),
    })),
    confusion_matrix: parseConfusion(record.confusion_matrix),
  }
}

export function parseEvaluation(data: unknown): Evaluation {
  const record = (data && typeof data === 'object' ? data : {}) as Record<string, unknown>
  const operational = record.operational && typeof record.operational === 'object'
    ? parseBlock(record.operational as Record<string, unknown>)
    : null
  return {
    model_id: str(record.model_id),
    model_version: str(record.model_version),
    partition: str(record.partition) ?? str(record.evaluation_partition) ?? str(record.split),
    dataset_sha256: str(record.dataset_sha256) ?? str(record.dataset_hash),
    split_manifest_sha256: str(record.split_manifest_sha256) ?? str(record.split_hash),
    decision_threshold: num(record.decision_threshold) ?? num(record.threshold),
    positive_label: str(record.positive_label) ?? str(record.positive_class),
    evaluated_at: str(record.evaluated_at) ?? str(record.created_at),
    support: countEntries(record.support ?? record.class_support ?? record.per_class_support),
    headline: parseBlock(record),
    operational,
    notes: strList(record.notes ?? record.limitations),
  }
}

export function parseClusters(data: unknown): Clusters {
  const record = (data && typeof data === 'object' ? data : {}) as Record<string, unknown>
  const points = Array.isArray(record.points)
    ? record.points
        .map(entry => {
          const item = entry as { id?: unknown; sample_id?: unknown; x?: unknown; y?: unknown; cluster_id?: unknown }
          const id = str(item?.id) ?? str(item?.sample_id)
          const x = num(item?.x)
          const y = num(item?.y)
          const cluster_id = num(item?.cluster_id)
          return id !== null && x !== null && y !== null && cluster_id !== null ? { id, x, y, cluster_id } : null
        })
        .filter((entry): entry is ClusterPoint => entry !== null)
    : []
  const clusters = Array.isArray(record.clusters)
    ? record.clusters
        .map(entry => {
          const item = entry as { cluster_id?: unknown; size?: unknown; description?: unknown; summary?: unknown }
          const cluster_id = num(item?.cluster_id)
          return cluster_id === null
            ? null
            : { cluster_id, size: count(item?.size), description: str(item?.description) ?? str(item?.summary) }
        })
        .filter((entry): entry is ClusterSummary => entry !== null)
    : []
  return {
    clustering_version: str(record.clustering_version),
    projection_version: str(record.projection_version),
    source_id: str(record.source_id),
    split_id: str(record.split_id),
    partition: str(record.partition),
    sampling: str(record.sampling),
    sample_count: count(record.sample_count),
    total_count: count(record.total_count),
    points,
    clusters,
  }
}

export function parseModelInfo(data: unknown): ModelInfo {
  const record = (data && typeof data === 'object' ? data : {}) as Record<string, unknown>
  const meanings = record.label_meanings && typeof record.label_meanings === 'object' && !Array.isArray(record.label_meanings)
    ? Object.entries(record.label_meanings as Record<string, unknown>)
        .map(([label, raw]) => ({ label, meaning: str(raw) }))
        .filter((entry): entry is { label: string; meaning: string } => entry.meaning !== null)
    : []
  return {
    model_id: str(record.model_id),
    model_version: str(record.model_version) ?? str(record.version),
    release_id: str(record.release_id),
    domain: str(record.domain),
    language: str(record.language),
    threshold: num(record.threshold),
    feature_family: str(record.feature_family),
    label_meanings: meanings,
    limitations: strList(record.limitations),
    source_name: str(record.source_name),
    source_url: str(record.source_url),
    source_license: str(record.source_license),
    source_citation: str(record.source_citation),
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : 'Unable to reach the API. Check that the backend is running and try again.'
}

export function errorCode(error: unknown): string | undefined {
  return error instanceof ApiError ? error.code : undefined
}

export function errorRequestId(error: unknown): string | undefined {
  return error instanceof ApiError ? error.requestId : undefined
}

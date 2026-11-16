import { useState } from 'react'
import { MODEL_IDS, MODEL_LABELS, parseEvaluation } from '../api'
import type { EvaluationBlock, ModelId } from '../api'
import { useResource } from '../hooks'
import { Awaiting, ConfusionMatrix, Panel } from '../components'

function formatMetric(value: number | null) {
  if (value === null) return '—'
  return value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
}

function Metrics({ block, positive }: { block: EvaluationBlock; positive: string | null }) {
  return (
    <div className="metric-grid">
      {block.metrics.map(metric => (
        <div className="metric" key={metric.key} title={`${metric.definition}${positive && (metric.key === 'precision' || metric.key === 'recall') ? ` Positive class: ${positive}.` : ''}`}>
          <span className="micro">{metric.label}</span>
          <span className={metric.value === null ? 'value absent' : 'value'}>{formatMetric(metric.value)}</span>
          {metric.note && <span className="definition"><b>Not reported:</b> {metric.note}</span>}
        </div>
      ))}
    </div>
  )
}

export default function Evaluation() {
  const [modelId, setModelId] = useState<ModelId>('primary')
  const evaluation = useResource(`evaluation?model_id=${modelId}`, parseEvaluation)
  const data = evaluation.data
  const hasMetrics = data !== null && data.headline.metrics.some(metric => metric.value !== null)

  return (
    <div className="rail">
      <div className="nameplate">
        <span className="eyebrow micro">MODEL EVALUATION</span>
        <h1>Evidence before confidence.</h1>
        <p className="lede">
          Held-out test results for each classifier.
        </p>
      </div>
      <div className="datum" />
      <dl className="plate-meta" aria-label="Evaluation identity">
        <div><dt className="visually-hidden">Model</dt><dd>MODEL <b>{MODEL_LABELS[modelId]}</b></dd></div>
        <div><dt className="visually-hidden">Threshold</dt><dd>THRESHOLD <b>{data?.decision_threshold != null ? data.decision_threshold.toFixed(2) : '—'}</b></dd></div>
        <div><dt className="visually-hidden">Samples</dt><dd>SAMPLES <b>{data?.headline.sample_count?.toLocaleString('en-US') ?? '—'}</b></dd></div>
      </dl>

      <div className="section-stack" style={{ marginTop: 'clamp(18px, 3vh, 26px)' }}>
        <Panel index="01" title="Classifier">
          <div className="panel-body">
            <div className="segmented" role="group" aria-label="Select a classifier">
              {MODEL_IDS.map(value => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={modelId === value}
                  onClick={() => setModelId(value)}
                >
                  {MODEL_LABELS[value]}
                </button>
              ))}
            </div>
          </div>
        </Panel>

        <Panel index="02" title="Headline metrics">
          <div className="panel-body">
            {hasMetrics && data ? (
              <Metrics block={data.headline} positive={data.positive_label} />
            ) : (
              <Awaiting
                title="Waiting for verified evidence"
                message={evaluation.error || 'Requesting the evaluation artifact…'}
                onRetry={evaluation.refresh}
                busy={evaluation.loading}
              />
            )}
          </div>
        </Panel>

        {data?.headline.confusion_matrix && (
          <Panel index="03" title="Confusion matrix">
            <div className="panel-body">
              <ConfusionMatrix
                labels={data.headline.confusion_matrix.labels}
                matrix={data.headline.confusion_matrix.matrix}
                positive={data.positive_label}
              />
            </div>
          </Panel>
        )}

        {data?.operational && (
          <Panel index="04" title="Coverage">
            <div className="panel-body">
              <dl className="kv">
                <dt>Evaluated</dt><dd className="n">{data.operational.total_count?.toLocaleString('en-US') ?? data.headline.sample_count?.toLocaleString('en-US') ?? '—'}</dd>
                <dt>Classified</dt><dd className="n">{data.operational.classified_count?.toLocaleString('en-US') ?? '—'}</dd>
                <dt>Abstained</dt><dd className="n">{data.operational.abstained_count?.toLocaleString('en-US') ?? '—'}</dd>
                <dt>Coverage</dt><dd className="n">{data.operational.coverage != null ? `${(data.operational.coverage * 100).toFixed(2)}%` : '—'}</dd>
              </dl>
            </div>
          </Panel>
        )}

        <Panel index={data ? '05' : '03'} title="Limitations">
          <div className="panel-body">
            <ul className="bullets">
              {(data?.notes.length ? data.notes : [
                'A ham result is not a safety guarantee.',
                'Measured on English SMS only.',
                'Spam wording changes, so performance can drop over time.',
              ]).map(note => <li key={note}>{note}</li>)}
            </ul>
          </div>
        </Panel>
      </div>
    </div>
  )
}

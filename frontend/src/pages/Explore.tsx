import { useEffect, useMemo, useState } from 'react'
import { MODEL_IDS, MODEL_LABELS, parseClusters, parseDatasetSummary, parseModelInfo } from '../api'
import type { ModelId } from '../api'
import { useResource } from '../hooks'
import {
  Awaiting,
  BarChart,
  ChartBlock,
  DataTable,
  Histogram,
  Panel,
  Scatter,
} from '../components'

type Split = 'all' | 'train' | 'validation' | 'test'
type ActualClass = 'all' | 'ham' | 'spam'

const SPLITS: Split[] = ['all', 'train', 'validation', 'test']
const CLASSES: ActualClass[] = ['all', 'ham', 'spam']

type Source = 'all' | 'uci_sms' | 'mendeley_smishing'
const SOURCES: Source[] = ['all', 'uci_sms', 'mendeley_smishing']
const SOURCE_LABELS: Record<string, string> = {
  all: 'All sources',
  uci_sms: 'UCI SMS Spam Collection',
  mendeley_smishing: 'Mendeley SMS Phishing (new messages only)',
}
const sourceLabel = (key: string) => SOURCE_LABELS[key] ?? key

const toneOf = (key: string) => (key.toLowerCase() === 'spam' ? 'flag' : key.toLowerCase() === 'ham' ? 'clear' : 'accent')

export default function Explore() {
  const [source, setSource] = useState<Source>('all')
  const [split, setSplit] = useState<Split>('all')
  const [actualClass, setActualClass] = useState<ActualClass>('all')
  const [modelId, setModelId] = useState<ModelId>('primary')
  const [clusterId, setClusterId] = useState<string>('all')
  const model = useResource(`model?model_id=${modelId}`, parseModelInfo)

  const summaryPath = `dataset-summary?source=${source}&split=${split}&model_id=${modelId}&actual_class=${actualClass}`
  const summary = useResource(summaryPath, parseDatasetSummary)
  const clusterPath = clusterId === 'all' ? 'clusters' : `clusters?cluster_id=${clusterId}`
  const clusters = useResource(clusterPath, parseClusters)

  const clusterIds = useMemo(() => {
    const data = clusters.data
    if (!data) return []
    const ids = new Set<number>()
    data.clusters.forEach(entry => ids.add(entry.cluster_id))
    data.points.forEach(point => ids.add(point.cluster_id))
    return [...ids].sort((a, b) => a - b)
  }, [clusters.data])

  const [allClusterIds, setAllClusterIds] = useState<number[]>([])
  useEffect(() => {
    if (clusterId === 'all' && clusterIds.length > 0) setAllClusterIds(clusterIds)
  }, [clusterId, clusterIds])
  const clusterOptions = allClusterIds.length > 0 ? allClusterIds : clusterIds

  const data = summary.data
  const scores = data?.score_histograms ?? null

  return (
    <div className="rail">
      <div className="nameplate">
        <span className="eyebrow micro">DATA EXPLORER</span>
        <h1>Understand the data.</h1>
        <p className="lede">
          Class balance, message length, model scores and clusters for the combined dataset.
        </p>
      </div>
      <div className="datum" />
      <dl className="plate-meta" aria-label="Artifact identity">
        <div><dt className="visually-hidden">Data</dt><dd>DATA <b>UCI + Mendeley</b></dd></div>
        <div><dt className="visually-hidden">Messages</dt><dd>MESSAGES <b>{data?.sample_count?.toLocaleString('en-US') ?? '—'}</b></dd></div>
      </dl>

      <div className="section-stack" style={{ marginTop: 'clamp(18px, 3vh, 26px)' }}>
        <Panel index="01" title="Filters">
          <div className="panel-body">
            <div className="filter-grid">
              <label>
                <span className="micro">Source</span>
                <select value={source} onChange={event => setSource(event.target.value as Source)}>
                  {SOURCES.map(value => <option key={value} value={value}>{sourceLabel(value)}</option>)}
                </select>
              </label>
              <label>
                <span className="micro">Split</span>
                <select value={split} onChange={event => setSplit(event.target.value as Split)}>
                  {SPLITS.map(value => <option key={value} value={value}>{value === 'all' ? 'All splits' : value}</option>)}
                </select>
              </label>
              <label>
                <span className="micro">Actual class</span>
                <select value={actualClass} onChange={event => setActualClass(event.target.value as ActualClass)}>
                  {CLASSES.map(value => <option key={value} value={value}>{value === 'all' ? 'All classes' : value}</option>)}
                </select>
              </label>
              <label>
                <span className="micro">Model</span>
                <select value={modelId} onChange={event => setModelId(event.target.value as ModelId)}>
                  {MODEL_IDS.map(value => <option key={value} value={value}>{MODEL_LABELS[value]}</option>)}
                </select>
              </label>
              <label>
                <span className="micro">Cluster</span>
                <select value={clusterId} onChange={event => setClusterId(event.target.value)}>
                  <option value="all">All clusters</option>
                  {clusterOptions.map(value => <option key={value} value={value}>Cluster {value}</option>)}
                </select>
              </label>
            </div>
          </div>
        </Panel>

        <Panel index="02" title="Class balance and sources">
          <div className="panel-body">
            {data && (data.class_counts.length > 0 || data.source_counts.length > 0) ? (
              <>
                {data.class_counts.length > 0 && (
                  <ChartBlock
                    title="Messages per class"
                    chart={<BarChart valueLabel="Messages per class" data={data.class_counts.map(entry => ({ key: entry.key, label: entry.key, count: entry.count, tone: toneOf(entry.key) }))} />}
                    table={<DataTable caption="Messages per class" columns={['Class', 'Messages']} rows={data.class_counts.map(entry => [entry.key, entry.count.toLocaleString('en-US')])} />}
                  />
                )}
                {data.source_counts.length > 0 && (
                  <ChartBlock
                    title="Messages per source"
                    note="Messages in both datasets are counted under UCI."
                    chart={<BarChart valueLabel="Messages per source" data={data.source_counts.map(entry => ({ key: entry.key, label: sourceLabel(entry.key), count: entry.count, tone: 'accent' as const }))} />}
                    table={<DataTable caption="Messages per source" columns={['Source', 'Messages']} rows={data.source_counts.map(entry => [sourceLabel(entry.key), entry.count.toLocaleString('en-US')])} />}
                  />
                )}
              </>
            ) : (
              <Awaiting
                title="No dataset summary published"
                message={summary.error || 'Requesting the dataset summary…'}
                onRetry={summary.refresh}
                busy={summary.loading}
              />
            )}
          </div>
        </Panel>

        <Panel index="03" title="Message length">
          <div className="panel-body">
            {data?.length_bins ? (
              <ChartBlock
                title="Message length distribution"
                chart={
                  <Histogram
                    edges={data.length_bins.bin_edges}
                    series={[{ key: 'length', label: 'Messages', counts: data.length_bins.counts, tone: 'accent' }]}
                    formatEdge={value => String(Math.round(value))}
                    xLabel="Characters per message"
                  />
                }
                table={
                  <DataTable
                    caption="Message length distribution"
                    columns={['Bin (characters)', 'Messages']}
                    rows={data.length_bins.counts.map((value, index) => [
                      `${Math.round(data.length_bins!.bin_edges[index])}–${Math.round(data.length_bins!.bin_edges[index + 1])}`,
                      value.toLocaleString('en-US'),
                    ])}
                  />
                }
              />
            ) : (
              <Awaiting
                title="No length distribution published"
                message={summary.error || 'Requesting length bins…'}
                onRetry={summary.refresh}
                busy={summary.loading}
              />
            )}
          </div>
        </Panel>

        <Panel index="04" title="Score distribution against the threshold">
          <div className="panel-body">
            {scores ? (
              <ChartBlock
                title="Model score distribution by actual class"
                note="Ham is hatched, spam is solid. The dashed line is the decision threshold."
                chart={
                  <Histogram
                    edges={scores.bin_edges}
                    series={[
                      { key: 'ham', label: 'Actual ham', counts: scores.ham_counts, tone: 'clear', hatch: true },
                      { key: 'spam', label: 'Actual spam', counts: scores.spam_counts, tone: 'flag' },
                    ]}
                    threshold={model.data?.threshold ?? null}
                    formatEdge={value => value.toFixed(1)}
                    xLabel="Spam model score"
                    height={280}
                  />
                }
                table={
                  <DataTable
                    caption="Score distribution by actual class"
                    columns={['Bin', 'All', 'Ham', 'Spam']}
                    rows={scores.all_counts.map((value, index) => [
                      `${scores.bin_edges[index].toFixed(1)}–${scores.bin_edges[index + 1].toFixed(1)}`,
                      value.toLocaleString('en-US'),
                      scores.ham_counts[index].toLocaleString('en-US'),
                      scores.spam_counts[index].toLocaleString('en-US'),
                    ])}
                  />
                }
              />
            ) : (
              <Awaiting
                title="No score distribution published"
                message={summary.error || 'Requesting score histograms…'}
                onRetry={summary.refresh}
                busy={summary.loading}
              />
            )}
          </div>
        </Panel>

        <Panel index="05" title="Cluster projection">
          <div className="panel-body">
            {clusters.data && clusters.data.points.length > 0 ? (
              <>
                <ChartBlock
                  title="Exploratory clustering"
                  note={`${clusters.data.points.length.toLocaleString('en-US')} test messages, KMeans clusters projected to 2D.`}
                  chart={<Scatter points={clusters.data.points} />}
                  table={
                    <DataTable
                      caption="Cluster summaries"
                      columns={['Cluster', 'Size', 'Description']}
                      rows={clusters.data.clusters.map(entry => [
                        `Cluster ${entry.cluster_id}`,
                        entry.size?.toLocaleString('en-US') ?? '—',
                        entry.description ?? '—',
                      ])}
                    />
                  }
                />
              </>
            ) : (
              <Awaiting
                title="No clustering artifact published"
                message={clusters.error || 'Requesting the clustering projection…'}
                onRetry={clusters.refresh}
                busy={clusters.loading}
              />
            )}
          </div>
        </Panel>
      </div>
    </div>
  )
}

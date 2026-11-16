import { useId, useState } from 'react'
import type { ReactNode } from 'react'


export type Tone = 'flag' | 'clear' | 'abstain' | 'critical' | 'accent' | 'neutral'

const TONE_VAR: Record<Tone, string> = {
  flag: 'var(--flag)',
  clear: 'var(--clear)',
  abstain: 'var(--abstain)',
  critical: 'var(--critical)',
  accent: 'var(--accent)',
  neutral: 'var(--ink-muted)',
}

export const toneStyle = (tone: Tone) => ({ ['--state' as string]: TONE_VAR[tone] })

export function Panel({ index, title, action, children, className }: {
  index?: string
  title: string
  action?: ReactNode
  children: ReactNode
  className?: string
}) {
  const id = useId()
  return (
    <section className={className ? `panel ${className}` : 'panel'} aria-labelledby={id}>
      <div className="panel-head">
        {index && <span className="index-chip" aria-hidden="true">{index}</span>}
        <h2 id={id}>{title}</h2>
        {action}
      </div>
      {children}
    </section>
  )
}

export function StateChip({ tone, glyph, children }: { tone: Tone; glyph: 'up' | 'down' | 'ring' | 'alert'; children: ReactNode }) {
  return (
    <span className="state-chip" style={toneStyle(tone)}>
      <svg className="state-glyph" viewBox="0 0 10 10" aria-hidden="true" focusable="false">
        {glyph === 'up' && <path d="M5 1 9 8H1Z" fill="currentColor" />}
        {glyph === 'down' && <path d="M5 9 1 2h8Z" fill="currentColor" />}
        {glyph === 'ring' && <circle cx="5" cy="5" r="3.4" fill="none" stroke="currentColor" strokeWidth="1.6" />}
        {glyph === 'alert' && <path d="M5 0.8 9.4 9H0.6Zm0 3v2.6M5 7.1v.9" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />}
      </svg>
      {children}
    </span>
  )
}


export function Meter({ score, threshold, tone, busy, label }: {
  score: number | null
  threshold: number | null
  tone: Tone
  busy?: boolean
  label: string
}) {
  const hasReading = score !== null && threshold !== null
  const style = {
    ...toneStyle(tone),
    ['--v' as string]: score === null ? 0 : score,
    ['--t' as string]: threshold === null ? 0.5 : threshold,
  }
  return (
    <div className="meter" style={style}>
      <div
        className="meter-track"
        role="img"
        aria-label={hasReading
          ? `${label}: ${(score * 100).toFixed(1)} percent on a 0 to 100 scale, with the decision threshold at ${(threshold * 100).toFixed(0)} percent.`
          : `${label}: no reading taken. The scale runs from 0 to 100 percent.`}
      >
        {hasReading && <div className="meter-fill" />}
        {hasReading && <div className="meter-needle" />}
        {hasReading && (
          <div className="meter-pin">
            <span className="meter-pin-label">THRESHOLD {threshold.toFixed(2)}</span>
          </div>
        )}
        {busy && <span className="sweep" aria-hidden="true" />}
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span>0.0</span>
        <span>0.5</span>
        <span>1.0</span>
      </div>
    </div>
  )
}


export function Awaiting({ title, message, detail, onRetry, busy }: {
  title: string
  message: string
  detail?: string
  onRetry?: () => void
  busy?: boolean
}) {
  return (
    <div className="awaiting">
      <span className="micro">AWAITING ARTIFACT</span>
      <h3>{title}</h3>
      <p role="status">{message}</p>
      {detail && <p className="field-help">{detail}</p>}
      {onRetry && (
        <button type="button" className="btn quiet" onClick={onRetry} disabled={busy}>
          {busy ? 'Checking…' : 'Check again'}
        </button>
      )}
    </div>
  )
}


export function DataTable({ caption, columns, rows }: {
  caption: string
  columns: string[]
  rows: (string | number)[][]
}) {
  return (
    <div className="data-table-wrap">
      <table className="data">
        <caption>{caption}</caption>
        <thead>
          <tr>{columns.map(column => <th key={column} scope="col">{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {row.map((cell, cellIndex) => (
                cellIndex === 0
                  ? <th key={cellIndex} scope="row">{cell}</th>
                  : <td key={cellIndex} className="n">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ChartBlock({ title, note, chart, table, legend }: {
  title: string
  note?: string
  chart: ReactNode
  table: ReactNode
  legend?: ReactNode
}) {
  const [open, setOpen] = useState(false)
  const id = useId()
  return (
    <div className="chart-block">
      <div className="chart-head">
        <h3>{title}</h3>
        <button
          type="button"
          className="btn quiet"
          aria-expanded={open}
          aria-controls={id}
          onClick={() => setOpen(value => !value)}
        >
          {open ? 'Hide data table' : 'Show data table'}
        </button>
      </div>
      {note && <p className="chart-note">{note}</p>}
      <div className="chart-frame">{chart}</div>
      {legend}
      <div id={id} hidden={!open}>{table}</div>
    </div>
  )
}

export { BarChart, Histogram, Scatter } from './charts'
export type { Series } from './charts'

export function ConfusionMatrix({ labels, matrix, positive }: {
  labels: string[]
  matrix: number[][]
  positive: string | null
}) {
  const total = matrix.flat().reduce((sum, value) => sum + value, 0) || 1
  return (
    <div className="data-table-wrap">
      <table className="matrix">
        <caption className="micro">Rows are the actual class, columns are the predicted class{positive ? `. Positive class: ${positive}` : ''}</caption>
        <thead>
          <tr>
            <td />
            {labels.map(label => <th key={label} scope="col">Predicted {label}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={labels[rowIndex] ?? rowIndex}>
              <th scope="row">Actual {labels[rowIndex] ?? rowIndex}</th>
              {row.map((value, columnIndex) => (
                <td key={columnIndex} style={{ ['--w' as string]: value / total }}>
                  {value.toLocaleString('en-US')}
                  <small>{rowIndex === columnIndex ? 'correct' : 'error'}</small>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

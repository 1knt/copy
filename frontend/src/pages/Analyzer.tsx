import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { Link } from 'react-router-dom'
import {
  ApiError,
  MODEL_LABELS,
  errorMessage,
  parseComparison,
  parseNumberLookup,
  parsePrediction,
  postJson,
  requestJson,
} from '../api'
import type { Comparison, NumberLookup, Prediction } from '../api'
import { extractArtefacts, lookupValue } from '../numbers'
import { REQUEST_TIMEOUT, useDebounced, useStoredFlag } from '../hooks'
import { useModel } from '../model-context'
import {
  Awaiting,
  ChartBlock,
  DataTable,
  Histogram,
  Meter,
  Panel,
  StateChip,
  toneStyle,
} from '../components'
import type { Tone } from '../components'
import {
  clearSession,
  exportSessionCsv,
  exportSessionJson,
  recordReading,
  sessionHistogram,
  useSession,
} from '../session'

const LIVE_DELAY = 700
const MAX = 2000

const EXAMPLES = [
  { label: 'Prize claim', text: 'WINNER!! You have been selected for a free 1000 dollar gift card. Reply CLAIM to 80086 to collect now.' },
  { label: 'Delivery notice', text: 'Your parcel could not be delivered. Confirm your address within 24 hours or it will be returned.' },
  { label: 'Everyday message', text: "Running about 10 minutes late, order me a coffee if you get there first." },
  { label: 'Symbols only', text: '!!! ??? ***' },
]

type Verdict = { tone: Tone; glyph: 'up' | 'down' | 'ring'; chip: string; heading: string }

function verdictOf(result: Prediction): Verdict {
  if (result.status === 'unable_to_assess') {
    return { tone: 'abstain', glyph: 'ring', chip: 'NO READING', heading: 'Unable to assess' }
  }
  return result.label === 'spam'
    ? { tone: 'flag', glyph: 'up', chip: 'AT OR ABOVE THRESHOLD', heading: 'Likely spam' }
    : { tone: 'clear', glyph: 'down', chip: 'BELOW THRESHOLD', heading: 'Likely legitimate' }
}

export default function Analyzer() {
  const [text, setText] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [requestId, setRequestId] = useState<string>()
  const [result, setResult] = useState<Prediction | null>(null)
  const [discarded, setDiscarded] = useState(false)
  const [live, setLive] = useStoredFlag('hn18.live', true)

  const [comparison, setComparison] = useState<Comparison | null>(null)
  const [comparisonError, setComparisonError] = useState('')
  const [comparing, setComparing] = useState(false)

  const [lookup, setLookup] = useState<NumberLookup | null>(null)
  const [lookupError, setLookupError] = useState('')
  const [lookingUp, setLookingUp] = useState(false)

  const sequence = useRef(0)
  const controller = useRef<AbortController | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const lastAnalyzed = useRef<string | null>(null)
  const compareSequence = useRef(0)
  const compareController = useRef<AbortController | null>(null)
  const compareTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const lookupSequence = useRef(0)
  const lookupController = useRef<AbortController | null>(null)
  const lookupTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const session = useSession()
  const model = useModel()
  const count = Array.from(text.trim()).length
  const valid = count >= 1 && count <= MAX
  const settled = useDebounced(text, LIVE_DELAY)

  useEffect(() => () => {
    sequence.current += 1
    controller.current?.abort()
    clearTimeout(timer.current)
    compareSequence.current += 1
    compareController.current?.abort()
    clearTimeout(compareTimer.current)
    lookupSequence.current += 1
    lookupController.current?.abort()
    clearTimeout(lookupTimer.current)
  }, [])

  function edit(value: string) {
    sequence.current += 1
    controller.current?.abort()
    clearTimeout(timer.current)
    lastAnalyzed.current = null
    compareSequence.current += 1
    compareController.current?.abort()
    clearTimeout(compareTimer.current)
    lookupSequence.current += 1
    lookupController.current?.abort()
    clearTimeout(lookupTimer.current)
    setLookup(null)
    setLookupError('')
    setLookingUp(false)
    setDiscarded(result !== null || error !== '')
    setText(value)
    setPending(false)
    setResult(null)
    setError('')
    setRequestId(undefined)
    setComparison(null)
    setComparisonError('')
    setComparing(false)
  }

  const run = useCallback(async (value: string) => {
    const trimmed = value.trim()
    const size = Array.from(trimmed).length
    if (size < 1 || size > MAX) {
      setError('Enter a message between 1 and 2,000 characters.')
      return
    }
    const current = ++sequence.current
    const abort = new AbortController()
    controller.current = abort
    lastAnalyzed.current = value
    setPending(true)
    setResult(null)
    setError('')
    setDiscarded(false)
    setRequestId(undefined)
    timer.current = setTimeout(() => {
      if (current !== sequence.current) return
      sequence.current += 1
      abort.abort()
      setPending(false)
      setError('The request timed out after 10 seconds. Try again when the service is available.')
    }, REQUEST_TIMEOUT)
    try {
      const data = parsePrediction(await requestJson('predictions', abort.signal, trimmed))
      if (current === sequence.current) {
        setResult(data)
        recordReading(data, size)
      }
    } catch (failure) {
      if (current === sequence.current) {
        setError(errorMessage(failure))
        setRequestId(failure instanceof ApiError ? failure.requestId : undefined)
      }
    } finally {
      if (current === sequence.current) {
        clearTimeout(timer.current)
        setPending(false)
      }
    }
  }, [])

  async function analyze(event: FormEvent) {
    event.preventDefault()
    if (pending) return
    await run(text)
  }

  useEffect(() => {
    if (!live || pending) return
    if (settled !== text) return
    const trimmed = settled.trim()
    const size = Array.from(trimmed).length
    if (size < 1 || size > MAX) return
    if (lastAnalyzed.current === settled) return
    void run(settled)
  }, [live, pending, settled, text, run])

  function shortcut(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      if (!pending) void run(text)
    }
  }

  async function compare() {
    if (comparing || !valid) return
    const current = ++compareSequence.current
    const abort = new AbortController()
    compareController.current = abort
    setComparing(true)
    setComparison(null)
    setComparisonError('')
    compareTimer.current = setTimeout(() => {
      if (current !== compareSequence.current) return
      compareSequence.current += 1
      abort.abort()
      setComparing(false)
      setComparisonError('The comparison timed out after 10 seconds. Try again when the service is available.')
    }, REQUEST_TIMEOUT)
    try {
      const data = parseComparison(await requestJson('comparisons', abort.signal, text.trim()))
      if (current === compareSequence.current) setComparison(data)
    } catch (failure) {
      if (current === compareSequence.current) setComparisonError(errorMessage(failure))
    } finally {
      if (current === compareSequence.current) {
        clearTimeout(compareTimer.current)
        setComparing(false)
      }
    }
  }

  const artefacts = extractArtefacts(text)
  const numbers = artefacts.filter(entry => entry.kind !== 'link')

  async function lookUpNumbers() {
    if (lookingUp || numbers.length === 0) return
    const current = ++lookupSequence.current
    const abort = new AbortController()
    lookupController.current = abort
    setLookingUp(true)
    setLookup(null)
    setLookupError('')
    lookupTimer.current = setTimeout(() => {
      if (current !== lookupSequence.current) return
      lookupSequence.current += 1
      abort.abort()
      setLookingUp(false)
      setLookupError('The lookup timed out after 10 seconds. Try again when the service is available.')
    }, REQUEST_TIMEOUT)
    try {
      const data = parseNumberLookup(await postJson('number-lookup', abort.signal, {
        numbers: numbers.slice(0, 5).map(lookupValue),
      }))
      if (current === lookupSequence.current) setLookup(data)
    } catch (failure) {
      if (current === lookupSequence.current) setLookupError(errorMessage(failure))
    } finally {
      if (current === lookupSequence.current) {
        clearTimeout(lookupTimer.current)
        setLookingUp(false)
      }
    }
  }

  const verdict = result ? verdictOf(result) : null

  const announcement = error
    ? ''
    : pending
      ? 'Analyzing the message.'
      : result
        ? result.spam_score === null
          ? 'Result: unable to assess. The model did not recognise enough features, so no score is reported.'
          : `Result: ${result.label === 'spam' ? 'spam' : 'legitimate'}. Spam model score ${(result.spam_score * 100).toFixed(1)} percent against a decision threshold of ${result.threshold.toFixed(2)}.`
        : ''
  const histogram = sessionHistogram(session)
  const scored = session.filter(entry => entry.spam_score !== null)

  return (
    <div className="rail">
      <div className="nameplate">
        <span className="eyebrow micro">MESSAGE ANALYZER</span>
        <h1>SMS Spam Screening</h1>
        <p className="lede">Analyze an English SMS-style message using a trained text classifier. The result is a reading against a decision threshold, not a safety verdict.</p>
      </div>
      <div className="datum" />
      <dl className="plate-meta" aria-label="Instrument calibration">
        <div><dt className="visually-hidden">Model</dt><dd>MODEL <b>{model?.data?.model_version ?? '—'}</b></dd></div>
        <div><dt className="visually-hidden">Threshold</dt><dd>THRESHOLD <b>{model?.data?.threshold != null ? model.data.threshold.toFixed(2) : '—'}</b></dd></div>
        <div><dt className="visually-hidden">Domain</dt><dd>DOMAIN <b>{model?.data?.domain ?? 'English SMS'}</b></dd></div>
        <div><dt className="visually-hidden">Real-time</dt><dd>LIVE <b>{live ? 'on' : 'off'}</b></dd></div>
      </dl>

      <div className="bench" style={{ marginTop: 'clamp(18px, 3vh, 26px)' }}>
        <Panel index="01" title="Check a message">
          <div className="panel-body">
            <form onSubmit={analyze} noValidate>
              <label className="field-label" htmlFor="message">Message text</label>
              <p id="message-help" className="field-help" style={{ margin: '6px 0 12px' }}>
                Paste a message without passwords, account details, or other personal information.
              </p>
              <textarea
                id="message"
                value={text}
                onChange={event => edit(event.target.value)}
                onKeyDown={shortcut}
                placeholder="Enter an English SMS message here…"
                aria-describedby={error ? 'message-help character-count message-error' : 'message-help character-count'}
                aria-invalid={count > MAX || (!!error && count === 0)}
                rows={9}
                spellCheck={false}
              />
              <div className="input-meta" style={{ marginTop: 9 }}>
                <span>English SMS only</span>
                <span
                  className={count > MAX ? 'capacity over' : 'capacity'}
                  style={{ ['--fill' as string]: Math.min(1, count / MAX) }}
                  aria-hidden="true"
                >
                  <i />
                </span>
                <span id="character-count" className={count > MAX ? 'over-limit' : undefined}>
                  {count.toLocaleString('en-US')} / 2,000 characters
                </span>
              </div>

              <div className="actions" style={{ marginTop: 18 }}>
                <button className="btn primary" type="submit" disabled={pending}>
                  {pending ? 'Analyzing…' : 'Analyze message'}
                  <span aria-hidden="true">↗</span>
                </button>
                <button className="btn ghost" type="button" onClick={() => edit('')}>Clear</button>
                <label className="switch" style={{ marginLeft: 'auto' }}>
                  <input type="checkbox" checked={live} onChange={event => setLive(event.target.checked)} />
                  <span className="track" aria-hidden="true" />
                  <span className="switch-text">Real-time</span>
                </label>
              </div>
            </form>

            <div>
              <span className="micro" style={{ display: 'block', marginBottom: 7 }}>
                ILLUSTRATIVE EXAMPLES · WRITTEN FOR THIS PAGE, NOT DATASET RECORDS
              </span>
              <div className="example-row">
                {EXAMPLES.map(example => (
                  <button key={example.label} type="button" onClick={() => edit(example.text)}>
                    {example.label}
                  </button>
                ))}
              </div>
            </div>

            <p className="privacy-note">
              <span aria-hidden="true">◇</span>
              Your message is sent to the configured API when you select Analyze, or automatically after you stop
              typing while Real-time is on. Nothing is stored by this interface, and the session record below keeps
              scores only — never the message text. Press Ctrl or Cmd + Enter to analyze from the keyboard.
            </p>
          </div>
        </Panel>

        <Panel index="02" title="Screening result" className="result">
          <div className="panel-body" aria-busy={pending}>
            <div className="visually-hidden" aria-live="polite" aria-atomic="true">{announcement}</div>
            <div className="readout" style={{ flex: 1 }}>
              {error ? (
                <div className="alert" role="alert" style={toneStyle('critical')}>
                  <StateChip tone="critical" glyph="alert">ANALYSIS UNAVAILABLE</StateChip>
                  <h3>No verdict returned</h3>
                  <p id="message-error">{error}</p>
                  {requestId && <p className="mono-id">Request: {requestId}</p>}
                </div>
              ) : pending ? (
                <>
                  <div className="verdict" style={toneStyle('accent')}>
                    <span className="micro">ACQUIRING</span>
                    <h3>Checking your message</h3>
                  </div>
                  <div className="score-line inert" style={toneStyle('accent')}>
                    <span className="value">—.—</span>
                  </div>
                  <Meter score={null} threshold={null} tone="accent" busy label="Spam model score" />
                  <p className="meter-caption">Waiting for the screening service. Editing or clearing the message cancels this request.</p>
                </>
              ) : result && verdict ? (
                <>
                  <div className="verdict" style={toneStyle(verdict.tone)}>
                    <span className="micro">{result.status === 'classified' ? 'MODEL ASSESSMENT' : 'INSUFFICIENT FEATURES'}</span>
                    <h3>{verdict.heading}</h3>
                    <StateChip tone={verdict.tone} glyph={verdict.glyph}>{verdict.chip}</StateChip>
                  </div>
                  {result.spam_score !== null ? (
                    <>
                      <div className="score-line" style={toneStyle(verdict.tone)}>
                        <span className="value">{(result.spam_score * 100).toFixed(1)}</span>
                        <span className="unit">%</span>
                      </div>
                      <Meter score={result.spam_score} threshold={result.threshold} tone={verdict.tone} label="Spam model score" />
                      <p className="meter-caption">
                        Spam model score · threshold {result.threshold.toFixed(2)}. This score is not a guarantee of safety.
                      </p>
                    </>
                  ) : (
                    <>
                      <div className="score-line inert" style={toneStyle(verdict.tone)}>
                        <span className="value">—.—</span>
                      </div>
                      <Meter score={null} threshold={null} tone={verdict.tone} label="Spam model score" />
                      <p className="meter-caption">
                        The model could not recognize enough features to make a useful assessment. No score is reported,
                        because a zero here would not mean the message is legitimate.
                      </p>
                    </>
                  )}
                  <div className="readout-foot">
                    <span className="mono-id">Model: {result.model_version}</span>
                    <Link to="/evaluation">View evaluation and limitations →</Link>
                  </div>
                </>
              ) : (
                <>
                  <div className="verdict">
                    <span className="micro" aria-hidden="true">{discarded ? 'PREVIOUS READING DISCARDED' : 'INSTRUMENT AT REST'}</span>
                    <h3>Your result will appear here</h3>
                  </div>
                  <div className="score-line inert">
                    <span className="value">—.—</span>
                  </div>
                  <Meter score={null} threshold={null} tone="neutral" label="Spam model score" />
                  <p className="meter-caption">
                    Enter a message and select Analyze to request a real model assessment. The scale above is the
                    instrument at rest: no fill, no needle, and no threshold pin, because no reading has been taken.
                  </p>
                  <div className="readout-foot">
                    <span className="mono-id">No message has been analyzed.</span>
                    <Link to="/evaluation">View evaluation and limitations →</Link>
                  </div>
                </>
              )}
            </div>
          </div>
        </Panel>
      </div>

      <div className="caution" style={{ ...toneStyle('flag'), marginTop: 'clamp(16px, 2.4vw, 22px)' }}>
        <StateChip tone="flag" glyph="alert">SCOPE</StateChip>
        <div>
          <strong>A screening aid, with clear limits.</strong>
          <p>
            A legitimate prediction does not confirm that links or attachments are safe. This project screens English
            SMS text for patterns of spam; it does not scan files for malware and it can be wrong in both directions.
          </p>
        </div>
      </div>

      <div className="section-stack" style={{ marginTop: 'clamp(16px, 2.4vw, 22px)' }}>
        <Panel
          index="03"
          title="Compare the frozen classifiers"
          action={
            <button className="btn quiet" type="button" onClick={compare} disabled={comparing || !valid}>
              {comparing ? 'Comparing…' : 'Run comparison'}
            </button>
          }
        >
          <div className="panel-body">
            <p className="field-help">
              Sends the same captured message to every configured classifier and shows each model's own score against
              its own threshold. A higher score does not mean the better model.
            </p>
            {comparisonError ? (
              <Awaiting title="Comparison unavailable" message={comparisonError} />
            ) : comparison ? (
              <>
                <div className="compare-grid">
                  {comparison.results.map((entry, index) => {
                    const entryVerdict = verdictOf(entry)
                    return (
                      <div className="compare-card" key={entry.model_id ?? index}>
                        <h4>{MODEL_LABELS[(entry.model_id ?? '') as keyof typeof MODEL_LABELS] ?? entry.model_id ?? `Model ${index + 1}`}</h4>
                        <StateChip tone={entryVerdict.tone} glyph={entryVerdict.glyph}>{entryVerdict.heading}</StateChip>
                        {entry.spam_score !== null ? (
                          <>
                            <div className="score-line" style={{ ...toneStyle(entryVerdict.tone), paddingLeft: 10 }}>
                              <span className="value" style={{ fontSize: '1.75rem' }}>{(entry.spam_score * 100).toFixed(1)}</span>
                              <span className="unit" style={{ fontSize: '0.875rem' }}>%</span>
                            </div>
                            <Meter score={entry.spam_score} threshold={entry.threshold} tone={entryVerdict.tone} label={`${entry.model_id ?? 'Model'} score`} />
                          </>
                        ) : (
                          <p className="field-help">This model could not recognize enough features to assess the message.</p>
                        )}
                        <span className="mono-id">{entry.model_version}</span>
                      </div>
                    )
                  })}
                </div>
                <p className="field-help">
                  Both readings refer to the same captured input. Each classifier has its own fitted vocabulary, so one
                  model can abstain while the other returns a score.
                </p>
              </>
            ) : (
              <p className="field-help">
                {valid ? 'Select Run comparison to request one result per configured classifier.' : 'Enter a message first, then run the comparison.'}
              </p>
            )}
          </div>
        </Panel>

        <Panel
          index="04"
          title="Numbers and links in this message"
          action={
            <button className="btn quiet" type="button" onClick={lookUpNumbers} disabled={lookingUp || numbers.length === 0}>
              {lookingUp ? 'Looking up…' : 'Look up numbers'}
            </button>
          }
        >
          <div className="panel-body">
            <p className="field-help">
              Found by reading the message in your browser. Nothing here is a reputation check: the presence of a
              number or a link is not evidence of spam, and no free service publishes a trustworthy spam rating for a
              phone number.
            </p>
            {artefacts.length === 0 ? (
              <p className="field-help">No phone numbers, short codes or web addresses were found in this message.</p>
            ) : (
              <ul className="session-list">
                {artefacts.map(entry => {
                  const found = lookup?.results.find(item => item.input === lookupValue(entry))
                  return (
                    <li className="session-row artefact-row" key={entry.id} style={toneStyle(entry.kind === 'link' ? 'accent' : 'neutral')}>
                      <span className="micro">{entry.kind === 'link' ? 'LINK' : entry.kind === 'shortcode' ? 'CODE' : 'NUMBER'}</span>
                      <span className="label">
                        <b className="num">{entry.raw}</b>
                        <br />
                        {found && found.status === 'looked_up'
                          ? [
                              found.valid === null ? null : found.valid ? 'Valid number' : 'Not a valid number',
                              found.country_name,
                              found.line_type,
                              found.carrier,
                            ].filter(Boolean).join(' · ')
                          : found && found.status === 'not_a_number'
                            ? 'The provider does not recognise this as a dialable number.'
                            : entry.shape}
                      </span>
                    </li>
                  )
                })}
              </ul>
            )}
            {lookupError && (
              <div className="alert" role="alert" style={toneStyle('critical')}>
                <p>{lookupError}</p>
                <p className="field-help">
                  Number lookup is optional and works only when the server has a validation provider key. Without
                  one, <code>POST /api/v1/number-lookup</code> answers that the service is unavailable and no number
                  leaves the server.
                </p>
              </div>
            )}
            {lookup?.provider && <span className="mono-id">Provider: {lookup.provider}</span>}
            <p className="privacy-note">
              <span aria-hidden="true">◇</span>
              Selecting Look up sends only the extracted numbers — never the message text — to the validation provider
              configured on the server. It returns carrier and line-type metadata, not a spam rating. Short codes are
              often not dialable numbers and may come back unrecognised.
            </p>
          </div>
        </Panel>

        <Panel
          index="05"
          title="This session"
          action={
            <div className="actions">
              <button className="btn quiet" type="button" onClick={() => exportSessionJson(session)} disabled={session.length === 0}>Export JSON</button>
              <button className="btn quiet" type="button" onClick={() => exportSessionCsv(session)} disabled={session.length === 0}>Export CSV</button>
              <button className="btn quiet" type="button" onClick={clearSession} disabled={session.length === 0}>Clear session</button>
            </div>
          }
        >
          <div className="panel-body">
            {session.length === 0 ? (
              <p className="field-help">
                Readings you take in this tab are collected here so you can see the distribution of your own scores.
                They are anonymous: only the score, label and model version are kept, never the message text. Clearing
                the session or closing the tab removes them.
              </p>
            ) : (
              <>
                <ChartBlock
                  title="Score distribution for this session"
                  note={`${scored.length.toLocaleString('en-US')} scored reading${scored.length === 1 ? '' : 's'}${histogram.abstained > 0 ? `, plus ${histogram.abstained} with no reading` : ''}. Bins are [left, right) with the final bin including 1.0, and share the analyzer's scale.`}
                  chart={
                    <Histogram
                      edges={histogram.edges}
                      series={[{ key: 'session', label: 'Readings', counts: histogram.counts, tone: 'accent' }]}
                      threshold={scored[0]?.threshold ?? null}
                      formatEdge={value => value.toFixed(1)}
                      xLabel="Model score"
                    />
                  }
                  table={
                    <DataTable
                      caption="Session score distribution"
                      columns={['Bin', 'Readings']}
                      rows={histogram.counts.map((value, index) => [
                        `${histogram.edges[index].toFixed(1)}–${histogram.edges[index + 1].toFixed(1)}`,
                        value,
                      ])}
                    />
                  }
                />
                <ul className="session-list">
                  {session.slice(0, 8).map(entry => {
                    const tone: Tone = entry.status === 'unable_to_assess' ? 'abstain' : entry.label === 'spam' ? 'flag' : 'clear'
                    return (
                      <li className="session-row" key={entry.id} style={{ ...toneStyle(tone), ['--v' as string]: entry.spam_score ?? 0 }}>
                        <span className="bar" aria-hidden="true"><i /></span>
                        <span className="label">
                          <b>{entry.status === 'unable_to_assess' ? 'No reading' : entry.label === 'spam' ? 'Flagged' : 'Not flagged'}</b>
                          {' · '}
                          {entry.characters.toLocaleString('en-US')} characters
                        </span>
                        <span className="score">{entry.spam_score === null ? '—.—' : `${(entry.spam_score * 100).toFixed(1)}%`}</span>
                      </li>
                    )
                  })}
                </ul>
              </>
            )}
          </div>
        </Panel>

        <Panel index="06" title="Model card">
          <div className="panel-body">
            {model?.data ? (
              <>
                <dl className="kv">
                  <dt>Model</dt><dd className="n">{model.data.model_version ?? '—'}</dd>
                  <dt>Identifier</dt><dd className="n">{model.data.model_id ?? '—'}</dd>
                  <dt>Domain</dt><dd>{model.data.domain ?? '—'}</dd>
                  <dt>Language</dt><dd>{model.data.language ?? '—'}</dd>
                  <dt>Threshold</dt><dd className="n">{model.data.threshold != null ? model.data.threshold.toFixed(2) : '—'}</dd>
                  <dt>Features</dt><dd>{model.data.feature_family ?? '—'}</dd>
                </dl>
                {model.data.label_meanings.length > 0 && (
                  <ul className="bullets">
                    {model.data.label_meanings.map(entry => (
                      <li key={entry.label}><b>{entry.label}</b> — {entry.meaning}</li>
                    ))}
                  </ul>
                )}
                {model.data.limitations.length > 0 && (
                  <>
                    <span className="micro">STATED LIMITATIONS</span>
                    <ul className="bullets">
                      {model.data.limitations.map(entry => <li key={entry}>{entry}</li>)}
                    </ul>
                  </>
                )}
                {(model.data.source_name || model.data.source_citation) && (
                  <>
                    <span className="micro">TRAINING DATA ATTRIBUTION</span>
                    <p className="field-help">
                      {model.data.source_url ? (
                        <a href={model.data.source_url} target="_blank" rel="noreferrer noopener">{model.data.source_name ?? model.data.source_url}</a>
                      ) : (
                        model.data.source_name
                      )}
                      {model.data.source_license ? ` · ${model.data.source_license}` : ''}
                    </p>
                    {model.data.source_citation && <p className="field-help">{model.data.source_citation}</p>}
                  </>
                )}
              </>
            ) : (
              <Awaiting
                title="No model metadata published"
                message={model?.error || 'Requesting model metadata…'}
                detail="Version, supported domain, label meanings, threshold and stated limitations appear here once a validated model release is installed."
                onRetry={model?.refresh}
                busy={model?.loading}
              />
            )}
          </div>
        </Panel>
      </div>
    </div>
  )
}

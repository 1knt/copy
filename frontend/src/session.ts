import { useCallback, useEffect, useState } from 'react'
import type { Prediction } from './api'

export type SessionEntry = {
  id: string
  at: number
  status: Prediction['status']
  label: Prediction['label']
  spam_score: number | null
  threshold: number
  model_version: string
  reason_code: string | null
  characters: number
}

const KEY = 'hn18.session'
const LIMIT = 200

let entries: SessionEntry[] = load()
const listeners = new Set<(value: SessionEntry[]) => void>()

function load(): SessionEntry[] {
  try {
    const raw = window.sessionStorage.getItem(KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as SessionEntry[]).slice(0, LIMIT) : []
  } catch {
    return []
  }
}

function persist() {
  try {
    window.sessionStorage.setItem(KEY, JSON.stringify(entries))
  } catch {
  }
}

function publish() {
  persist()
  listeners.forEach(listener => listener(entries))
}

export function recordReading(prediction: Prediction, characters: number) {
  const entry: SessionEntry = {
    id: prediction.request_id,
    at: Date.now(),
    status: prediction.status,
    label: prediction.label,
    spam_score: prediction.spam_score,
    threshold: prediction.threshold,
    model_version: prediction.model_version,
    reason_code: prediction.reason_code,
    characters,
  }
  if (entries.some(existing => existing.id === entry.id)) return
  entries = [entry, ...entries].slice(0, LIMIT)
  publish()
}

export function clearSession() {
  entries = []
  publish()
}

export function useSession(): SessionEntry[] {
  const [value, setValue] = useState(entries)
  useEffect(() => {
    listeners.add(setValue)
    setValue(entries)
    return () => {
      listeners.delete(setValue)
    }
  }, [])
  return value
}

export function useClearSession() {
  return useCallback(() => clearSession(), [])
}

export function sessionHistogram(list: SessionEntry[], binCount = 10) {
  const edges = Array.from({ length: binCount + 1 }, (_, index) => index / binCount)
  const counts = new Array<number>(binCount).fill(0)
  let abstained = 0
  for (const entry of list) {
    if (entry.spam_score === null) {
      abstained += 1
      continue
    }
    const raw = Math.floor(entry.spam_score * binCount)
    counts[Math.min(binCount - 1, Math.max(0, raw))] += 1
  }
  return { edges, counts, abstained }
}

function download(filename: string, mime: string, body: string) {
  const blob = new Blob([body], { type: mime })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

const STAMP_FIELDS = ['id', 'at', 'status', 'label', 'spam_score', 'threshold', 'model_version', 'reason_code', 'characters'] as const

export function exportSessionJson(list: SessionEntry[]) {
  download(
    `spam-screening-session-${new Date().toISOString().slice(0, 10)}.json`,
    'application/json',
    JSON.stringify(
      {
        exported_at: new Date().toISOString(),
        note: 'Anonymous readings from one browser session. Message text was never stored.',
        reading_count: list.length,
        readings: list,
      },
      null,
      2,
    ),
  )
}

export function exportSessionCsv(list: SessionEntry[]) {
  const escape = (value: unknown) => {
    let text = value === null || value === undefined ? '' : String(value)
    if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
  }
  const rows = [
    STAMP_FIELDS.join(','),
    ...list.map(entry => STAMP_FIELDS.map(field => escape(entry[field])).join(',')),
  ]
  download(
    `spam-screening-session-${new Date().toISOString().slice(0, 10)}.csv`,
    'text/csv;charset=utf-8',
    rows.join('\n'),
  )
}

import '@testing-library/jest-dom/vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Analyzer from './Analyzer'


const SPAM = 'WINNER claim your free prize now'
const HAM = 'running ten minutes late, see you soon'

const spam = { request_id: 'r1', status: 'classified', label: 'spam', spam_score: 0.8, threshold: 0.5, model_version: 'test-model', reason_code: null }
const ham = { request_id: 'r2', status: 'classified', label: 'ham', spam_score: 0.2, threshold: 0.5, model_version: 'test-model', reason_code: null }

function bodyOf(call: unknown[]): string {
  return JSON.parse((call[1] as RequestInit).body as string).text as string
}

function predictionFetch() {
  return vi.fn(async (url: string, init: RequestInit) => {
    if (!String(url).includes('predictions')) return { ok: false, json: async () => ({ error: { code: 'MODEL_UNAVAILABLE' } }) } as Response
    const text = JSON.parse(init.body as string).text as string
    return { ok: true, json: async () => (text.includes('WINNER') ? spam : ham) } as Response
  })
}

function mount() {
  render(<MemoryRouter><Analyzer /></MemoryRouter>)
}

function enter(value: string) {
  fireEvent.change(screen.getByLabelText('Message text'), { target: { value } })
}

async function tick(ms: number) {
  await act(async () => {
    vi.advanceTimersByTime(ms)
    for (let index = 0; index < 6; index += 1) await Promise.resolve()
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  window.localStorage.clear()
  window.sessionStorage.clear()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('real-time analysis', () => {
  it('analyzes automatically once typing settles, without pressing Analyze', async () => {
    const fetch = predictionFetch()
    vi.stubGlobal('fetch', fetch)
    mount()
    enter(SPAM)
    await tick(800)
    expect(screen.getByText('Likely spam')).toBeInTheDocument()
    expect(fetch.mock.calls.filter(call => String(call[0]).includes('predictions'))).toHaveLength(1)
  })

  it('never analyzes a message the visitor has already replaced', async () => {
    const fetch = predictionFetch()
    vi.stubGlobal('fetch', fetch)
    mount()

    enter(SPAM)
    await tick(800)
    expect(screen.getByText('Likely spam')).toBeInTheDocument()

    enter(HAM)
    await tick(120)
    fireEvent.click(screen.getByRole('button', { name: /Analyze message/ }))
    await tick(50)
    await tick(900)

    const sent = fetch.mock.calls.filter(call => String(call[0]).includes('predictions')).map(bodyOf)
    expect(sent).toEqual([SPAM, HAM])
    expect(screen.getByText('Likely legitimate')).toBeInTheDocument()
    expect(screen.queryByText('Likely spam')).not.toBeInTheDocument()
  })

  it('re-analyzes the same message after Clear', async () => {
    const fetch = predictionFetch()
    vi.stubGlobal('fetch', fetch)
    mount()

    enter(SPAM)
    await tick(800)
    expect(fetch.mock.calls.filter(call => String(call[0]).includes('predictions'))).toHaveLength(1)

    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await tick(800)
    enter(SPAM)
    await tick(800)

    expect(fetch.mock.calls.filter(call => String(call[0]).includes('predictions'))).toHaveLength(2)
    expect(screen.getByText('Likely spam')).toBeInTheDocument()
  })

  it('does not send anything while the message is blank or oversized', async () => {
    const fetch = predictionFetch()
    vi.stubGlobal('fetch', fetch)
    mount()
    enter('    ')
    await tick(1200)
    enter('a'.repeat(2001))
    await tick(1200)
    expect(fetch.mock.calls.filter(call => String(call[0]).includes('predictions'))).toHaveLength(0)
  })
})

describe('classifier comparison', () => {
  it('discards a comparison for a message that has since been edited', async () => {
    let settle!: (value: Response) => void
    const fetch = vi.fn(async (url: string, init: RequestInit) => {
      if (String(url).includes('comparisons')) return new Promise<Response>(resolve => { settle = resolve })
      const text = JSON.parse(init.body as string).text as string
      return { ok: true, json: async () => (text.includes('WINNER') ? spam : ham) } as Response
    })
    vi.stubGlobal('fetch', fetch)
    mount()

    enter(SPAM)
    await tick(800)
    fireEvent.click(screen.getByRole('button', { name: /Run comparison/ }))
    await tick(50)

    enter(HAM)
    await tick(50)
    await act(async () => {
      settle({ ok: true, json: async () => ({ request_id: 'c1', status: 'compared', results: [{ ...spam, model_id: 'naive_bayes' }] }) } as Response)
      for (let index = 0; index < 6; index += 1) await Promise.resolve()
    })

    expect(screen.queryByText('Naive Bayes')).not.toBeInTheDocument()
  })

  it('recovers the comparison button when the endpoint never answers', async () => {
    const fetch = vi.fn(async (url: string, init: RequestInit) => {
      if (String(url).includes('comparisons')) return new Promise<Response>(() => {})
      const text = JSON.parse(init.body as string).text as string
      return { ok: true, json: async () => (text.includes('WINNER') ? spam : ham) } as Response
    })
    vi.stubGlobal('fetch', fetch)
    mount()

    enter(SPAM)
    await tick(800)
    fireEvent.click(screen.getByRole('button', { name: /Run comparison/ }))
    await tick(50)
    expect(screen.getByRole('button', { name: /Comparing/ })).toBeDisabled()

    await tick(10000)
    expect(screen.getByRole('button', { name: /Run comparison/ })).toBeEnabled()
    expect(screen.getByText(/comparison timed out/i)).toBeInTheDocument()
  })
})

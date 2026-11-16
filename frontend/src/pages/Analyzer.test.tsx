import '@testing-library/jest-dom/vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Analyzer from './Analyzer'

const prediction = { request_id: 'test-id', status: 'classified', label: 'spam', spam_score: 0.8, threshold: 0.5, model_version: 'test-model', reason_code: null }
function mount() { render(<MemoryRouter><Analyzer /></MemoryRouter>); return screen.getByLabelText('Message text') }
function response(data: unknown, ok = true) { return { ok, json: async () => data } as Response }
function enter(value: string) { fireEvent.change(screen.getByLabelText('Message text'), { target: { value } }) }
function submit() { fireEvent.click(screen.getByRole('button', { name: /Analyze message/ })) }
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers() })

describe('Analyzer', () => {
  it('rejects blank and oversized input before any request', () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch); mount()
    enter('   '); submit()
    expect(screen.getByRole('alert')).toHaveTextContent('between 1 and 2,000')
    enter('a'.repeat(2001)); submit()
    expect(screen.getByRole('alert')).toHaveTextContent('between 1 and 2,000')
    expect(fetch).not.toHaveBeenCalled()
  })
  it('counts Unicode code points and sends exactly the trimmed message', async () => {
    const fetch = vi.fn().mockResolvedValue(response(prediction)); vi.stubGlobal('fetch', fetch); mount()
    const text = '😀'.repeat(2000)
    enter(`  ${text}  `)
    expect(screen.getByText('2,000 / 2,000 characters')).toBeInTheDocument()
    submit()
    await screen.findByText('Likely spam')
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(fetch.mock.calls[0][1].body).toBe(JSON.stringify({ text }))
    expect(fetch.mock.calls[0][0]).toBe('/api/v1/predictions')
  })
  it('shows model unavailability without fabricating a verdict', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({ error: { code: 'MODEL_UNAVAILABLE', message: 'internal details', request_id: 'req-1' } }, false)))
    mount(); enter('A test message'); submit()
    expect(await screen.findByRole('alert')).toHaveTextContent('model is not connected yet')
    expect(screen.queryByText('Likely legitimate')).not.toBeInTheDocument()
    expect(screen.queryByText('internal details')).not.toBeInTheDocument()
  })
  it('invalidates an old response when input is edited, even if abort is ignored', async () => {
    let resolve!: (value: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(done => { resolve = done })))
    mount(); enter('First message'); submit(); enter('New message')
    await act(async () => resolve(response(prediction)))
    expect(screen.queryByText('Likely spam')).not.toBeInTheDocument()
    expect(screen.getByText('Your result will appear here')).toBeInTheDocument()
  })
  it('clears pending input and ignores its eventual result', async () => {
    let resolve!: (value: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(done => { resolve = done })))
    const field = mount(); enter('First message'); submit()
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await act(async () => resolve(response(prediction)))
    expect(field).toHaveValue('')
    expect(screen.queryByText('Likely spam')).not.toBeInTheDocument()
  })
  it('times out after ten seconds even if the transport ignores cancellation', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    mount(); enter('Test'); submit()
    await act(async () => vi.advanceTimersByTime(10000))
    expect(screen.getByRole('alert')).toHaveTextContent('timed out after 10 seconds')
    expect(screen.getByRole('button', { name: /Analyze message/ })).toBeEnabled()
  })
  it('rejects a response with an inconsistent score and label', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({ ...prediction, label: 'ham' })))
    mount(); enter('Test'); submit()
    expect(await screen.findByRole('alert')).toHaveTextContent('response is invalid')
    expect(screen.queryByText('Likely legitimate')).not.toBeInTheDocument()
  })
  it('displays abstention without a score or classified verdict', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({ ...prediction, status: 'unable_to_assess', label: null, spam_score: null, reason_code: 'NO_RECOGNIZED_FEATURES' })))
    mount(); enter('!!!'); submit()
    await screen.findByText('Unable to assess')
    expect(screen.queryByText('Likely spam')).not.toBeInTheDocument()
    expect(screen.queryByText(/Spam model score/)).not.toBeInTheDocument()
  })
  it('prevents duplicate submissions while loading and reports network failure', async () => {
    let reject!: (reason: Error) => void
    const fetch = vi.fn(() => new Promise<Response>((_, fail) => { reject = fail }))
    vi.stubGlobal('fetch', fetch); mount(); enter('Test'); submit()
    expect(screen.getByRole('button', { name: /Analyzing/ })).toBeDisabled()
    expect(fetch).toHaveBeenCalledTimes(1)
    await act(async () => reject(new Error('Disconnected')))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Unable to reach the API'))
  })
})

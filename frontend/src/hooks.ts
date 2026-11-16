import { useCallback, useEffect, useRef, useState } from 'react'
import { errorCode, errorMessage, requestJson } from './api'

export const REQUEST_TIMEOUT = 10000

export type Resource<T> = {
  data: T | null
  error: string
  code: string | undefined
  loading: boolean
  refresh: () => void
}

export function useResource<T>(
  path: string | null,
  parse: (value: unknown) => T,
  options: { pollMs?: number } = {},
): Resource<T> {
  const { pollMs } = options
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState('')
  const [code, setCode] = useState<string | undefined>(undefined)
  const [loading, setLoading] = useState(path !== null)
  const [tick, setTick] = useState(0)
  const parser = useRef(parse)
  parser.current = parse

  const refresh = useCallback(() => setTick(value => value + 1), [])

  useEffect(() => {
    if (path === null) {
      setData(null)
      setError('')
      setCode(undefined)
      setLoading(false)
      return
    }
    const abort = new AbortController()
    let active = true
    setLoading(true)
    const timer = setTimeout(() => {
      if (!active) return
      abort.abort()
    }, REQUEST_TIMEOUT)

    requestJson(path, abort.signal)
      .then(value => {
        if (!active) return
        setData(parser.current(value))
        setError('')
        setCode(undefined)
      })
      .catch(failure => {
        if (!active) return
        setData(null)
        setError(errorMessage(failure))
        setCode(errorCode(failure))
      })
      .finally(() => {
        clearTimeout(timer)
        if (active) setLoading(false)
      })

    return () => {
      active = false
      abort.abort()
      clearTimeout(timer)
    }
  }, [path, tick])

  useEffect(() => {
    if (!pollMs || path === null) return
    const id = setInterval(refresh, pollMs)
    return () => clearInterval(id)
  }, [pollMs, path, refresh])

  return { data, error, code, loading, refresh }
}

function readStored(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function writeStored(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value)
  } catch {
  }
}

export function useStoredFlag(key: string, fallback: boolean): [boolean, (value: boolean) => void] {
  const [value, setValue] = useState(() => {
    const stored = readStored(key)
    return stored === null ? fallback : stored === 'true'
  })
  const update = useCallback((next: boolean) => {
    setValue(next)
    writeStored(key, String(next))
  }, [key])
  return [value, update]
}

export type Theme = 'system' | 'light' | 'dark'

export function useTheme(): [Theme, (value: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = readStored('hn18.theme')
    return stored === 'light' || stored === 'dark' ? stored : 'system'
  })
  const apply = useCallback((next: Theme) => {
    setTheme(next)
    if (next === 'system') {
      delete document.documentElement.dataset.theme
      try {
        window.localStorage.removeItem('hn18.theme')
      } catch {
      }
    } else {
      document.documentElement.dataset.theme = next
      writeStored('hn18.theme', next)
    }
  }, [])
  return [theme, apply]
}

export function useDebounced(value: string, delay: number): string {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setSettled(value), delay)
    return () => clearTimeout(id)
  }, [value, delay])
  return settled
}

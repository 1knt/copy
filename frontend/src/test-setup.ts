import { vi } from 'vitest'

vi.mock('plotly.js-basic-dist-min', () => ({
  default: {
    react: vi.fn(async () => undefined),
    purge: vi.fn(),
  },
}))

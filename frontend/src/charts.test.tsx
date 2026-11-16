import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import Plotly from 'plotly.js-basic-dist-min'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { BarChart, Histogram, Scatter, binPosition } from './charts'

type Call = [HTMLElement, Record<string, unknown>[], Record<string, any>, Record<string, any>]
const react = (Plotly as unknown as { react: ReturnType<typeof vi.fn> }).react
const lastCall = () => react.mock.calls.at(-1) as unknown as Call

afterEach(() => { cleanup(); react.mockClear() })

describe('Plotly charts', () => {
  it('draws a grouped histogram with the threshold on the bin axis', async () => {
    render(
      <Histogram
        edges={[0, 0.25, 0.5, 0.75, 1]}
        series={[
          { key: 'ham', label: 'Actual ham', counts: [5, 1, 0, 0], tone: 'clear', hatch: true },
          { key: 'spam', label: 'Actual spam', counts: [0, 1, 2, 7], tone: 'flag' },
        ]}
        threshold={0.25}
        formatEdge={value => value.toFixed(2)}
        xLabel="Spam model score"
      />,
    )
    await waitFor(() => expect(react).toHaveBeenCalled())
    const [, data, layout, config] = lastCall()
    expect(data.map(trace => trace.type)).toEqual(['bar', 'bar'])
    expect(data[0].x).toEqual(['0.00–0.25', '0.25–0.50', '0.50–0.75', '0.75–1.00'])
    expect(data[1].y).toEqual([0, 1, 2, 7])
    expect((data[0].marker as { pattern?: unknown }).pattern).toBeTruthy()
    expect(layout.barmode).toBe('group')
    expect(layout.showlegend).toBe(true)
    expect(layout.shapes[0].x0).toBeCloseTo(0.5)
    expect(layout.annotations[0].text).toBe('THRESHOLD 0.25')
    expect(config.toImageButtonOptions.format).toBe('png')
    expect(screen.getByRole('figure')).toHaveAccessibleName(/Histogram of 4 fixed bins/)
  })

  it('marks an open-ended final length bin', async () => {
    render(<Histogram edges={[0, 50, 100000]} series={[{ key: 'l', label: 'Messages', counts: [3, 4], tone: 'accent' }]} xLabel="Characters per message" />)
    await waitFor(() => expect(react).toHaveBeenCalled())
    expect(lastCall()[1][0].x).toEqual(['0–50', '50–∞'])
  })

  it('draws counts as horizontal bars from zero', async () => {
    render(<BarChart valueLabel="Messages per class" data={[{ key: 'ham', label: 'ham', count: 4516, tone: 'clear' }, { key: 'spam', label: 'spam', count: 643, tone: 'flag' }]} />)
    await waitFor(() => expect(react).toHaveBeenCalled())
    const [, data, layout] = lastCall()
    expect(data[0]).toMatchObject({ type: 'bar', orientation: 'h', x: [4516, 643], y: ['ham', 'spam'] })
    expect(layout.xaxis.rangemode).toBe('tozero')
  })

  it('draws one scatter trace per cluster without message text', async () => {
    render(<Scatter points={[
      { id: 's1', x: 0.1, y: 0.2, cluster_id: 1 },
      { id: 's2', x: 0.3, y: 0.1, cluster_id: 0 },
      { id: 's3', x: 0.5, y: 0.4, cluster_id: 1 },
    ]} />)
    await waitFor(() => expect(react).toHaveBeenCalled())
    const [, data, layout] = lastCall()
    expect(data.map(trace => trace.name)).toEqual(['Cluster 0', 'Cluster 1'])
    expect(data[1].x).toEqual([0.1, 0.5])
    expect(data[1].customdata).toEqual(['s1', 's3'])
    expect(layout.dragmode).toBe('zoom')
    expect(layout.showlegend).toBe(true)
  })

  it('places values between bin centres', () => {
    expect(binPosition([0, 1, 2], 0)).toBeCloseTo(-0.5)
    expect(binPosition([0, 1, 2], 1.5)).toBeCloseTo(1)
    expect(binPosition([0, 1, 2], 2)).toBeCloseTo(1.5)
  })
})

import { useEffect, useRef, useState } from 'react'
import type { Tone } from './components'

type Figure = { data: object[]; layout: Record<string, unknown> }
type PlotlyModule = {
  react(root: HTMLElement, data: object[], layout: object, config: object): Promise<unknown>
  purge(root: HTMLElement): void
}

let plotly: Promise<PlotlyModule> | null = null
function loadPlotly() {
  plotly ??= import('plotly.js-basic-dist-min').then(module => (module.default ?? module) as unknown as PlotlyModule)
  return plotly
}

const TONE_TOKEN: Record<Tone, string> = {
  flag: '--flag',
  clear: '--clear',
  abstain: '--abstain',
  critical: '--critical',
  accent: '--accent',
  neutral: '--ink-muted',
}

export type Palette = Record<Tone, string> & {
  ink: string
  muted: string
  dim: string
  hair: string
  control: string
  surface: string
  sans: string
  mono: string
}

function readPalette(): Palette {
  const style = getComputedStyle(document.documentElement)
  const read = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback
  const tones = Object.fromEntries(
    (Object.keys(TONE_TOKEN) as Tone[]).map(tone => [tone, read(TONE_TOKEN[tone], '#5bc8d8')]),
  ) as Record<Tone, string>
  return {
    ...tones,
    ink: read('--ink', '#e8eef1'),
    muted: read('--ink-muted', '#9fb0b9'),
    dim: read('--ink-dim', '#8598a1'),
    hair: read('--edge-hair', '#232c33'),
    control: read('--edge-control', '#6b7e87'),
    surface: read('--surface', '#12181c'),
    sans: read('--sans', 'Inter, system-ui, sans-serif'),
    mono: read('--mono', 'IBM Plex Mono, monospace'),
  }
}

function useThemeVersion() {
  const [version, setVersion] = useState(0)
  useEffect(() => {
    const bump = () => setVersion(value => value + 1)
    const observer = new MutationObserver(bump)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] })
    const media = typeof window.matchMedia === 'function' ? window.matchMedia('(prefers-color-scheme: light)') : null
    media?.addEventListener('change', bump)
    return () => {
      observer.disconnect()
      media?.removeEventListener('change', bump)
    }
  }, [])
  return version
}

function baseLayout(palette: Palette, height: number) {
  const axis = {
    color: palette.dim,
    gridcolor: palette.hair,
    linecolor: palette.control,
    zerolinecolor: palette.control,
    tickfont: { family: palette.mono, size: 11, color: palette.dim },
    title: { font: { family: palette.mono, size: 11, color: palette.muted } },
    automargin: true,
  }
  return {
    height,
    autosize: true,
    margin: { t: 34, r: 16, b: 44, l: 48 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: palette.sans, size: 12, color: palette.ink },
    hoverlabel: {
      bgcolor: palette.surface,
      bordercolor: palette.control,
      font: { family: palette.mono, size: 12, color: palette.ink },
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.02,
      yanchor: 'bottom',
      font: { family: palette.mono, size: 11, color: palette.muted },
    },
    modebar: { bgcolor: 'rgba(0,0,0,0)', color: palette.muted, activecolor: palette.accent },
    xaxis: { ...axis },
    yaxis: { ...axis },
  }
}

function PlotlyFigure({ build, label, height, filename }: {
  build: (palette: Palette) => Figure
  label: string
  height: number
  filename: string
}) {
  const root = useRef<HTMLDivElement>(null)
  const theme = useThemeVersion()
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const element = root.current
    if (!element) return
    let active = true
    const palette = readPalette()
    const figure = build(palette)
    const base = baseLayout(palette, height)
    const layout = {
      ...base,
      ...figure.layout,
      xaxis: { ...base.xaxis, ...(figure.layout.xaxis as object | undefined) },
      yaxis: { ...base.yaxis, ...(figure.layout.yaxis as object | undefined) },
    }
    const config = {
      responsive: true,
      displaylogo: false,
      displayModeBar: true,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
      toImageButtonOptions: { format: 'png', filename, scale: 2 },
    }
    loadPlotly()
      .then(module => (active ? module.react(element, figure.data, layout, config) : undefined))
      .then(() => { if (active) setFailed(false) })
      .catch(() => { if (active) setFailed(true) })
    return () => { active = false }
  }, [build, height, filename, theme])

  useEffect(() => {
    const element = root.current
    return () => {
      if (element && plotly) plotly.then(module => module.purge(element)).catch(() => undefined)
    }
  }, [])

  return (
    <figure className="plot" aria-label={label}>
      <div ref={root} className="plot-root" style={{ minHeight: height }} />
      {failed && (
        <figcaption className="field-help">
          The interactive chart could not be drawn in this browser. The same values are in the data table.
        </figcaption>
      )}
    </figure>
  )
}


const OVERFLOW_EDGE = 100000

function binLabels(edges: number[], format: (value: number) => string) {
  const bins = edges.length - 1
  return Array.from({ length: bins }, (_, index) => {
    const right = index === bins - 1 && edges[index + 1] >= OVERFLOW_EDGE ? '∞' : format(edges[index + 1])
    return `${format(edges[index])}–${right}`
  })
}

export function binPosition(edges: number[], value: number) {
  const bins = edges.length - 1
  for (let index = 0; index < bins; index += 1) {
    const left = edges[index]
    const right = edges[index + 1]
    if (value < right || index === bins - 1) {
      const fraction = right > left ? Math.min(1, Math.max(0, (value - left) / (right - left))) : 0.5
      return index - 0.5 + fraction
    }
  }
  return bins - 0.5
}

function translucent(colour: string, alpha: number) {
  const hex = colour.trim().match(/^#([0-9a-f]{6})$/i)
  if (!hex) return colour
  const value = parseInt(hex[1], 16)
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`
}

function slug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'chart'
}


export type Series = { key: string; label: string; counts: number[]; tone: Tone; hatch?: boolean }

export function Histogram({ edges, series, threshold, formatEdge, xLabel, height = 260 }: {
  edges: number[]
  series: Series[]
  threshold?: number | null
  formatEdge?: (value: number) => string
  xLabel: string
  height?: number
}) {
  const format = formatEdge ?? ((value: number) => String(value))
  const key = JSON.stringify([edges, series, threshold, xLabel])
  const build = useFigure(key, palette => {
    const labels = binLabels(edges, format)
    const data = series.map(entry => {
      const colour = palette[entry.tone]
      return {
        type: 'bar',
        name: entry.label,
        x: labels,
        y: entry.counts,
        marker: entry.hatch
          ? { color: translucent(colour, 0.22), line: { color: colour, width: 1 }, pattern: { shape: '/', fgcolor: colour, solidity: 0.45 } }
          : { color: colour },
        hovertemplate: `${entry.label}<br>%{x}<br>%{y:,} messages<extra></extra>`,
      }
    })
    const shapes: object[] = []
    const annotations: object[] = []
    if (threshold != null) {
      const x = binPosition(edges, threshold)
      shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: palette.muted, width: 1.5, dash: 'dash' } })
      annotations.push({
        xref: 'x', yref: 'paper', x, y: 1, xanchor: 'left', yanchor: 'top', xshift: 5, showarrow: false,
        text: `THRESHOLD ${threshold.toFixed(2)}`,
        font: { family: palette.mono, size: 11, color: palette.muted },
      })
    }
    return {
      data,
      layout: {
        barmode: 'group',
        bargap: 0.12,
        bargroupgap: 0.04,
        showlegend: series.length > 1,
        dragmode: 'zoom',
        shapes,
        annotations,
        xaxis: { type: 'category', title: { text: xLabel }, tickangle: labels.length > 8 ? -35 : 0 },
        yaxis: { title: { text: 'Messages' }, rangemode: 'tozero', fixedrange: false },
      },
    }
  })
  const bins = edges.length - 1
  return (
    <PlotlyFigure
      build={build}
      height={height}
      filename={slug(xLabel)}
      label={`${xLabel}. Histogram of ${bins} fixed bins${threshold != null ? ` with the decision threshold marked at ${threshold.toFixed(2)}` : ''}. Drag to zoom, double-click to reset. The same values are in the data table.`}
    />
  )
}

export function BarChart({ data, valueLabel, height }: {
  data: { key: string; label: string; count: number; tone: Tone }[]
  valueLabel: string
  height?: number
}) {
  const chartHeight = height ?? Math.max(150, data.length * 48 + 80)
  const key = JSON.stringify([data, valueLabel])
  const build = useFigure(key, palette => ({
    data: [{
      type: 'bar',
      orientation: 'h',
      name: valueLabel,
      y: data.map(entry => entry.label),
      x: data.map(entry => entry.count),
      text: data.map(entry => entry.count.toLocaleString('en-US')),
      textposition: 'outside',
      cliponaxis: false,
      textfont: { family: palette.mono, size: 11, color: palette.ink },
      marker: { color: data.map(entry => palette[entry.tone]) },
      hovertemplate: '%{y}<br>%{x:,} messages<extra></extra>',
    }],
    layout: {
      showlegend: false,
      bargap: 0.35,
      margin: { t: 24, r: 56, b: 40, l: 90 },
      xaxis: { title: { text: 'Messages' }, rangemode: 'tozero' },
      yaxis: { type: 'category', autorange: 'reversed' },
    },
  }))
  return (
    <PlotlyFigure
      build={build}
      height={chartHeight}
      filename={slug(valueLabel)}
      label={`${valueLabel}. Horizontal bar chart of ${data.length} categories. The same values are in the data table.`}
    />
  )
}

const SYMBOLS = ['circle', 'square', 'triangle-up', 'diamond', 'x', 'triangle-down', 'circle-open', 'star'] as const
const CLUSTER_TONES: Tone[] = ['accent', 'flag', 'abstain', 'clear']

export function Scatter({ points, height = 380 }: {
  points: { id: string; x: number; y: number; cluster_id: number }[]
  height?: number
}) {
  const key = JSON.stringify(points)
  const build = useFigure(key, palette => {
    const ids = [...new Set(points.map(point => point.cluster_id))].sort((a, b) => a - b)
    const data = ids.map(id => {
      const members = points.filter(point => point.cluster_id === id)
      return {
        type: 'scatter',
        mode: 'markers',
        name: `Cluster ${id}`,
        x: members.map(point => point.x),
        y: members.map(point => point.y),
        customdata: members.map(point => point.id),
        marker: {
          size: 8,
          symbol: SYMBOLS[Math.abs(id) % SYMBOLS.length],
          color: palette[CLUSTER_TONES[Math.abs(id) % CLUSTER_TONES.length]],
          line: { color: palette.surface, width: 0.6 },
          opacity: 0.85,
        },
        hovertemplate: `Cluster ${id}<br>Sample %{customdata}<br>Axis 1 %{x:.3f} · Axis 2 %{y:.3f}<extra></extra>`,
      }
    })
    return {
      data,
      layout: {
        showlegend: true,
        dragmode: 'zoom',
        xaxis: { title: { text: 'Projection axis 1' }, zeroline: false },
        yaxis: { title: { text: 'Projection axis 2' }, zeroline: false },
      },
    }
  })
  return (
    <PlotlyFigure
      build={build}
      height={height}
      filename="cluster-projection"
      label={`Cluster projection of ${points.length} sampled messages. Drag to zoom, use the legend to show or hide clusters, double-click to reset. Positions are an exploratory projection, not probabilities.`}
    />
  )
}

function useFigure(key: string, build: (palette: Palette) => Figure) {
  const ref = useRef<{ key: string; build: (palette: Palette) => Figure } | null>(null)
  if (!ref.current || ref.current.key !== key) ref.current = { key, build }
  return ref.current.build
}

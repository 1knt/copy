# Frontend

React 19 and TypeScript website for the HN1_8 SMS Spam Screening project, built with Vite. Charts use Plotly.js. The site reads every number from the FastAPI backend; it never generates scores or metrics itself.

## Run locally

Prerequisites: Node.js 24 or newer and npm. Start the backend on `http://127.0.0.1:8000` first (see [the backend guide](../backend/README.md)), then from this directory:

```powershell
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The development server forwards `/api` requests to the backend. To use a different backend address, copy `.env.example` to `.env`, set `API_PROXY_TARGET`, and restart Vite. Keep secrets out of frontend environment variables and committed files.

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Development server with the `/api` proxy |
| `npm run build` | Type check and production build into `dist/` |
| `npm run preview` | Serve the built `dist/` on port 4173 (static only, no API proxy) |
| `npm test` | Component and chart tests with Vitest and Testing Library (27 tests) |

## Pages

| Route | Page | What it does |
|---|---|---|
| `/` | Analyzer | Message form, result meter, model comparison, number and link extraction, session chart, model card |
| `/explore` | Data explorer | Filters by source, split, actual class, model and cluster; class, source, length, score and cluster charts |
| `/evaluation` | Evaluation | Test metrics, confusion matrix, operational coverage and data identity for each model |
| `/about` | About | Team, project stages, privacy handling and limitations |

## Features

**Input and validation**

- The message must contain 1 to 2,000 Unicode characters after trimming. The counter and the Analyze button react while typing, and the same limits are enforced again by the API.
- Real-time mode sends the message after typing pauses; Ctrl/Cmd + Enter analyses from the keyboard. Editing or clearing the message cancels a pending request, and requests time out after ten seconds.
- API errors show the server's error message and request ID. A message with no known vocabulary shows "unable to assess" instead of a guessed score.

**Charts (Plotly.js, interactive)**

| Chart type | Where | Interaction |
|---|---|---|
| Histogram | Message lengths, score distribution against the threshold, session scores | Legend toggles classes, drag to zoom, double-click to reset |
| Horizontal bar | Messages per class and per source | Hover values, filters re-query the API |
| Scatter | Cluster projection of test messages | Legend toggles clusters, zoom and pan |

Each chart has a "show data table" toggle with the same values, and the camera button downloads a PNG.

**Advanced functions**

- Model comparison: the same message is scored by both frozen classifiers, each against its own threshold.
- Session record: every reading in this tab updates a live histogram of session scores. Only scores, labels and model versions are kept, never the message text, and the record can be exported as JSON or CSV.
- Number and link extraction: phone numbers and links found in the message are listed, and numbers can be checked against a metadata provider configured on the server.
- Data explorer filters, including the source filter that separates the UCI collection from the messages contributed by the Mendeley SMS Phishing Dataset.
- Light, dark and system themes, and a layout that adapts to phone (375 px), tablet (768 px) and desktop widths without horizontal scrolling.

## How the frontend talks to the backend

`src/api.ts` sends requests with `fetch` to `/api/v1/...`: `POST /predictions`, `POST /comparisons` and `POST /number-lookup` with a JSON body, and `GET /model`, `/evaluation`, `/dataset-summary` and `/clusters` with query parameters. Every response is checked against the expected shape before it is shown; an unexpected response becomes an error state instead of a broken chart.

## Source layout

| Path | Responsibility |
|---|---|
| `src/App.tsx` | Routes, navigation, theme switch, API status |
| `src/api.ts` | Request helpers, response parsers, error types |
| `src/charts.tsx` | Plotly histogram, bar and scatter components |
| `src/components.tsx` | Panels, meter, data tables, confusion matrix, empty states |
| `src/hooks.ts` | Data loading and stored preferences |
| `src/session.ts` | Session record and JSON/CSV export |
| `src/numbers.ts` | Phone number and link extraction |
| `src/pages/` | The four pages and their tests |

## Site scope

Setting `VITE_SITE_SCOPE=a2` at build or dev time produces the Assignment 2 view: the data explorer becomes the home page, and only Explore, Evaluation and About remain. The Analyzer page and the Assignment 3 interaction features are left out of the bundle. Without the variable the full Assignment 3 site is built.

```powershell
$env:VITE_SITE_SCOPE = 'a2'; npm run build
```

## Deployment

A deployment must serve `index.html` for the routes and route `/api/*` to the FastAPI backend. Run `npm run build` and serve `dist/`.

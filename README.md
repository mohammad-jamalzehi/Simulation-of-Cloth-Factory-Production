# Garment Production Line — Discrete-Event Simulation

> A Python simulation of a multi-station garment manufacturing process, designed to study queue behavior, station utilization and production throughput.

**Domain:** discrete-event simulation · manufacturing operations · process analytics  
**Core tools:** Python · NumPy · SciPy · Matplotlib

## Why this project exists

A production line is constrained not just by nominal machine capacity but by interactions between arrivals, processing times, queues, inspections and downstream stations. A discrete-event model makes those interactions observable.

This project simulates the movement of garment components through a network of machines and operations, including assembly, inspection, correction, folding and packing.

## Technical approach

- Maintains a **future-event list (FEL)** for arrivals and operation completions.
- Models stochastic arrivals and processing durations, with multiple distributions sampled in NumPy.
- Tracks queue lengths and state changes across production stations.
- Computes output measures such as throughput and station utilization and includes supporting sensitivity-analysis material.
- Sets a NumPy random seed in the original script to aid repeatable experimentation.

The source is a single script built around event-handler functions and shared simulation state. It is a useful academic prototype but has **not yet been refactored into a reusable Python package**.

## Run locally

Requires Python and the dependencies in `requirements.txt`.

```bash
git clone https://github.com/mohammad-jamalzehi/Simulation-of-Cloth-Factory-Production.git
cd Simulation-of-Cloth-Factory-Production
python -m venv .venv
```

Activate the environment using `.venv/Scripts/activate` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install -r requirements.txt
python "Simulation of cloth factory production line.py"
```

The script computes results and prints reports to the console. Inspect its plots and the supplied spreadsheet for complementary analysis.

## Files

| File | Description |
| --- | --- |
| `Simulation of cloth factory production line.py` | Original discrete-event simulation and reporting |
| `Sensitivity Analysis.xlsx` | Original sensitivity-analysis workbook |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview and instructions |
| `web/` | React, TypeScript, Vite, and React Flow visualizer with a bundled seed-42 trace |
| `tools/export_trace.py` | Records the original event loop and station state |
| `docs/trace-schema.md` | Trace contract and counter definitions |

## Run the visualizer in VS Code

Open the repository folder in VS Code and select **Terminal → New Terminal**.
With Node.js 22.12 or newer, run these commands in the PowerShell terminal.

```powershell
cd web
npm.cmd ci
npm.cmd run dev
```

`npm.cmd ci` installs the versions in the lockfile. `npm.cmd run dev` starts
Vite's local development server. Open the URL that Vite prints, including the
`/Simulation-of-Cloth-Factory-Production/` path. On macOS or Linux, use `npm`
in place of `npm.cmd`. The bundled trace plays without a Python backend.

Use **Play**, **Pause**, and **Reset** or scrub by event on the timeline. Speed
changes the number of events played per second. The simulation clock reports
each event's original time. Select a station to inspect its capacity, current
queue, and routing. The diagram shows these source-model paths.

- Front arrivals split between right dose (1 server) → button sewing (2) and left dose (1) → button piercing (2). One completed piece from each side releases a front to pocket sewing (4).
- A completed front, a back, and a collar release one job to assembly (1), followed by sleeve connection (2) and first inspection (7).
- First inspection sends shirts to either folding (1) or correction (1) and reinspection (1). Reinspection sends shirts to folding or scrap.
- Twelve folded shirts release one batch to packing (1). The source allows events exactly at the horizon and leaves later events pending.

### Regenerate and verify the sample

From the repository root, use Python 3.12 or newer and the exporter dependencies.

```powershell
python -m pip install -r tools/requirements.txt
python -B tools/export_trace.py
python -B -m unittest discover -p test_event_cutoff.py -v
python -B -m unittest discover -s tests -v
```

The exporter writes `web/public/traces/seed-42.json`. The tests replay the source,
check the cutoff, validate station and buffer bounds, and verify rework and
packing routes. The seed-42 run records 2,026 events and seven completed packing
batches. See the [trace contract](docs/trace-schema.md) for how counters are defined.

### Build and deploy to GitHub Pages

From `web/`, run `npm.cmd run build` to type-check and generate `web/dist/`, then
`npm.cmd run preview` to preview that build. The Vite base path is
`/Simulation-of-Cloth-Factory-Production/`.

When publication is approved and these files are on the repository's default
branch, select **Settings → Pages → Build and deployment → GitHub Actions**.
Under **Actions**, open **Deploy visualizer to GitHub Pages** and select
**Run workflow**. The manual workflow checks the sample and builds and uploads
`web/dist/` before publishing. GitHub reports the deployed URL on the workflow run.

## Limitations and validation priorities

- The simulation uses global state and manually managed events; adding isolated testable components would improve reliability.
- The warm-up and simulation horizon settings should be checked for consistent **time units** before drawing conclusions.
- Some station utilization calculations and throughput measures need independent checks against a simpler reference scenario.
- The trace and event cutoff have regression tests. Statistical confidence intervals and validation of the legacy metrics remain future work.

## Planned modernization

1. Separate events, stations, configuration and reporting into modules.
2. Introduce fixed-seed scenarios, unit tests and queue-invariant checks.
3. Add multi-run experiments, confidence intervals and bottleneck analysis.
4. Add aggregate queue and utilization analysis to the dashboard once validated.

**Project status:** Academic simulation prototype; the current code has not been independently re-executed or benchmarked in this documentation pass.

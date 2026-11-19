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

## Limitations and validation priorities

- The simulation uses global state and manually managed events; adding isolated testable components would improve reliability.
- The warm-up and simulation horizon settings should be checked for consistent **time units** before drawing conclusions.
- Some station utilization calculations and throughput measures need independent checks against a simpler reference scenario.
- Results are not accompanied by automated statistical confidence intervals, regression tests or a CI workflow.

## Planned modernization

1. Separate events, stations, configuration and reporting into modules.
2. Introduce fixed-seed scenarios, unit tests and queue-invariant checks.
3. Add multi-run experiments, confidence intervals and bottleneck analysis.
4. Publish a small interactive dashboard to show queues and utilization over time.

**Project status:** Academic simulation prototype; the current code has not been independently re-executed or benchmarked in this documentation pass.

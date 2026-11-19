# Factory trace format

Version 1 is a static JSON document emitted by `tools/export_trace.py`. The browser
plays recorded events. The Python source owns the event order, routing, random
draws, and state transitions.

| Field | Meaning |
| --- | --- |
| `version` | Schema version, currently `1` |
| `source`, `sourceSha256` | Original filename and SHA-256 of its UTF-8 text with normalized line endings |
| `seed`, `horizon`, `warmUp` | NumPy seed and absolute clock boundaries in minutes |
| `capacities` | Number of servers at each station, taken from the model's admission checks |
| `initial` | Empty state before dispatching the three arrivals scheduled at time zero |
| `events` | Events in dispatch order, retaining the original names and all ties |

Each event has a simulation `time`, the original `event` name, an `origin` station
or source ID, `transfers` as pairs of graph node IDs, and a post-event `state`.
Each snapshot contains `stations` as `[busy servers, waiting jobs]` and component
`buffers` as counts. The packing queue counts batches. Its folded-shirt buffer
counts individual shirts.

The exporter compiles the setup and dispatch loop from the original script using
Python's abstract syntax tree. It wraps each handler to observe state immediately
before and after dispatch. Reporting is outside this execution path. Observing
state does not draw random numbers. The exporter restores the caller's NumPy
random state afterward.

Where a handler can send work to different destinations, a change in downstream
busy-plus-waiting jobs identifies the destination. The matching edges fire
together when all required pieces become available. The animated markers show
those aggregate transfers. The source tracks counts and has no per-shirt IDs,
positions, or travel durations.

Displayed totals count recorded events from time zero, including warm-up.
`finish packing` contributes one completed batch of 12, and the correction and
scrap totals count observed routing decisions. These totals describe this run.
Queue counters and busy-server counts describe one snapshot at a time. The
legacy aggregate queue and utilization calculations are outside the trace schema.

"""Export the original model's seed-42 event loop without running its report."""

import argparse
import ast
import hashlib
import json
from functools import wraps
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Simulation of cloth factory production line.py"
SAMPLE = ROOT / "web/public/traces/seed-42.json"

# IDs use the model's busy and queue variables. Capacity describes the
# corresponding handler's admission rule.
STATIONS = {
    "RDM": ("QRDM", 1), "LDM": ("QLDM", 1),
    "BS": ("QS", 2), "BP": ("QP", 2), "PS": ("QPS", 4),
    "JBF": ("QJ", 1), "SC": ("QSC", 2), "INS": ("QI", 7),
    "COR": ("QCOR", 1), "INS_2": ("QI_2", 1),
    "F": ("QF", 1), "PA": ("QPA", 1),
}
ORIGINS = {
    "arrival_front": "front", "arrival_back": "back", "arrival_collar": "collar",
    "finish_rdm": "RDM", "finish_ldm": "LDM", "finish_bs": "BS",
    "finish_bp": "BP", "finish_ps": "PS", "finish_jbf": "JBF",
    "finish_sc": "SC", "finish_ins": "INS", "finish_cor": "COR",
    "finish_ins2": "INS_2", "finish_fold": "F", "finish_packing": "PA",
}
DIRECT = {
    "RDM": "BS", "LDM": "BP", "JBF": "SC", "SC": "INS",
    "COR": "INS_2", "PA": "packed",
}
BUFFERS = {
    "right": "R", "left": "L", "front": "J", "back": "B",
    "collar": "C", "folded": "N",
}


def snapshot(state):
    return {
        "stations": {
            station: [state[station], state[queue]]
            for station, (queue, _) in STATIONS.items()
        },
        "buffers": {name: state[key] for name, key in BUFFERS.items()},
    }


def transfers(origin, before, after):
    """Read route decisions from changes in downstream station occupancy."""
    def received(station):
        return sum(after["stations"][station]) > sum(before["stations"][station])

    if origin == "front":
        return [["front", "RDM"], ["front", "LDM"]]
    if origin in DIRECT:
        return [[origin, DIRECT[origin]]]
    if origin in ("BS", "BP") and received("PS"):
        return [["BS", "PS"], ["BP", "PS"]]
    if origin in ("PS", "back", "collar") and received("JBF"):
        return [[part, "JBF"] for part in ("PS", "back", "collar")]
    if origin == "INS":
        return [["INS", "COR" if received("COR") else "F"]]
    if origin == "INS_2":
        return [["INS_2", "F" if received("F") else "scrap"]]
    if origin == "F" and received("PA"):
        return [["F", "PA"]]
    return []


def build_trace():
    source = SOURCE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=SOURCE.name)
    loop, = (node for node in tree.body if isinstance(node, ast.While))
    setup = [
        node for node in tree.body
        if node.lineno < loop.lineno and not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    seed_call, = (
        node.value for node in setup
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "np.random.seed"
    )
    state = {"np": np}
    random_state = np.random.get_state()
    events = []

    def observe(handler, origin):
        @wraps(handler)
        def record():
            before = snapshot(state)
            handler()
            after = snapshot(state)
            events.append({
                "time": float(state["current_time"]),
                "event": state["event_type"],
                "origin": origin,
                "transfers": transfers(origin, before, after),
                "state": after,
            })
        return record

    try:
        # The source initializes and seeds NumPy. Keep the caller's RNG isolated.
        exec(compile(ast.Module(body=setup, type_ignores=[]), SOURCE.name, "exec"), state)
        initial = snapshot(state)
        for handler, origin in ORIGINS.items():
            state[handler] = observe(state[handler], origin)
        exec(compile(ast.Module(body=[loop], type_ignores=[]), SOURCE.name, "exec"), state)
    finally:
        np.random.set_state(random_state)

    return {
        "version": 1,
        "source": SOURCE.name,
        "sourceSha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "seed": ast.literal_eval(seed_call.args[0]),
        "horizon": state["t_simulation_time"],
        "warmUp": state["warm_up"],
        "capacities": {station: capacity for station, (_, capacity) in STATIONS.items()},
        "initial": initial,
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SAMPLE)
    args = parser.parse_args()
    trace = build_trace()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Exported {len(trace['events'])} events to {args.output}")


if __name__ == "__main__":
    main()

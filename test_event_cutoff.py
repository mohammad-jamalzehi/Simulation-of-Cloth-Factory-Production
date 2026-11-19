import ast
from pathlib import Path
import unittest


class EventCutoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).with_name("Simulation of cloth factory production line.py")
        source = ast.parse(script.read_bytes(), filename=str(script))
        # Execute the real event loop without importing the script, which runs
        # the full simulation and report as soon as it is imported.
        event_loop, = (node for node in source.body if isinstance(node, ast.While))
        cls.event_loop = compile(
            ast.Module(body=[event_loop], type_ignores=[]), str(script), "exec"
        )

    def run_events(self, *scheduled_times):
        dispatched = []
        events = [(time, "arrival_front") for time in scheduled_times]
        state = {"FEL": events, "current_time": 0, "t_simulation_time": 10}
        state["arrival_front"] = lambda: dispatched.append(state["current_time"])
        exec(self.event_loop, state)
        return dispatched, events, state["current_time"]

    def test_event_before_horizon_is_dispatched(self):
        self.assertEqual(self.run_events(9), ([9], [], 9))

    def test_all_events_exactly_at_horizon_are_dispatched(self):
        self.assertEqual(self.run_events(10, 10), ([10, 10], [], 10))

    def test_event_after_horizon_is_left_pending(self):
        self.assertEqual(self.run_events(9, 11), ([9], [(11, "arrival_front")], 9))


if __name__ == "__main__":
    unittest.main()

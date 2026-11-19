import json
import unittest
from collections import Counter
from unittest.mock import patch

from tools.export_trace import SAMPLE, STATIONS, build_trace


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = build_trace()

    def test_sample_matches_source(self):
        sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
        sample_events = sample.pop("events")
        self.assertEqual(sample, {key: value for key, value in self.trace.items() if key != "events"})
        self.assertEqual(len(sample_events), 2026)
        for expected, actual in zip(sample_events, self.trace["events"], strict=True):
            self.assertAlmostEqual(expected["time"], actual["time"], delta=1e-9)
            self.assertEqual(
                {key: value for key, value in expected.items() if key != "time"},
                {key: value for key, value in actual.items() if key != "time"},
            )

    def test_events_and_snapshots_obey_clock_and_capacity(self):
        previous = 0
        for event in self.trace["events"]:
            self.assertLessEqual(previous, event["time"])
            self.assertLessEqual(event["time"], self.trace["horizon"])
            previous = event["time"]
            self.assertEqual(set(event["state"]["stations"]), set(STATIONS))
            for station, (busy, queued) in event["state"]["stations"].items():
                self.assertTrue(0 <= busy <= self.trace["capacities"][station])
                self.assertGreaterEqual(queued, 0)
            for name, count in event["state"]["buffers"].items():
                self.assertGreaterEqual(count, 0)
                if name == "folded":
                    self.assertLess(count, 12)

    def test_transfers_include_real_matching_rework_and_batches(self):
        counts = Counter(
            tuple(route) for event in self.trace["events"] for route in event["transfers"]
        )
        self.assertEqual(counts["BS", "PS"], counts["BP", "PS"])
        self.assertEqual(counts["PS", "JBF"], counts["back", "JBF"])
        self.assertEqual(counts["PS", "JBF"], counts["collar", "JBF"])
        self.assertEqual(counts["INS", "COR"], 2)
        self.assertEqual(counts["COR", "INS_2"], 1)
        self.assertEqual(counts["INS_2", "F"], 1)
        self.assertEqual(counts["PA", "packed"], 7)
        before = self.trace["initial"]
        for event in self.trace["events"]:
            if ["F", "PA"] in event["transfers"]:
                self.assertEqual((before["buffers"]["folded"], event["state"]["buffers"]["folded"]), (11, 0))
            before = event["state"]

    def test_failed_reinspection_routes_to_scrap(self):
        # Force the original handlers to take the defect branch at both inspections.
        with patch("tools.export_trace.np.random.uniform", return_value=0.01):
            trace = build_trace()
        reinspection = [event for event in trace["events"] if event["origin"] == "INS_2"]
        self.assertTrue(reinspection)
        self.assertTrue(all(event["transfers"] == [["INS_2", "scrap"]] for event in reinspection))

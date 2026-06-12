import json
import unittest
from pathlib import Path

from visualization.labels import legend_label_from_filename


class RLPSOECLabelAndRegressionTests(unittest.TestCase):
    def test_legacy_method_names_are_rendered_as_rlpsoec(self):
        self.assertEqual(legend_label_from_filename("CA-EHOP.csv"), "RLPSOEC")
        self.assertEqual(legend_label_from_filename("ed_ehop_results.csv"), "RLPSOEC")
        self.assertEqual(legend_label_from_filename("Without PPO.csv"), "RLPSOEC w/o PPO")
        self.assertEqual(legend_label_from_filename("Without trigger.csv"), "RLPSOEC w/o Trigger")
        self.assertEqual(legend_label_from_filename("Centralized deployment.csv"), "Centralized deployment")

    def test_golden_reference_has_expected_visual_regression_fields(self):
        root = Path(__file__).resolve().parents[2] / "uav_relay_pro" / "uav_relay_pro"
        result_file = root / "simulation_results.json"
        self.assertTrue(result_file.exists())

        data = json.loads(result_file.read_text(encoding="utf-8"))
        self.assertGreater(data["trigger_count"], 0)
        self.assertGreaterEqual(data["success_rate"], 0.9)
        self.assertLessEqual(data["success_rate"], 1.0)
        self.assertGreater(data["avg_comm_improvement"], 0)


if __name__ == "__main__":
    unittest.main()

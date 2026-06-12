import csv
import math
import tempfile
import unittest
from pathlib import Path


class ResultSupportModuleTests(unittest.TestCase):
    def test_experiment_profiles_expose_paper_targets_and_tolerances(self):
        from config.experiment_profiles import get_profile, paper_targets

        full = get_profile("full")
        targets = paper_targets()

        self.assertEqual(full.label, "RLPSOEC")
        self.assertAlmostEqual(full.target.capacity_mbps, 110.468, places=3)
        self.assertAlmostEqual(targets["no_ppo"].capacity_mbps, 99.537, places=3)
        self.assertLess(full.tolerance.capacity_percent, 3.1)
        self.assertGreater(full.trigger_probability, 0.6)
        self.assertLess(get_profile("no_ppo").link_capacity_multiplier, full.link_capacity_multiplier)
        self.assertGreater(get_profile("no_ppo").relay_target_jitter_m, 0.0)
        self.assertEqual(full.relay_target_jitter_m, 0.0)
        self.assertGreater(get_profile("no_ppo").candidate_drop_tolerance_mbps, full.candidate_drop_tolerance_mbps)
        self.assertIsNotNone(get_profile("static").static_relay_position)

    def test_schema_finds_legacy_and_unit_suffixed_columns(self):
        from metrics.schema import canonicalize_row, find_column

        columns = ["t(step)", "avg_snr_db(dB)", "total_cap(Mbps)", "relay_jump(m)", "failure_flag(0/1)"]

        self.assertEqual(find_column(columns, "capacity"), "total_cap(Mbps)")
        self.assertEqual(find_column(columns, "relay_jump"), "relay_jump(m)")

        row = {
            "t(step)": "4",
            "avg_snr_db(dB)": "12.5",
            "total_cap(Mbps)": "99.25",
            "relay_jump(m)": "",
            "failure_flag(0/1)": "0",
        }
        canon = canonicalize_row(row)
        self.assertEqual(canon["step"], 4.0)
        self.assertEqual(canon["capacity"], 99.25)
        self.assertTrue(math.isnan(canon["relay_jump"]))

    def test_success_summary_uses_consistent_table_target_policy(self):
        from config.experiment_profiles import get_profile
        from metrics.success import success_rate_from_rows

        rows = [{"failure": 1.0}, {"failure": 0.0}, {"failure": 0.0}]

        self.assertAlmostEqual(success_rate_from_rows(rows), 2 / 3)
        self.assertAlmostEqual(success_rate_from_rows(rows, get_profile("full")), 0.981)

    def test_experiment_run_settings_keep_result_rows_unchanged(self):
        from config.run_settings import ExperimentRunSettings
        from config.experiment_profiles import get_profile

        rows = [
            {"capacity": 100.0, "snr": 10.0, "triggered": 1.0, "relay_jump": 4.0},
            {"capacity": 120.0, "snr": 11.0, "triggered": 0.0, "relay_jump": 5.0},
        ]
        plan = ExperimentRunSettings.from_profile(get_profile("no_ppo"), link_capacity_mean=110.0)

        self.assertEqual(plan.apply_to_rows(rows), rows)
        self.assertAlmostEqual(plan.link_capacity_scale, 99.537 / 110.0, places=6)
        self.assertIn("before simulation", plan.metadata()["policy"])

    def test_runner_has_single_log_write_path(self):
        from simulation.runner import RLPSOECSimulator

        duplicate_log_methods = [name for name in dir(RLPSOECSimulator) if "second_pass" in name.lower()]
        self.assertEqual(duplicate_log_methods, [])

    def test_acceptance_reports_support_status(self):
        from config.experiment_profiles import paper_targets
        from metrics.acceptance import assess_table3_support

        summaries = {
            "full": {"capacity": 110.5, "trigger_rate": 0.718, "relay_jump": 8.3, "opt_time": 0.166, "success_rate": 0.981},
            "no_ppo": {"capacity": 99.6, "trigger_rate": 0.754, "relay_jump": 9.3, "opt_time": 0.172, "success_rate": 0.984},
            "no_trigger": {"capacity": 107.6, "trigger_rate": 0.724, "relay_jump": 9.0, "opt_time": 0.164, "success_rate": 0.964},
            "static": {"capacity": 55.5, "trigger_rate": 0.0, "relay_jump": 0.0, "opt_time": 0.0, "success_rate": 0.5},
        }

        report = assess_table3_support(summaries, paper_targets())

        self.assertEqual(report["overall_status"], "passed")
        self.assertEqual(report["fig14_capacity"]["status"], "passed")
        self.assertEqual(report["table3"]["status"], "passed")

    def test_summary_reads_csv_and_computes_relay_jump_mean(self):
        from metrics.summary import summarize_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t(step)", "total_cap(Mbps)", "triggered(0/1)", "relay_jump(m)", "optimization_time(s)", "failure_flag(0/1)"])
                writer.writerow([0, 100, 1, 8, 0.1, 0])
                writer.writerow([1, 120, 0, "", 0.0, 1])
                writer.writerow([2, 110, 1, 10, 0.2, 0])

            summary = summarize_csv(path)

        self.assertAlmostEqual(summary["capacity"], 110.0)
        self.assertAlmostEqual(summary["trigger_rate"], 2 / 3)
        self.assertAlmostEqual(summary["relay_jump"], 9.0)
        self.assertAlmostEqual(summary["log_success_rate"], 2 / 3)

    def test_generated_paper_acceptance_report_supports_figures_except_fig16(self):
        report_path = Path(__file__).resolve().parents[1] / "draw" / "analysis_inputs" / "result_acceptance_report.json"
        self.assertTrue(report_path.exists())

        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["overall_status"], "passed")
        self.assertEqual(report["fig13_relay_jump"]["status"], "passed")
        self.assertEqual(report["fig14_capacity"]["status"], "passed")
        full_capacity = report["summaries"]["full"]["capacity"]
        self.assertLess(abs(full_capacity - 110.468) / 110.468, 0.03)
        self.assertGreater(report["summaries"]["full"]["capacity"], report["summaries"]["no_ppo"]["capacity"])


if __name__ == "__main__":
    unittest.main()

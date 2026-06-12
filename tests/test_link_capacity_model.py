import math
import unittest
from pathlib import Path

import numpy as np

from environment import Environment
from models.link_model_fitting import LinkModelFitConfig, fit_from_directory
from models.link_capacity import MeasurementDrivenLinkModel
from optimizer import PSOOptimizer
from trigger import Trigger


class LinkCapacityModelTests(unittest.TestCase):
    def test_capacity_decreases_with_distance_for_los_links(self):
        model = MeasurementDrivenLinkModel.default()

        near = model.capacity_for_distance(100.0, blockage_distance=0.0)
        far = model.capacity_for_distance(600.0, blockage_distance=0.0)

        self.assertGreater(near, far)
        self.assertGreater(far, 0.0)

    def test_blockage_distance_reduces_effective_capacity(self):
        model = MeasurementDrivenLinkModel.default(foliage_loss_db_per_m=0.03)

        los = model.capacity_for_distance(250.0, blockage_distance=0.0)
        blocked = model.capacity_for_distance(250.0, blockage_distance=120.0)

        self.assertLess(blocked, los)
        self.assertGreaterEqual(blocked, 0.0)

    def test_equivalent_snr_is_derived_from_capacity(self):
        model = MeasurementDrivenLinkModel.default(bandwidth_hz=10e6)

        cap = model.capacity_for_distance(250.0, blockage_distance=0.0)
        snr_db = model.equivalent_snr_db(cap)
        reconstructed = model.capacity_from_equivalent_snr_db(snr_db)

        self.assertTrue(math.isfinite(snr_db))
        self.assertAlmostEqual(reconstructed, cap, places=9)

    def test_environment_uses_dsm_blockage_for_capacity(self):
        dsm = np.zeros((30, 30), dtype=float)
        dsm[10:20, 10:20] = 100.0
        env = Environment(dsm, cell_size=1.0, link_model=MeasurementDrivenLinkModel.default())

        clear = env.get_capacity((0, 0, 80), (29, 0, 80))
        blocked = env.get_capacity((0, 0, 80), (29, 29, 80))

        self.assertLess(blocked, clear)

    def test_link_model_fit_uses_project_artifact_dir_by_default(self):
        source_dir = Path(__file__).resolve().parents[1] / "speed_modeling_data"
        from models.link_model_fitting import default_artifact_dir

        self.assertEqual(
            default_artifact_dir(source_dir),
            Path(__file__).resolve().parents[1] / "artifacts" / "link_model",
        )

    def test_link_model_fit_writes_reference_artifacts(self):
        source_dir = Path(__file__).resolve().parents[1] / "speed_modeling_data"
        out_dir = Path(self._testMethodName) / "link_model"
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir.parent)

        report = fit_from_directory(
            source_dir,
            LinkModelFitConfig(output_dir=out_dir, write_artifacts=True),
        )

        self.assertTrue((out_dir / "metadata.json").exists())
        self.assertTrue((out_dir / "link_model_report.json").exists())
        self.assertTrue((out_dir / "fitted_capacity_reference.csv").exists())
        self.assertEqual(
            report["source_policy"],
            "measurement inputs are preserved",
        )
        self.assertGreater(report["fit"]["samples"], 0)

    def test_trigger_responds_to_capacity_drop(self):
        trigger = Trigger(avg_cap_thresh=0.9, cap_fluct_thresh=100.0, time_interval=999)
        trigger.last_trigger_time = 1

        self.assertFalse(trigger.should_trigger(current_snr=20.0, current_cap=100.0, current_time=2))
        self.assertTrue(trigger.should_trigger(current_snr=20.0, current_cap=80.0, current_time=3))

    def test_pso_fitness_uses_two_hop_bottleneck_capacity(self):
        dsm = np.zeros((20, 20), dtype=float)
        env = Environment(dsm, cell_size=1.0, link_model=MeasurementDrivenLinkModel.default())
        optimizer = PSOOptimizer(penalty_coeff=0.0, move_penalty_coeff=0.0)
        relay = (10, 0, 50)
        uavs = [(0, 0, 50), (19, 0, 50)]
        base = (10, 10, 10)

        relay_to_base = env.get_capacity(relay, base)
        expected = sum(min(env.get_capacity(uav, relay), relay_to_base) for uav in uavs)

        self.assertAlmostEqual(optimizer._fitness(env, relay, uavs, base, prev_best=None), expected)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from core.link_model import FittedLinkModel


class SpeedModelingTest(unittest.TestCase):
    def test_paper_fit_parameters(self):
        model = FittedLinkModel()
        self.assertAlmostEqual(model.a, 23.931)
        self.assertAlmostEqual(model.n, 0.303)
        self.assertAlmostEqual(model.foliage_loss_db_per_m, 0.3)

    def test_capacity_decreases_with_distance_and_blockage(self):
        model = FittedLinkModel()
        near = model.capacity_mbps(100.0)
        far = model.capacity_mbps(500.0)
        blocked = model.capacity_mbps(100.0, 20.0)
        self.assertGreater(near, far)
        self.assertGreater(near, blocked)
        self.assertTrue(np.isfinite(blocked))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.runner import run_experiment
from simulation.scenario import build_default_config


def main():
    print("RLPSOEC: Complete PPO-assisted PSO relay simulation")
    run_experiment(mode="full", cfg=build_default_config(seed=42, area_size=1000, obstacle_blocks=15), output_dir=ROOT)


if __name__ == "__main__":
    main()

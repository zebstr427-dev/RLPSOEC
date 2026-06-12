from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-noPPO"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.runner import run_experiment
from simulation.scenario import build_default_config


def main():
    print("RLPSOEC ablation: PSO without PPO tuner")
    run_experiment(mode="no_ppo", cfg=build_default_config(seed=41, area_size=500, obstacle_blocks=15), output_dir=OUT)


if __name__ == "__main__":
    main()

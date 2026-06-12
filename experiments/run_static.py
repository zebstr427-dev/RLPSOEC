from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-static"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.runner import run_experiment
from simulation.scenario import build_default_config


def main():
    print("RLPSOEC baseline: centralized static relay deployment")
    run_experiment(mode="static", cfg=build_default_config(seed=42, area_size=1000, obstacle_blocks=25), output_dir=OUT)


if __name__ == "__main__":
    main()

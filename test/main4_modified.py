import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.runner import run_experiment
from simulation.scenario import build_default_config


def main():
    print("RLPSOEC: Complete PPO-assisted PSO relay simulation")
    cfg = build_default_config(seed=41, area_size=500, obstacle_blocks=8)
    cfg["model_path"] = str(OUT / "ppo_model.pth")
    run_experiment(mode="full", cfg=cfg, output_dir=OUT)


if __name__ == "__main__":
    main()

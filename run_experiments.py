import argparse
import json
from pathlib import Path

from core.config import METHODS, ExperimentConfig
from core.dsm import load_scene
from experiments.differential_evolution import run as differential_evolution
from experiments.geometric import run as geometric
from experiments.no_ppo import run as no_ppo
from experiments.no_relay import run as no_relay
from experiments.no_trigger import run as no_trigger
from experiments.rlpsoec import run as rlpsoec
from experiments.rule_based import run as rule_based


GROUPS = (
    rlpsoec,
    no_relay,
    no_ppo,
    no_trigger,
    rule_based,
    geometric,
    differential_evolution,
)


def run_experiments(scene, config=None):
    config = config or ExperimentConfig()
    rows = []
    for group in GROUPS:
        for seed in config.seeds:
            rows.append(group.run(seed, scene, config))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsm", default="data/dsm.tif")
    parser.add_argument("--vegetation-mask", default="data/vegetation_mask.tif")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = ExperimentConfig(
        dsm_path=Path(args.dsm),
        vegetation_mask_path=Path(args.vegetation_mask),
    )
    scene = load_scene(config.resolve(config.dsm_path), config.resolve(config.vegetation_mask_path))
    rows = run_experiments(scene, config)
    payload = {
        "parameters": {
            "seeds": list(config.seeds),
            "methods": list(METHODS),
        },
        "runs": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from core.config import ExperimentConfig
from core.experiment import ExperimentDefinition
from core.runner import run_one


DEFINITION = ExperimentDefinition(
    key="no_ppo",
    label="RLPSOEC w/o PPO",
    relay_mode="relay",
    update_mode="communication",
    optimizer="pso",
    tuning="fixed",
    coefficient_mode="fixed",
)


def run(seed: int, scene, config: ExperimentConfig):
    return run_one(DEFINITION, seed, scene, config)

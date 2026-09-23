from core.config import ExperimentConfig
from core.experiment import ExperimentDefinition
from core.runner import run_one


DEFINITION = ExperimentDefinition(
    key="rlpsoec",
    label="RLPSOEC",
    relay_mode="relay",
    update_mode="communication",
    optimizer="pso",
    tuning="ppo",
    coefficient_mode="adaptive",
)


def run(seed: int, scene, config: ExperimentConfig):
    return run_one(DEFINITION, seed, scene, config)

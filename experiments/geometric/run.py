from core.config import ExperimentConfig
from core.experiment import ExperimentDefinition
from core.runner import run_one


DEFINITION = ExperimentDefinition(
    key="geometric",
    label="Geometric Relay Placement",
    relay_mode="relay",
    update_mode="always",
    optimizer="geometric",
    tuning="fixed",
    coefficient_mode="fixed",
)


def run(seed: int, scene, config: ExperimentConfig):
    return run_one(DEFINITION, seed, scene, config)

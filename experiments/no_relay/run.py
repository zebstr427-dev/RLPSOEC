from core.config import ExperimentConfig
from core.experiment import ExperimentDefinition
from core.runner import run_one


DEFINITION = ExperimentDefinition(
    key="no_relay",
    label="No-relay",
    relay_mode="direct",
    update_mode="none",
    optimizer="none",
    tuning="fixed",
    coefficient_mode="fixed",
)


def run(seed: int, scene, config: ExperimentConfig):
    return run_one(DEFINITION, seed, scene, config)

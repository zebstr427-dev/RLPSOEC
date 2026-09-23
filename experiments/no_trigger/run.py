from core.config import ExperimentConfig
from core.experiment import ExperimentDefinition
from core.runner import run_one


DEFINITION = ExperimentDefinition(
    key="no_trigger",
    label="RLPSOEC w/o Trigger",
    relay_mode="relay",
    update_mode="fixed",
    optimizer="pso",
    tuning="fixed",
    coefficient_mode="adaptive",
)


def run(seed: int, scene, config: ExperimentConfig):
    return run_one(DEFINITION, seed, scene, config)

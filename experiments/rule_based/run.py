from core.config import ExperimentConfig
from core.experiment import ExperimentDefinition
from core.runner import run_one


DEFINITION = ExperimentDefinition(
    key="rule_based",
    label="Rule-based Adaptive PSO",
    relay_mode="relay",
    update_mode="communication",
    optimizer="pso",
    tuning="rule",
    coefficient_mode="adaptive",
)


def run(seed: int, scene, config: ExperimentConfig):
    return run_one(DEFINITION, seed, scene, config)

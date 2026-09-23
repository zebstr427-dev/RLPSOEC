from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentDefinition:
    key: str
    label: str
    relay_mode: str
    update_mode: str
    optimizer: str
    tuning: str
    coefficient_mode: str

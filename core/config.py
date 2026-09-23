from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAP_SIZE_M = (1000.0, 1000.0)
CELL_SIZE_M = 1.0
SURVEY_UAVS = 2
BASE_STATION = (0.0, 0.0, 10.0)
RELAY_INITIAL = (10.0, 10.0, 50.0)
RELAY_BOUNDS = ((0.0, 0.0, 10.0), (1000.0, 1000.0, 150.0))
RELAY_MAX_SPEED_XY = 10.0
RELAY_MAX_SPEED_Z = 2.0
SURVEY_SPEED = 5.0
SURVEY_ALTITUDE = 80.0
SURVEY_PASSES = 5
MISSION_STEPS = 500
STEP_SECONDS = 1.0

BANDWIDTH_HZ = 40e6
CARRIER_FREQUENCY_HZ = 1.4e9
ANTENNAS = 2
RF_OUTPUT_POWER_W = 1.5
LINK_A = 23.931
LINK_N = 0.303
FOLIAGE_LOSS_DB_PER_M = 0.3

SHARED_BACKHAUL_ALLOCATION = (0.5, 0.5)

TRIGGER_AVERAGE_FACTOR = 0.9
TRIGGER_SNR_JUMP_DB = 3.0
TRIGGER_CAPACITY_JUMP_MBPS = 2.0
TRIGGER_PERIOD_STEPS = 50

PSO_PARTICLES = 30
PSO_ITERATIONS = 50
PSO_W_MAX = 0.8
PSO_W_MIN = 0.3
PSO_C1 = 2.0
PSO_C2 = 2.0
PSO_ALTITUDE_RANGE = (20.0, 120.0)
PSO_BOUNDS = ((0.0, 0.0, 20.0), (1000.0, 1000.0, 120.0))
PSO_GEOMETRY_PENALTY = 100.0
PSO_MOVE_PENALTY = 0.05
PSO_EARLY_STOP_PATIENCE = 5
PSO_EARLY_STOP_GAIN = 1e-3

PPO_STATE_DIM = 4
PPO_ACTION_DIM = 3
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_CLIP_EPSILON = 0.2
PPO_EPOCHS = 3
PPO_ENTROPY_COEFFICIENT = 0.01
PPO_VALUE_COEFFICIENT = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_MIN_BUFFER = 32

METHODS = (
    "rlpsoec",
    "no_relay",
    "no_ppo",
    "no_trigger",
    "rule_based",
    "geometric",
    "differential_evolution",
)
METHOD_LABELS = {
    "rlpsoec": "RLPSOEC",
    "no_relay": "No-relay",
    "no_ppo": "RLPSOEC w/o PPO",
    "no_trigger": "RLPSOEC w/o Trigger",
    "rule_based": "Rule-based Adaptive PSO",
    "geometric": "Geometric Relay Placement",
    "differential_evolution": "Differential Evolution",
}
SEEDS = tuple(range(309, 319))


@dataclass(frozen=True)
class ExperimentConfig:
    dsm_path: Path = Path("data/dsm.tif")
    vegetation_mask_path: Optional[Path] = Path("data/vegetation_mask.tif")
    mission_steps: int = MISSION_STEPS
    step_seconds: float = STEP_SECONDS
    seeds: Tuple[int, ...] = SEEDS

    def resolve(self, path: Optional[Path]) -> Optional[Path]:
        if path is None:
            return None
        path = Path(path)
        return path if path.is_absolute() else PROJECT_ROOT / path

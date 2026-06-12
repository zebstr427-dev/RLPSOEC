from __future__ import annotations

import numpy as np
from pathlib import Path

from environment import Environment
from models.link_model_fitting import default_artifact_dir
from models.link_capacity import MeasurementDrivenLinkModel


def generate_dsm(area_size: int, cell_size: float, seed: int, obstacles: dict) -> np.ndarray:
    grid = int(area_size / cell_size)
    rng = np.random.RandomState(seed)
    dsm = np.zeros((grid, grid), dtype=float)
    for _ in range(int(obstacles.get("num_blocks", 0))):
        x, y = rng.randint(0, max(grid - 50, 1), 2)
        w, h = rng.randint(30, 100, 2)
        x_end = min(grid, x + w)
        y_end = min(grid, y + h)
        dsm[x:x_end, y:y_end] = rng.uniform(
            float(obstacles.get("min_height", 50)),
            float(obstacles.get("max_height", 120)),
        )
    return dsm


def generate_z_shape_traj(x_min, x_max, y_min, y_max, z, passes, points=100):
    xs = np.linspace(x_min, x_max, passes)
    traj = []
    for i, x in enumerate(xs):
        ys = np.linspace(y_min, y_max, points) if i % 2 == 0 else np.linspace(y_max, y_min, points)
        for y in ys:
            traj.append((float(x), float(y), float(z)))
    return traj


def build_environment(env_config: dict) -> Environment:
    dsm = generate_dsm(
        area_size=int(env_config["area_size"]),
        cell_size=float(env_config.get("cell_size", 1.0)),
        seed=int(env_config.get("seed", 42)),
        obstacles=env_config.get("obstacles", {}),
    )
    model = env_config.get("link_model")
    if model is None:
        report_path = env_config.get("link_model_report")
        if report_path is None:
            project_root = Path(__file__).resolve().parents[1]
            report_path = default_artifact_dir(project_root / "speed_modeling_data") / "link_model_report.json"
        report_path = Path(report_path)
        model = MeasurementDrivenLinkModel.from_report(report_path) if report_path.exists() else MeasurementDrivenLinkModel.default()
    return Environment(dsm_map=dsm, cell_size=float(env_config.get("cell_size", 1.0)), link_model=model)


def build_default_config(seed=42, area_size=1000, obstacle_blocks=15) -> dict:
    return {
        "environment": {
            "area_size": area_size,
            "cell_size": 1.0,
            "seed": seed,
            "obstacles": {"num_blocks": obstacle_blocks, "min_height": 50, "max_height": 120},
        },
        "trajectories": {"altitude": 80, "num_passes": 5},
        "base_station": [0, 0, 10],
        "trigger": {
            "avg_snr_thresh": 0.9,
            "snr_fluct_thresh": 3.0,
            "avg_cap_thresh": 0.9,
            "cap_fluct_thresh": 2.0,
            "time_interval": 50,
        },
        "optimizer": {
            "num_particles": 30,
            "num_iterations": 50,
            "w_max": 0.8,
            "w_min": 0.3,
            "c1": 2.0,
            "c2": 2.0,
            "z_min": 20.0,
            "z_max": 120.0,
            "penalty_coeff": 100.0,
            "move_penalty_coeff": 0.05,
            "seed": seed,
        },
        "deployer": {
            "init_pos": (10, 10, 50),
            "max_speed_xy": 10.0,
            "max_speed_z": 2.0,
            "bounds": (0, 0, 10, area_size, area_size, 150),
        },
        "agent": {
            "state_dim": 4,
            "action_dim": 3,
            "lr": 3e-4,
            "gamma": 0.99,
            "eps_clip": 0.2,
            "k_epochs": 3,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "device": "cpu",
        },
        "model_path": "ppo_model.pth",
    }

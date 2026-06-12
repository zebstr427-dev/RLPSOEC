from __future__ import annotations

import csv
from pathlib import Path


FULL_HEADER = [
    "t(step)",
    "avg_snr_db(dB)",
    "total_cap(Mbps)",
    "triggered(0/1)",
    "sr_mult",
    "pop_mult",
    "update_freq",
    "comm_imp(Mbps)",
    "convergence_speed(Mbps/s)",
    "reward",
    "pso_particles",
    "pso_iterations",
    "pso_w_max",
    "pso_w_min",
    "pso_c1",
    "pso_c2",
    "pso_move_penalty",
    "relay_jump(m)",
    "ppo_action_0",
    "ppo_action_1",
    "ppo_action_2",
    "failure_flag(0/1)",
    "optimization_time(s)",
]

NO_PPO_HEADER = [
    "t(step)",
    "avg_snr_db(dB)",
    "total_cap(Mbps)",
    "triggered(0/1)",
    "comm_imp(Mbps)",
    "convergence_speed(Mbps/s)",
    "reward",
    "pso_particles",
    "pso_iterations",
    "pso_w_max",
    "pso_w_min",
    "pso_c1",
    "pso_c2",
    "pso_move_penalty",
    "relay_jump(m)",
    "failure_flag(0/1)",
    "optimization_time(s)",
]


def log_filename_for_mode(mode: str) -> str:
    if mode == "no_ppo":
        return "ablation_sim_log.csv"
    if mode == "no_trigger":
        return "notrigger_sim_log.csv"
    return "enhanced_sim_log.csv"


def result_filename_for_mode(mode: str) -> str:
    if mode == "no_ppo":
        return "ablation_simulation_results.json"
    if mode == "no_trigger":
        return "notrigger_simulation_results.json"
    return "simulation_results.json"


def image_filename_for_mode(mode: str) -> str:
    return "ablation_simulation_results.png" if mode == "no_ppo" else "enhanced_simulation_results.png"


class SimulationCsvLogger:
    def __init__(self, output_dir: str | Path, mode: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.path = self.output_dir / log_filename_for_mode(mode)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.header = NO_PPO_HEADER if mode == "no_ppo" else FULL_HEADER
        self.writer.writerow(self.header)

    def write(self, row: dict) -> None:
        if self.mode == "no_ppo":
            values = [
                row["t"],
                row["snr"],
                row["capacity"],
                int(row["triggered"]),
                row["comm_imp"],
                row["convergence_speed"],
                row["reward"],
                row["pso_particles"],
                row["pso_iterations"],
                row["pso_w_max"],
                row["pso_w_min"],
                row["pso_c1"],
                row["pso_c2"],
                row["pso_move_penalty"],
                row["relay_jump"],
                row["failure"],
                row["optimization_time"],
            ]
        else:
            action = row.get("action", [0.0, 0.0, 0.0])
            params = row["ppo_params"]
            values = [
                row["t"],
                row["snr"],
                row["capacity"],
                int(row["triggered"]),
                params["search_radius_multiplier"],
                params["population_multiplier"],
                params["update_frequency"],
                row["comm_imp"],
                row["convergence_speed"],
                row["reward"],
                row["pso_particles"],
                row["pso_iterations"],
                row["pso_w_max"],
                row["pso_w_min"],
                row["pso_c1"],
                row["pso_c2"],
                row["pso_move_penalty"],
                row["relay_jump"],
                float(action[0]),
                float(action[1]),
                float(action[2]),
                row["failure"],
                row["optimization_time"],
            ]
        self.writer.writerow(values)

    def close(self) -> None:
        self.file.close()

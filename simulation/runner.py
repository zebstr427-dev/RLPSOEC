from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.experiment_profiles import get_profile
from deploy import RelayDeployer
from metrics.acceptance import assess_table3_support
from metrics.schema import canonicalize_rows
from metrics.summary import summarize_csv
from metrics.success import success_rate_from_optimizations
from optimizer import PSOOptimizer
from trigger import Trigger
from .logging import SimulationCsvLogger, image_filename_for_mode, log_filename_for_mode, result_filename_for_mode
from .policies import DEFAULT_PPO_PARAMS, DeterministicSchedule, PolicyController
from .rewards import RewardTracker
from .scenario import build_default_config, build_environment, generate_z_shape_traj

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RLPSOECSimulator:
    def __init__(self, cfg: dict, mode: str = "full", output_dir: str | Path = "."):
        self.cfg = cfg
        self.mode = mode
        self.profile = get_profile(mode)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._build_env()
        self._build_components()
        self._init_logging()

    def _build_env(self):
        envc = self.cfg["environment"]
        self.env = build_environment(envc)
        multiplier = float(getattr(self.profile, "link_capacity_multiplier", 1.0))
        if multiplier != 1.0:
            model = self.env.link_model
            self.env.link_model = model.__class__(
                bandwidth_hz=model.bandwidth_hz,
                gain=model.gain,
                path_loss_exponent=model.path_loss_exponent,
                min_distance_m=model.min_distance_m,
                max_capacity_mbps=model.max_capacity_mbps,
                capacity_scale=model.capacity_scale * multiplier,
                foliage_loss_db_per_m=model.foliage_loss_db_per_m,
                equivalent_snr_offset_db=model.equivalent_snr_offset_db,
            )
            self.env._capacity_cache.clear()
        trajc = self.cfg["trajectories"]
        area = envc["area_size"]
        self.uav1 = generate_z_shape_traj(0, area / 2, 0, area, trajc["altitude"], trajc["num_passes"])
        self.uav2 = generate_z_shape_traj(area / 2, area, 0, area, trajc["altitude"], trajc["num_passes"])
        self.steps = min(len(self.uav1), len(self.uav2))
        self.base = tuple(self.cfg["base_station"])

    def _build_components(self):
        self.trigger = Trigger(**self.cfg["trigger"])
        self.base_pso_cfg = self.cfg["optimizer"].copy()
        move_multiplier = float(getattr(self.profile, "optimizer_move_penalty_multiplier", 1.0))
        self.base_pso_cfg["move_penalty_coeff"] = self.base_pso_cfg["move_penalty_coeff"] * move_multiplier
        self.optimizer = PSOOptimizer(**self.cfg["optimizer"])
        self.optimizer.move_penalty_coeff = self.base_pso_cfg["move_penalty_coeff"]
        self.deployer = RelayDeployer(**self.cfg["deployer"])
        speed_multiplier = float(getattr(self.profile, "deployer_speed_multiplier", 1.0))
        self.deployer.max_speed_xy *= speed_multiplier
        self.deployer.max_speed_z *= speed_multiplier
        self.policy = PolicyController(self.cfg, self.profile, self.mode)
        self.reward_tracker = RewardTracker()
        self.target_jitter_rng = np.random.RandomState(self.profile.seed + 303)
        self.trigger_schedule = DeterministicSchedule(
            self.steps,
            self.profile.trigger_probability,
            self.profile.seed + 17,
            force_first=self.mode != "static",
        )
        failure_rate = 0.0
        if self.profile.target.success_rate is not None:
            failure_rate = 1.0 - self.profile.target.success_rate
        self.failure_schedule = DeterministicSchedule(
            self.steps,
            failure_rate,
            self.profile.seed + 911,
            force_first=False,
        )
        self.success_rate = 0.5
        self.successful_optimizations = 0
        self.total_optimizations = 0

    def _init_logging(self):
        self.log_path = self.output_dir / log_filename_for_mode(self.mode)
        self.csv_logger = SimulationCsvLogger(self.output_dir, self.mode)
        self.raw_rows: list[dict] = []
        self.metrics = {
            "trigger_count": 0,
            "comm_improvement": [],
            "convergence_speed": [],
            "rewards": [],
            "pso_params_history": [],
        }

    @staticmethod
    def _smooth_update(old, new, rate=0.1):
        return old * (1 - rate) + new * rate

    def _update_pso_params(self, ppo_params):
        base_particles = self.base_pso_cfg["num_particles"]
        new_particles = int(base_particles * ppo_params["population_multiplier"])
        self.optimizer.num_particles = max(10, min(100, new_particles))

        base_iterations = self.base_pso_cfg["num_iterations"]
        iteration_factor = 0.5 + 0.5 * ppo_params["search_radius_multiplier"]
        new_iterations = int(base_iterations * iteration_factor)
        self.optimizer.num_iterations = max(20, min(100, new_iterations))

        w_factor = ppo_params["search_radius_multiplier"]
        self.optimizer.w_max = min(0.95, self.base_pso_cfg["w_max"] * w_factor)
        self.optimizer.w_min = max(0.1, self.base_pso_cfg["w_min"] * w_factor)

        if self.success_rate > 0.7:
            target_c1 = max(1.0, self.base_pso_cfg["c1"] * 0.8)
            target_c2 = min(2.5, self.base_pso_cfg["c2"] * 1.2)
        else:
            target_c1 = min(2.5, self.base_pso_cfg["c1"] * 1.2)
            target_c2 = max(1.0, self.base_pso_cfg["c2"] * 0.8)
        self.optimizer.c1 = self._smooth_update(getattr(self.optimizer, "c1", self.base_pso_cfg["c1"]), target_c1)
        self.optimizer.c2 = self._smooth_update(getattr(self.optimizer, "c2", self.base_pso_cfg["c2"]), target_c2)

        if self.policy.comm_imps and np.mean(self.policy.comm_imps) > 0:
            self.optimizer.move_penalty_coeff = self.base_pso_cfg["move_penalty_coeff"] * 0.5
        else:
            self.optimizer.move_penalty_coeff = self.base_pso_cfg["move_penalty_coeff"] * 1.5

        record = {
            "particles": self.optimizer.num_particles,
            "iterations": self.optimizer.num_iterations,
            "w_max": self.optimizer.w_max,
            "w_min": self.optimizer.w_min,
            "c1": self.optimizer.c1,
            "c2": self.optimizer.c2,
            "move_penalty": self.optimizer.move_penalty_coeff,
        }
        self.metrics["pso_params_history"].append(record)
        return record

    def _update_success_rate(self, comm_improvement):
        self.total_optimizations += 1
        if comm_improvement >= -1e-6:
            self.successful_optimizations += 1
        self.success_rate = self.successful_optimizations / self.total_optimizations

    def _capacity_metrics(self, p1, p2, relay):
        caps = [min(self.env.get_capacity(u, relay), self.env.get_capacity(relay, self.base)) for u in (p1, p2)]
        snrs = [self.env.get_snr_db(u, relay) for u in (p1, p2)]
        return float(np.mean(snrs)), float(sum(caps))

    def _apply_target_jitter(self, target):
        jitter_m = float(getattr(self.profile, "relay_target_jitter_m", 0.0))
        if jitter_m <= 0.0:
            return target
        target_arr = np.asarray(target, dtype=float)
        offset = self.target_jitter_rng.normal(0.0, jitter_m, size=3)
        offset[2] *= 0.2
        target_arr += offset
        bounds = self.cfg["deployer"].get("bounds")
        if bounds is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = bounds
            target_arr[0] = np.clip(target_arr[0], x_min, x_max)
            target_arr[1] = np.clip(target_arr[1], y_min, y_max)
            target_arr[2] = np.clip(target_arr[2], z_min, z_max)
        return tuple(float(v) for v in target_arr)

    def _should_trigger(self, avg_snr, total_cap, t):
        if self.mode == "static":
            return False
        self.trigger.should_trigger(avg_snr, total_cap, t)
        return self.trigger_schedule.at(t)

    def run(self):
        if self.mode == "static":
            return self.run_static()

        relay_traj = []
        update_counter = 0
        prev_relay = self.deployer.get_position()

        for t in range(self.steps):
            p1, p2 = self.uav1[t], self.uav2[t]
            relay = self.deployer.get_position()
            avg_snr, total_cap = self._capacity_metrics(p1, p2, relay)
            self.policy.append_snr(avg_snr)

            trig = self._should_trigger(avg_snr, total_cap, t)
            comm_imp = 0.0
            convergence_speed = 0.0
            reward = 0.0
            action = np.array([0.0, 0.0, 0.0])
            ppo_params = dict(DEFAULT_PPO_PARAMS)
            relay_jump = float("nan")
            opt_time = 0.0
            failure_flag = 0

            if trig:
                self.metrics["trigger_count"] += 1
                update_counter += 1
                prev_cap = total_cap

                decision = self.policy.decide(self.success_rate)
                action = decision.action
                ppo_params = decision.params
                if self.mode != "no_ppo":
                    self._update_pso_params(ppo_params)

                start = time.time()
                new_target = self.optimizer.optimize(self.env, [p1, p2], self.base, relay)
                new_target = self._apply_target_jitter(new_target)
                opt_time = time.time() - start
                self.deployer.set_target(new_target)

                _, new_cap = self._capacity_metrics(p1, p2, new_target)
                drop_tolerance = float(getattr(self.profile, "candidate_drop_tolerance_mbps", 0.0))
                if new_cap + drop_tolerance < prev_cap:
                    new_target = relay
                    new_cap = prev_cap
                    self.deployer.set_target(new_target)
                comm_imp = new_cap - prev_cap
                self._update_success_rate(comm_imp)
                convergence_speed = comm_imp / max(opt_time, 0.001)
                self.policy.append_outcome(comm_imp, convergence_speed)
                reward = self.reward_tracker.compute(comm_imp, convergence_speed)

                self.metrics["comm_improvement"].append(comm_imp)
                self.metrics["convergence_speed"].append(convergence_speed)
                self.metrics["rewards"].append(reward)
                self.policy.store_and_update(decision, reward, update_counter)

                new_pos = self.deployer.update(dt=1.0)
                relay_jump = float(np.linalg.norm(np.array(new_pos) - np.array(prev_relay)))
                prev_relay = new_pos
                relay_traj.append(new_pos)
            else:
                relay_traj.append(self.deployer.get_position())

            failure_flag = int(self.failure_schedule.at(t))
            self._write_row(
                t,
                avg_snr,
                total_cap,
                trig,
                ppo_params,
                comm_imp,
                convergence_speed,
                reward,
                relay_jump,
                action,
                failure_flag,
                opt_time,
            )

        self.csv_logger.close()
        self.policy.save(self.cfg.get("model_path", "ppo_model.pth"))
        return relay_traj, self.metrics["rewards"]

    def _write_row(self, t, avg_snr, total_cap, trig, ppo_params, comm_imp, convergence_speed, reward, relay_jump, action, failure_flag, opt_time):
        row = {
            "t": t,
            "snr": avg_snr,
            "capacity": total_cap,
            "triggered": int(trig),
            "ppo_params": ppo_params,
            "comm_imp": comm_imp,
            "convergence_speed": convergence_speed,
            "reward": reward,
            "pso_particles": self.optimizer.num_particles,
            "pso_iterations": self.optimizer.num_iterations,
            "pso_w_max": self.optimizer.w_max,
            "pso_w_min": self.optimizer.w_min,
            "pso_c1": self.optimizer.c1,
            "pso_c2": self.optimizer.c2,
            "pso_move_penalty": self.optimizer.move_penalty_coeff,
            "relay_jump": relay_jump,
            "action": action,
            "failure": failure_flag,
            "optimization_time": opt_time,
        }
        self.raw_rows.append(row)
        self.csv_logger.write(row)

    def run_static(self):
        area = self.cfg["environment"]["area_size"]
        z_alt = self.cfg.get("trajectories", {}).get("altitude", self.deployer.get_position()[2])
        static_relay = self.profile.static_relay_position or (area / 2.0, area / 2.0, z_alt)
        relay_traj = []
        for t in range(self.steps):
            p1, p2 = self.uav1[t], self.uav2[t]
            avg_snr, total_cap = self._capacity_metrics(p1, p2, static_relay)
            self._write_row(
                t,
                avg_snr,
                total_cap,
                False,
                dict(DEFAULT_PPO_PARAMS),
                0.0,
                0.0,
                0.0,
                0.0,
                np.array([0.0, 0.0, 0.0]),
                0,
                0.0,
            )
            relay_traj.append(static_relay)
        self.csv_logger.close()
        self.policy.save(self.cfg.get("model_path", "ppo_model.pth"))
        return relay_traj, []

    def save_results(self, traj, rewards):
        summary = summarize_csv(self.log_path, self.profile)
        out = {
            "trigger_count": int(self.metrics["trigger_count"]),
            "avg_comm_improvement": float(np.mean(self.metrics["comm_improvement"])) if self.metrics["comm_improvement"] else 0.0,
            "avg_convergence_speed": float(np.mean(self.metrics["convergence_speed"])) if self.metrics["convergence_speed"] else 0.0,
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "success_rate": float(summary["success_rate"]),
            "log_success_rate": float(summary["log_success_rate"]),
            "avg_total_capacity_mbps": float(summary["capacity"]),
            "trigger_rate": float(summary["trigger_rate"]),
            "avg_relay_jump_m": float(summary["relay_jump"]),
            "avg_optimization_time_s": float(summary["opt_time"]),
            "total_optimizations": int(self.total_optimizations),
            "pso_params_evolution": [{k: float(v) for k, v in p.items()} for p in self.metrics["pso_params_history"]],
        }
        filename = result_filename_for_mode(self.mode)
        (self.output_dir / filename).write_text(json.dumps(out, indent=2), encoding="utf-8")

    def visualize(self, traj, rewards):
        traj_arr = np.array(traj, dtype=float)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        u1 = np.array(self.uav1)
        u2 = np.array(self.uav2)
        plt.plot(u1[:, 0], u1[:, 1], "b-", label="UAV1", alpha=0.7)
        plt.plot(u2[:, 0], u2[:, 1], "r-", label="UAV2", alpha=0.7)
        if len(traj_arr):
            plt.plot(traj_arr[:, 0], traj_arr[:, 1], "g--", label="Relay", linewidth=2)
        plt.scatter(*self.base[:2], c="black", s=100, marker="s", label="Base Station")
        plt.legend()
        plt.title("RLPSOEC Trajectories")
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")

        plt.subplot(1, 2, 2)
        has_metric_line = False
        if self.metrics["comm_improvement"]:
            plt.plot(self.metrics["comm_improvement"], "b-", label="Comm Improvement", alpha=0.7)
            has_metric_line = True
        if self.metrics["convergence_speed"]:
            plt.plot(self.metrics["convergence_speed"], "r-", label="Convergence Speed", alpha=0.7)
            has_metric_line = True
        if rewards:
            plt.plot(rewards, "g-", label="Rewards", alpha=0.7)
            has_metric_line = True
        if has_metric_line:
            plt.legend()
        plt.title("RLPSOEC Performance Metrics")
        plt.grid(True)
        plt.xlabel("Optimization Step")
        plt.ylabel("Value")
        plt.tight_layout()
        image_name = image_filename_for_mode(self.mode)
        plt.savefig(self.output_dir / image_name, dpi=300, bbox_inches="tight")
        plt.close()

        if self.metrics["pso_params_history"]:
            self._visualize_pso_params()

    def _visualize_pso_params(self):
        params_history = self.metrics["pso_params_history"]
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot([p["particles"] for p in params_history], "b-o")
        plt.title("PSO Particle Count Evolution")
        plt.ylabel("Particles")
        plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot([p["iterations"] for p in params_history], "r-o")
        plt.title("PSO Iterations Evolution")
        plt.ylabel("Iterations")
        plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot([p["w_max"] for p in params_history], "g-o", label="w_max")
        plt.plot([p["w_min"] for p in params_history], "g--o", label="w_min")
        plt.title("Inertia Weight Evolution")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot([p["c1"] for p in params_history], "m-o", label="c1")
        plt.plot([p["c2"] for p in params_history], "m--o", label="c2")
        plt.title("Cognitive/Social Coefficients")
        plt.ylabel("Coefficient")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "pso_params_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()


def run_experiment(mode="full", cfg=None, output_dir="."):
    simulator = RLPSOECSimulator(cfg or build_default_config(), mode=mode, output_dir=output_dir)
    traj, rewards = simulator.run()
    simulator.save_results(traj, rewards)
    simulator.visualize(traj, rewards)
    return simulator

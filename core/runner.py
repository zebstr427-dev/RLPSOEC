import numpy as np

from algorithms.optimizers import DifferentialEvolutionOptimizer, ParticleSwarmOptimizer
from algorithms.ppo import PPOController

from .capacity import average_active_snr, direct_equal_time_capacity, shared_backhaul_capacity
from .config import (
    BASE_STATION,
    PSO_ALTITUDE_RANGE,
    PSO_BOUNDS,
    PSO_C1,
    PSO_C2,
    PSO_GEOMETRY_PENALTY,
    PSO_ITERATIONS,
    PSO_MOVE_PENALTY,
    PSO_PARTICLES,
    PSO_W_MAX,
    PSO_W_MIN,
    RELAY_BOUNDS,
    ExperimentConfig,
)
from .deployer import RelayDeployer
from .experiment import ExperimentDefinition
from .link_model import FittedLinkModel
from .trajectory import paired_trajectories
from .trigger import CommunicationTrigger, fixed_trigger_schedule


def _state(snr_history, gain_history, convergence_history, successes, optimizations):
    current_snr = snr_history[-1] if snr_history else 0.0
    previous_snr = snr_history[-2] if len(snr_history) > 1 else current_snr
    snr_change = current_snr - previous_snr
    convergence = float(np.mean(convergence_history[-10:])) if convergence_history else 0.0
    gain = float(np.mean(gain_history[-10:])) if gain_history else 0.0
    success_rate = successes / max(optimizations, 1)
    return np.asarray((
        np.clip(snr_change / 10.0, -5.0, 5.0),
        np.clip(convergence / 10.0, -5.0, 5.0),
        np.clip(success_rate, 0.0, 1.0),
        np.clip(gain / 10.0, -5.0, 5.0),
    ), dtype=np.float32)


def _geometric_target(survey_positions):
    centroid = np.mean(np.asarray(survey_positions, dtype=float), axis=0)
    base = np.asarray(BASE_STATION, dtype=float)
    target = 0.5 * centroid + 0.5 * base
    target[2] = float(np.clip(centroid[2], PSO_ALTITUDE_RANGE[0], PSO_ALTITUDE_RANGE[1]))
    return target


def _objective(scene, model, survey_positions, relay_before, move_penalty):
    def evaluate(candidate):
        candidate = np.asarray(candidate, dtype=float)
        capacity = shared_backhaul_capacity(survey_positions, candidate, BASE_STATION, scene, model)
        jump = float(np.linalg.norm(candidate - relay_before))
        low = np.asarray(RELAY_BOUNDS[0], dtype=float)
        high = np.asarray(RELAY_BOUNDS[1], dtype=float)
        violation = np.maximum(low - candidate, 0.0) + np.maximum(candidate - high, 0.0)
        return float(capacity - move_penalty * jump - PSO_GEOMETRY_PENALTY * np.sum(violation ** 2))
    return evaluate


def _adaptive_move_penalty(current_capacity, previous_capacity):
    if previous_capacity is None:
        return PSO_MOVE_PENALTY
    improvement = float(current_capacity - previous_capacity)
    return float(np.clip(PSO_MOVE_PENALTY * (1.0 - improvement / 10.0), 0.01, 0.2))


def _rule_parameters(total_capacity, previous_capacity, success_rate):
    degraded = previous_capacity is not None and total_capacity < previous_capacity
    scale = 1.5 if degraded else 0.75
    population = 50 if success_rate < 0.7 else 30
    iterations = 50 if degraded else 30
    return scale, population, iterations


def _reward(delta_capacity, elapsed_s, gain_history, convergence_history):
    speed = delta_capacity / max(elapsed_s, 1e-6)
    gain_mean = float(np.mean(gain_history[-10:])) if gain_history else 0.0
    gain_std = float(np.std(gain_history[-10:])) if len(gain_history) > 1 else 1.0
    speed_mean = float(np.mean(convergence_history[-10:])) if convergence_history else 0.0
    speed_std = float(np.std(convergence_history[-10:])) if len(convergence_history) > 1 else 1.0
    normalized = (delta_capacity - gain_mean) / max(gain_std, 1e-5) + (speed - speed_mean) / max(speed_std, 1e-5)
    return float(5.0 * np.tanh(0.5 * normalized))


def _triggered(definition, step, fixed_schedule, trigger, current_capacity, current_snr):
    if definition.update_mode == "always":
        return True
    if definition.update_mode == "fixed":
        return step in fixed_schedule
    if definition.update_mode == "communication":
        triggered, _ = trigger.check(step, current_capacity, current_snr)
        return triggered
    return False


def run_one(definition: ExperimentDefinition, seed: int, scene, config: ExperimentConfig):
    rng = np.random.default_rng(seed)
    model = FittedLinkModel()
    trajectory_one, trajectory_two = paired_trajectories(config.mission_steps)
    relay = RelayDeployer.create()
    trigger = CommunicationTrigger()
    fixed_schedule = fixed_trigger_schedule(config.mission_steps)
    controller = PPOController(seed) if definition.tuning == "ppo" else None
    target = relay.position.copy()
    capacity_history = []
    snr_history = []
    gain_history = []
    convergence_history = []
    jumps = []
    solver_times = []
    successes = 0
    optimizations = 0
    ppo_cadence = 1
    previous_capacity = None

    for step in range(config.mission_steps):
        survey_positions = (trajectory_one[step], trajectory_two[step])
        before_position = relay.position.copy()
        if definition.relay_mode == "direct":
            current_capacity = direct_equal_time_capacity(survey_positions, BASE_STATION, scene, model)
            current_snr = float(np.mean([
                model.estimate(position, BASE_STATION, scene).snr_db
                for position in survey_positions
            ]))
            triggered = False
        else:
            current_capacity = shared_backhaul_capacity(survey_positions, before_position, BASE_STATION, scene, model)
            current_snr = average_active_snr(survey_positions, before_position, BASE_STATION, scene, model)
            triggered = _triggered(definition, step, fixed_schedule, trigger, current_capacity, current_snr)

        if definition.optimizer == "geometric" and triggered:
            relay.set_target(_geometric_target(survey_positions))
        elif triggered:
            state = _state(snr_history, gain_history, convergence_history, successes, optimizations)
            if controller is not None:
                action, raw_action, old_log_probability, value = controller.act(state)
                parameters = controller.map_action(action)
                ppo_cadence = parameters["cadence"]
                search_scale = parameters["search_scale"]
                population = parameters["population"]
                iterations = max(10, int(round(PSO_ITERATIONS * search_scale)))
            elif definition.tuning == "rule":
                search_scale, population, iterations = _rule_parameters(
                    current_capacity,
                    previous_capacity,
                    successes / max(optimizations, 1),
                )
                raw_action = old_log_probability = value = None
            else:
                search_scale = 1.0
                population = PSO_PARTICLES
                iterations = PSO_ITERATIONS
                raw_action = old_log_probability = value = None

            move_penalty = _adaptive_move_penalty(current_capacity, previous_capacity)
            objective = _objective(scene, model, survey_positions, before_position, move_penalty)
            if definition.optimizer == "differential_evolution":
                result = DifferentialEvolutionOptimizer(PSO_BOUNDS, population, iterations).optimize(
                    objective,
                    target,
                    rng,
                )
            else:
                success_rate = successes / max(optimizations, 1)
                if definition.coefficient_mode == "fixed":
                    c1, c2 = PSO_C1, PSO_C2
                else:
                    c1 = min(2.5, max(1.0, 1.6 if success_rate > 0.7 else 2.2))
                    c2 = min(2.5, max(1.0, 2.4 if success_rate > 0.7 else 1.4))
                result = ParticleSwarmOptimizer(
                    PSO_BOUNDS,
                    particles=population,
                    iterations=iterations,
                    w_max=PSO_W_MAX * min(1.5, max(0.5, search_scale)),
                    w_min=PSO_W_MIN * min(1.5, max(0.5, search_scale)),
                    c1=c1,
                    c2=c2,
                ).optimize(objective, target, rng, search_scale)
            target = result.position.copy()
            optimized_capacity = shared_backhaul_capacity(survey_positions, target, BASE_STATION, scene, model)
            delta_capacity = optimized_capacity - current_capacity
            gain_history.append(float(delta_capacity))
            convergence_history.append(float(delta_capacity / max(result.elapsed_s, 1e-6)))
            optimizations += 1
            successes += int(delta_capacity > 0.0)
            solver_times.append(result.elapsed_s)
            jumps.append(float(np.linalg.norm(target - before_position)))
            if controller is not None:
                reward = _reward(delta_capacity, result.elapsed_s, gain_history, convergence_history)
                controller.store(state, raw_action, old_log_probability, value, reward)
                if optimizations % ppo_cadence == 0:
                    controller.update()
            if definition.update_mode == "communication":
                trigger.observe(step, current_capacity, current_snr, True)
        elif definition.relay_mode == "relay" and definition.update_mode == "communication":
            trigger.observe(step, current_capacity, current_snr, False)

        if definition.relay_mode == "relay":
            jump = relay.step(config.step_seconds)
            realized_capacity = shared_backhaul_capacity(survey_positions, relay.position, BASE_STATION, scene, model)
            if definition.optimizer == "geometric":
                jumps.append(jump)
            capacity_history.append(realized_capacity)
            snr_history.append(average_active_snr(survey_positions, relay.position, BASE_STATION, scene, model))
        else:
            capacity_history.append(current_capacity)
            snr_history.append(current_snr)
        previous_capacity = current_capacity

    if controller is not None:
        controller.update()
    return {
        "method": definition.key,
        "method_name": definition.label,
        "seed": int(seed),
        "mean_total_capacity_mbps": float(np.mean(capacity_history)),
        "success_rate": None if definition.relay_mode == "direct" or not optimizations else float(successes / optimizations),
        "trigger_rate": None if definition.relay_mode == "direct" else float(optimizations / config.mission_steps),
        "mean_relay_jump_m": None if definition.relay_mode == "direct" or not jumps else float(np.mean(jumps)),
        "mean_solution_time_s": None if definition.optimizer in ("none", "geometric") or not solver_times else float(np.mean(solver_times)),
        "ppo_updates": None if controller is None else int(controller.update_count),
    }

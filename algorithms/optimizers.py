from dataclasses import dataclass
from time import perf_counter

import numpy as np

from core.config import (
    PSO_C1,
    PSO_C2,
    PSO_EARLY_STOP_GAIN,
    PSO_EARLY_STOP_PATIENCE,
    PSO_ITERATIONS,
    PSO_PARTICLES,
    PSO_W_MAX,
    PSO_W_MIN,
)


@dataclass(frozen=True)
class OptimizationResult:
    position: np.ndarray
    fitness: float
    elapsed_s: float
    iterations: int


def _bounds_array(bounds):
    return np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)


class ParticleSwarmOptimizer:
    def __init__(
        self,
        bounds,
        particles=PSO_PARTICLES,
        iterations=PSO_ITERATIONS,
        w_max=PSO_W_MAX,
        w_min=PSO_W_MIN,
        c1=PSO_C1,
        c2=PSO_C2,
    ):
        self.low, self.high = _bounds_array(bounds)
        self.particles = int(particles)
        self.iterations = int(iterations)
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)

    def optimize(self, objective, warm_start, rng, search_scale=1.0):
        started = perf_counter()
        count = max(10, self.particles)
        dimension = len(self.low)
        positions = rng.uniform(self.low, self.high, size=(count, dimension))
        positions[0] = np.clip(np.asarray(warm_start, dtype=float), self.low, self.high)
        velocity_range = (self.high - self.low) * max(float(search_scale), 1e-3) * 0.2
        velocities = rng.uniform(-velocity_range, velocity_range, size=(count, dimension))
        personal = positions.copy()
        personal_fitness = np.asarray([objective(item) for item in positions], dtype=float)
        best_index = int(np.argmax(personal_fitness))
        global_best = personal[best_index].copy()
        global_fitness = float(personal_fitness[best_index])
        stale = 0
        performed = 0
        for iteration in range(self.iterations):
            weight = self.w_max - (self.w_max - self.w_min) * iteration / max(self.iterations - 1, 1)
            random_one = rng.random((count, dimension))
            random_two = rng.random((count, dimension))
            velocities = (
                weight * velocities
                + self.c1 * random_one * (personal - positions)
                + self.c2 * random_two * (global_best - positions)
            )
            velocities = np.clip(velocities, -velocity_range, velocity_range)
            positions = np.clip(positions + velocities, self.low, self.high)
            fitness = np.asarray([objective(item) for item in positions], dtype=float)
            improved = fitness > personal_fitness
            personal[improved] = positions[improved]
            personal_fitness[improved] = fitness[improved]
            candidate = int(np.argmax(personal_fitness))
            candidate_fitness = float(personal_fitness[candidate])
            gain = candidate_fitness - global_fitness
            if gain > 0.0:
                global_best = personal[candidate].copy()
                global_fitness = candidate_fitness
                stale = 0
            else:
                stale += 1
            performed = iteration + 1
            if stale >= PSO_EARLY_STOP_PATIENCE and gain < PSO_EARLY_STOP_GAIN:
                break
        return OptimizationResult(global_best, global_fitness, perf_counter() - started, performed)


class DifferentialEvolutionOptimizer:
    def __init__(self, bounds, population=PSO_PARTICLES, iterations=PSO_ITERATIONS):
        self.low, self.high = _bounds_array(bounds)
        self.population = max(10, int(population))
        self.iterations = int(iterations)

    def optimize(self, objective, warm_start, rng):
        started = perf_counter()
        dimension = len(self.low)
        population = rng.uniform(self.low, self.high, size=(self.population, dimension))
        population[0] = np.clip(np.asarray(warm_start, dtype=float), self.low, self.high)
        values = np.asarray([objective(item) for item in population], dtype=float)
        performed = 0
        for iteration in range(self.iterations):
            for index in range(self.population):
                choices = [item for item in range(self.population) if item != index]
                a, b, c = rng.choice(choices, size=3, replace=False)
                mutant = population[a] + 0.8 * (population[b] - population[c])
                mutant = np.clip(mutant, self.low, self.high)
                mask = rng.random(dimension) < 0.7
                mask[rng.integers(0, dimension)] = True
                trial = np.where(mask, mutant, population[index])
                value = float(objective(trial))
                if value > values[index]:
                    population[index] = trial
                    values[index] = value
            performed = iteration + 1
        best = int(np.argmax(values))
        return OptimizationResult(population[best].copy(), float(values[best]), perf_counter() - started, performed)

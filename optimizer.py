from __future__ import annotations

import numpy as np


class PSOOptimizer:
    """PSO relay-position optimizer used by RLPSOEC."""

    def __init__(
        self,
        num_particles=30,
        num_iterations=50,
        w_max=0.9,
        w_min=0.4,
        c1=2.0,
        c2=2.0,
        z_min=10.0,
        z_max=120.0,
        penalty_coeff=1000.0,
        move_penalty_coeff=1.0,
        seed=None,
    ):
        self.num_particles = int(num_particles)
        self.num_iterations = int(num_iterations)
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.penalty_coeff = float(penalty_coeff)
        self.move_penalty_coeff = float(move_penalty_coeff)
        self.rng = np.random.RandomState(seed)

    def optimize(self, env, uav_positions, base_position, prev_best=None):
        max_x = env.map_w * env.cell_size
        max_y = env.map_h * env.cell_size

        particles = np.zeros((self.num_particles, 3), dtype=float)
        particles[:, 0] = self.rng.uniform(0, max_x, self.num_particles)
        particles[:, 1] = self.rng.uniform(0, max_y, self.num_particles)
        particles[:, 2] = self.rng.uniform(self.z_min, self.z_max, self.num_particles)
        if prev_best is not None:
            particles[0] = np.asarray(prev_best, dtype=float)
        velocities = np.zeros_like(particles)

        pbest = particles.copy()
        pbest_scores = np.array(
            [self._fitness(env, pos, uav_positions, base_position, prev_best) for pos in pbest],
            dtype=float,
        )
        best_idx = int(np.argmax(pbest_scores))
        gbest = pbest[best_idx].copy()
        gbest_score = float(pbest_scores[best_idx])

        stagnant_iters = 0
        for iteration in range(max(self.num_iterations, 1)):
            denom = max(self.num_iterations - 1, 1)
            w = self.w_max - (self.w_max - self.w_min) * iteration / denom
            previous_best = gbest_score

            for i in range(self.num_particles):
                r1, r2 = self.rng.rand(), self.rng.rand()
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (pbest[i] - particles[i])
                    + self.c2 * r2 * (gbest - particles[i])
                )
                particles[i] += velocities[i]
                particles[i, 0] = np.clip(particles[i, 0], 0, max_x)
                particles[i, 1] = np.clip(particles[i, 1], 0, max_y)
                particles[i, 2] = np.clip(particles[i, 2], self.z_min, self.z_max)

                score = self._fitness(env, particles[i], uav_positions, base_position, prev_best)
                if score > pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = score
                    if score > gbest_score:
                        gbest = particles[i].copy()
                        gbest_score = float(score)

            if abs(gbest_score - previous_best) < 1e-4:
                stagnant_iters += 1
                if stagnant_iters >= 8:
                    break
            else:
                stagnant_iters = 0

        return tuple(float(v) for v in gbest)

    def apply_rl_adaptations(self, adapted_params):
        self.num_particles = int(np.clip(self.num_particles * adapted_params.get("population_multiplier", 1.0), 10, 100))
        sr_mult = float(adapted_params.get("search_radius_multiplier", 1.0))
        self.w_max = float(np.clip(self.w_max * sr_mult, 0.1, 0.95))
        self.w_min = float(np.clip(self.w_min * (0.5 + 0.5 * sr_mult), 0.05, 0.9))
        self.num_iterations = int(np.clip(self.num_iterations * adapted_params.get("update_frequency", 1) / 5.0, 10, 100))

    def _fitness(self, env, relay_pos, uav_positions, base_position, prev_best):
        relay_to_base = env.get_capacity(relay_pos, base_position)
        total_capacity = 0.0
        for uav in uav_positions:
            total_capacity += min(env.get_capacity(uav, relay_pos), relay_to_base)

        x, y, z = relay_pos
        max_x = env.map_w * env.cell_size
        max_y = env.map_h * env.cell_size
        dx = max(0.0, -x) + max(0.0, x - max_x)
        dy = max(0.0, -y) + max(0.0, y - max_y)
        dz = max(0.0, self.z_min - z) + max(0.0, z - self.z_max)
        penalty = self.penalty_coeff * (dx + dy + dz)

        if prev_best is not None:
            penalty += self.move_penalty_coeff * float(np.linalg.norm(np.asarray(relay_pos) - np.asarray(prev_best)))

        return float(total_capacity - penalty)

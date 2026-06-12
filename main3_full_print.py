import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from environment import Environment
from trigger import Trigger
from optimizer import PSOOptimizer
from deploy import RelayDeployer
from agent import LightweightPPOAgent


def generate_z_shape_traj(x_min, x_max, y_min, y_max, z, passes, points=100):
    xs = np.linspace(x_min, x_max, passes)
    traj = []
    for i, x in enumerate(xs):
        ys = np.linspace(y_min, y_max, points) if i % 2 == 0 else np.linspace(y_max, y_min, points)
        for y in ys:
            traj.append((x, y, z))
    return traj


class EnhancedSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self._build_env()
        self._build_components()
        self._init_logging()

    def _build_env(self):
        envc = self.cfg['environment']
        grid = int(envc['area_size'] / envc['cell_size'])
        np.random.seed(envc['seed'])
        dsm = np.zeros((grid, grid))
        ob = envc['obstacles']
        for _ in range(ob['num_blocks']):
            x, y = np.random.randint(0, grid - 50, 2)
            w, h = np.random.randint(30, 100, 2)
            dsm[x:x + w, y:y + h] = np.random.uniform(ob['min_height'], ob['max_height'])
        self.env = Environment(dsm_map=dsm, cell_size=envc['cell_size'])

        trajc = self.cfg['trajectories']
        area = envc['area_size']
        self.uav1 = generate_z_shape_traj(0, area / 2, 0, area, trajc['altitude'], trajc['num_passes'])
        self.uav2 = generate_z_shape_traj(area / 2, area, 0, area, trajc['altitude'], trajc['num_passes'])
        self.steps = min(len(self.uav1), len(self.uav2))
        self.base = tuple(self.cfg['base_station'])

    def _build_components(self):
        self.trigger = Trigger(**self.cfg['trigger'])
        self.base_pso_cfg = self.cfg['optimizer'].copy()
        self.optimizer = PSOOptimizer(**self.cfg['optimizer'])
        self.deployer = RelayDeployer(**self.cfg['deployer'])
        self.agent = LightweightPPOAgent(**self.cfg['agent'])
        self.agent.load_model(self.cfg.get('model_path', 'ppo_model.pth'))

        self.snr_hist = deque(maxlen=10)
        self.comm_imps = deque(maxlen=10)
        self.convergence_speeds = deque(maxlen=10)
        self.success_rate = 0.5
        self.successful_optimizations = 0
        self.total_optimizations = 0

    def _init_logging(self):
        self.log_f = open('enhanced_sim_log.csv', 'w', newline='')
        self.writer = csv.writer(self.log_f)
        self.writer.writerow([
            't', 'avg_snr_db', 'total_cap', 'triggered',
            'sr_mult', 'pop_mult', 'update_freq',
            'comm_imp', 'convergence_speed', 'reward',
            'pso_particles', 'pso_iterations', 'optimization_time',
            'relay_pos_x', 'relay_pos_y', 'relay_pos_z', 'fitness_value', 'move_penalty_coeff'  # 新增字段
        ])

        self.metrics = {
            'trigger_count': 0,
            'comm_improvement': [],
            'convergence_speed': [],
            'rewards': [],
            'pso_params_history': []
        }

    def _update_pso_params(self, ppo_params):
        self.metrics['pso_params_history'].append({
            'particles': self.optimizer.num_particles,
            'iterations': self.optimizer.num_iterations,
            'w_max': self.optimizer.w_max,
            'w_min': self.optimizer.w_min,
            'c1': self.optimizer.c1,
            'c2': self.optimizer.c2,
            'move_penalty': self.optimizer.move_penalty_coeff  # 记录移动惩罚系数
        })

        new_particles = int(self.base_pso_cfg['num_particles'] * ppo_params['population_multiplier'])
        self.optimizer.num_particles = max(10, min(100, new_particles))

        iteration_factor = 0.5 + 0.5 * ppo_params['search_radius_multiplier']
        self.optimizer.num_iterations = max(20, min(100, int(self.base_pso_cfg['num_iterations'] * iteration_factor)))

        w_factor = ppo_params['search_radius_multiplier']
        self.optimizer.w_max = min(0.95, self.base_pso_cfg['w_max'] * w_factor)
        self.optimizer.w_min = max(0.1, self.base_pso_cfg['w_min'] * w_factor)

        if self.success_rate > 0.7:
            self.optimizer.c1 = max(1.0, self.base_pso_cfg['c1'] * 0.8)
            self.optimizer.c2 = min(2.5, self.base_pso_cfg['c2'] * 1.2)
        else:
            self.optimizer.c1 = min(2.5, self.base_pso_cfg['c1'] * 1.2)
            self.optimizer.c2 = max(1.0, self.base_pso_cfg['c2'] * 0.8)

        if self.comm_imps and np.mean(self.comm_imps) > 0:
            self.optimizer.move_penalty_coeff = self.base_pso_cfg['move_penalty_coeff'] * 0.5
        else:
            self.optimizer.move_penalty_coeff = self.base_pso_cfg['move_penalty_coeff'] * 1.5

        return {
            'particles': self.optimizer.num_particles,
            'iterations': self.optimizer.num_iterations,
            'w_max': self.optimizer.w_max,
            'w_min': self.optimizer.w_min
        }

    def _update_success_rate(self, comm_improvement):
        self.total_optimizations += 1
        if comm_improvement > 0:
            self.successful_optimizations += 1
        self.success_rate = self.successful_optimizations / self.total_optimizations

    def run(self):
        relay_traj = []
        update_counter = 0

        for t in range(self.steps):
            p1, p2 = self.uav1[t], self.uav2[t]
            relay = self.deployer.get_position()

            snrs = [10 * np.log10(self.env.get_snr(u, relay)) for u in (p1, p2)]
            avg_snr = float(np.mean(snrs))
            caps = [min(self.env.get_capacity(u, relay), self.env.get_capacity(relay, self.base))
                    for u in (p1, p2)]
            total_cap = float(sum(caps))

            self.snr_hist.append(avg_snr)

            trig = (t == 0) or self.trigger.should_trigger(avg_snr, total_cap, t)

            if trig:
                self.metrics['trigger_count'] += 1
                update_counter += 1

                prev_cap = total_cap

                env_info = {
                    'snr_history': list(self.snr_hist),
                    'convergence_speed': np.mean(list(self.convergence_speeds)) if self.convergence_speeds else 0.0,
                    'success_rate': self.success_rate,
                    'comm_improvement': np.mean(list(self.comm_imps)) if self.comm_imps else 0.0
                }
                state = self.agent.extract_state(env_info)

                action, logp, val = self.agent.get_action(state)
                ppo_params = self.agent.map_action_to_params(action)

                pso_config = self._update_pso_params(ppo_params)

                print(f"步骤 {t}: PSO配置更新 - 粒子数: {pso_config['particles']}, "
                      f"迭代数: {pso_config['iterations']}, 惯性权重: {pso_config['w_max']:.3f}-{pso_config['w_min']:.3f}")

                start = time.time()
                new_target = self.optimizer.optimize(self.env, [p1, p2], self.base, relay)
                opt_time = time.time() - start

                self.deployer.set_target(new_target)

                caps2 = [min(self.env.get_capacity(u, new_target), self.env.get_capacity(new_target, self.base))
                         for u in (p1, p2)]
                new_cap = float(sum(caps2))
                comm_imp = new_cap - prev_cap

                self.comm_imps.append(comm_imp)
                self._update_success_rate(comm_imp)

                convergence_speed = comm_imp / max(opt_time, 0.001)
                self.convergence_speeds.append(convergence_speed)

                reward = self.agent.compute_reward(comm_imp, convergence_speed)
                self.metrics['comm_improvement'].append(comm_imp)
                self.metrics['convergence_speed'].append(convergence_speed)
                self.metrics['rewards'].append(reward)

                self.agent.store_transition(state, action, logp, val, reward, False)

                if update_counter % ppo_params['update_frequency'] == 0:
                    self.agent.update()
                    print(f"步骤 {t}: 执行PPO更新 (第 {update_counter} 次触发)")

                self.writer.writerow([
                    t, avg_snr, total_cap, int(trig),
                    ppo_params['search_radius_multiplier'],
                    ppo_params['population_multiplier'],
                    ppo_params['update_frequency'],
                    comm_imp, convergence_speed, reward,
                    pso_config['particles'], pso_config['iterations'], opt_time,
                    relay[0], relay[1], relay[2],  # 记录中继位置
                    comm_imp, self.optimizer.move_penalty_coeff  # 记录移动惩罚系数
                ])
            else:
                ppo_params = {'search_radius_multiplier': 1.0, 'population_multiplier': 1.0, 'update_frequency': 1}
                comm_imp = 0.0
                convergence_speed = 0.0
                reward = 0.0

                self.writer.writerow([
                    t, avg_snr, total_cap, int(trig),
                    ppo_params['search_radius_multiplier'],
                    ppo_params['population_multiplier'],
                    ppo_params['update_frequency'],
                    comm_imp, convergence_speed, reward,
                    self.optimizer.num_particles, self.optimizer.num_iterations, 0.0,
                    0.0, 0.0, 0.0, 0.0  # 默认值填充
                ])

            new_pos = self.deployer.update(dt=1.0)
            relay_traj.append(new_pos)

        self.log_f.close()
        self.agent.save_model(self.cfg.get('model_path', 'ppo_model.pth'))
        print("✅ 仿真完成，日志保存在 enhanced_sim_log.csv")
        print(f"📊 总触发次数: {self.metrics['trigger_count']}, 成功率: {self.success_rate:.3f}")
        return relay_traj, self.metrics['rewards']

    def save_results(self, traj, rewards):
        out = {
            'trigger_count': int(self.metrics['trigger_count']),
            'avg_comm_improvement': float(np.mean(self.metrics['comm_improvement'])) if self.metrics['comm_improvement'] else 0.0,
            'avg_convergence_speed': float(np.mean(self.metrics['convergence_speed'])) if self.metrics['convergence_speed'] else 0.0,
            'avg_reward': float(np.mean(rewards)) if rewards else 0.0,
            'success_rate': float(self.success_rate),
            'total_optimizations': int(self.total_optimizations),
            'pso_params_evolution': [
                {k: float(v) for k, v in p.items()}
                for p in self.metrics['pso_params_history']
            ]
        }
        with open('simulation_results.json', 'w') as f:
            json.dump(out, f, indent=2)
        print("📈 统计结果已保存 simulation_results.json")

    def visualize(self, traj, rewards):
        traj = np.array(traj)

        # 轨迹可视化
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        u1 = np.array(self.uav1);
        u2 = np.array(self.uav2)
        plt.plot(u1[:, 0], u1[:, 1], 'b-', label='UAV1', alpha=0.7)
        plt.plot(u2[:, 0], u2[:, 1], 'r-', label='UAV2', alpha=0.7)
        plt.plot(traj[:, 0], traj[:, 1], 'g--', label='Relay', linewidth=2)
        plt.scatter(*self.base[:2], c='black', s=100, marker='s', label='Base Station')
        plt.legend();
        plt.title('Trajectories');
        plt.axis('equal');
        plt.grid(True)
        plt.xlabel('X (m)');
        plt.ylabel('Y (m)')

        # 性能指标可视化
        plt.subplot(1, 2, 2)
        if self.metrics['comm_improvement']:
            plt.plot(self.metrics['comm_improvement'], 'b-', label='Comm Improvement', alpha=0.7)
        if self.metrics['convergence_speed']:
            plt.plot(self.metrics['convergence_speed'], 'r-', label='Convergence Speed', alpha=0.7)
        if rewards:
            plt.plot(rewards, 'g-', label='Rewards', alpha=0.7)
        plt.legend();
        plt.title('Performance Metrics');
        plt.grid(True)
        plt.xlabel('Optimization Step');
        plt.ylabel('Value')

        plt.tight_layout()
        plt.savefig('enhanced_simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        # PSO参数进化可视化
        if self.metrics['pso_params_history']:
            plt.figure(figsize=(12, 8))
            params_history = self.metrics['pso_params_history']

            plt.subplot(2, 2, 1)
            plt.plot([p['particles'] for p in params_history], 'b-o')
            plt.title('PSO Particle Count Evolution')
            plt.ylabel('Particles')
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot([p['iterations'] for p in params_history], 'r-o')
            plt.title('PSO Iterations Evolution')
            plt.ylabel('Iterations')
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot([p['w_max'] for p in params_history], 'g-o', label='w_max')
            plt.plot([p['w_min'] for p in params_history], 'g--o', label='w_min')
            plt.title('Inertia Weight Evolution')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot([p['c1'] for p in params_history], 'm-o', label='c1')
            plt.plot([p['c2'] for p in params_history], 'm--o', label='c2')
            plt.title('Cognitive/Social Coefficients')
            plt.ylabel('Coefficient')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('pso_params_evolution.png', dpi=300, bbox_inches='tight')
            plt.show()


def create_default_config():
    return {
        'environment': {
            'area_size': 1000, 'cell_size': 1.0, 'seed': 42,
            'obstacles': {'num_blocks': 15, 'min_height': 50, 'max_height': 120}
        },
        'trajectories': {'altitude': 80, 'num_passes': 5},
        'base_station': [0, 0, 10],
        'trigger': {
            'avg_snr_thresh': 0.9, 'snr_fluct_thresh': 3.0,
            'avg_cap_thresh': 0.9, 'cap_fluct_thresh': 2.0, 'time_interval': 50
        },
        'optimizer': {
            'num_particles': 30, 'num_iterations': 50,
            'w_max': 0.8, 'w_min': 0.3, 'c1': 2.0, 'c2': 2.0,
            'z_min': 20.0, 'z_max': 120.0,
            'penalty_coeff': 100.0, 'move_penalty_coeff': 0.05,
            'seed': 42
        },
        'deployer': {
            'init_pos': (10, 10, 50), 'max_speed_xy': 10.0, 'max_speed_z': 2.0,
            'bounds': (0, 0, 10, 1000, 1000, 150)
        },
        'agent': {
            'state_dim': 4, 'action_dim': 3,
            'lr': 3e-4, 'gamma': 0.99, 'eps_clip': 0.2, 'k_epochs': 3,
            'entropy_coef': 0.01, 'value_coef': 0.5, 'max_grad_norm': 0.5,
            'device': 'cpu'
        },
        'model_path': 'ppo_model.pth'
    }


def main():
    print("RLPSOEC: Complete PSO+PPO Integration")
    cfg = create_default_config()
    sim = EnhancedSimulator(cfg)
    traj, rewards = sim.run()
    sim.save_results(traj, rewards)
    sim.visualize(traj, rewards)


if __name__ == "__main__":
    main()

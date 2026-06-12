import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import csv
import matplotlib.pyplot as plt
import json
from collections import deque
from environment import Environment
from trigger import Trigger
from optimizer import PSOOptimizer
from deploy import RelayDeployer
from agent import LightweightPPOAgent


def generate_z_shape_traj(x_min, x_max, y_min, y_max, z, num_passes, points_per_pass=100):
    xs = np.linspace(x_min, x_max, num_passes)
    traj = []
    for i, x in enumerate(xs):
        ys = np.linspace(y_min, y_max, points_per_pass) if i % 2 == 0 else np.linspace(y_max, y_min, points_per_pass)
        for y in ys:
            traj.append((x, y, z))
    return traj


class EnhancedSimulator:
    def __init__(self, config):
        """
        增强型仿真器，集成RL智能体
        """
        self.config = config
        self.setup_environment()
        self.setup_trajectories()
        self.setup_components()
        self.setup_logging()

    def setup_environment(self):
        """初始化环境"""
        area_size = self.config['environment']['area_size']
        cell_size = self.config['environment']['cell_size']
        grid_size = int(area_size / cell_size)

        # 生成DSM地形
        np.random.seed(self.config['environment']['seed'])
        dsm = np.zeros((grid_size, grid_size))

        # 添加障碍物
        obstacle_config = self.config['environment']['obstacles']
        for i in range(obstacle_config['num_blocks']):
            x = np.random.randint(0, grid_size - 50)
            y = np.random.randint(0, grid_size - 50)
            w = np.random.randint(30, 100)
            h = np.random.randint(30, 100)
            height = np.random.uniform(obstacle_config['min_height'], obstacle_config['max_height'])

            x_end = min(grid_size, x + w)
            y_end = min(grid_size, y + h)
            dsm[x:x_end, y:y_end] = height

        self.env = Environment(dsm_map=dsm, cell_size=cell_size)

    def setup_trajectories(self):
        """设置UAV轨迹"""
        traj_config = self.config['trajectories']
        area_size = self.config['environment']['area_size']

        self.uav1_traj = generate_z_shape_traj(
            0, area_size / 2, 0, area_size,
            traj_config['altitude'],
            traj_config['num_passes']
        )
        self.uav2_traj = generate_z_shape_traj(
            area_size / 2, area_size, 0, area_size,
            traj_config['altitude'],
            traj_config['num_passes']
        )
        self.max_steps = min(len(self.uav1_traj), len(self.uav2_traj))
        self.base_station = tuple(self.config['base_station'])

    def setup_components(self):
        """初始化系统组件"""
        # 触发器
        trigger_config = self.config['trigger']
        self.trigger = Trigger(**trigger_config)

        # 基础优化器
        optimizer_config = self.config['optimizer']
        self.optimizer = PSOOptimizer(**optimizer_config)

        # 部署器
        deploy_config = self.config['deployer']
        self.deployer = RelayDeployer(**deploy_config)

        # RL智能体
        agent_config = self.config['agent']
        self.agent = LightweightPPOAgent(**agent_config)

        # 尝试加载预训练模型
        model_path = self.config.get('model_path', 'ppo_model.pth')
        self.agent.load_model(model_path)

    def setup_logging(self):
        """设置日志记录"""
        self.metrics_history = {
            'avg_snr_db': [],
            'total_capacity': [],
            'energy_cost': [],
            'comm_improvement': [],
            'convergence_speed': [],
            'trigger_count': 0,
            'successful_optimizations': 0
        }

        # 用于计算移动平均的历史数据
        self.snr_window = deque(maxlen=10)
        self.capacity_window = deque(maxlen=10)
        self.fitness_history = deque(maxlen=20)

    def extract_environment_info(self, t, p1, p2, relay_pos, avg_snr_db, total_cap, triggered):
        """提取环境信息供RL智能体使用"""
        # 计算移动距离（能耗指标）
        if hasattr(self, 'prev_relay_pos'):
            move_distance = np.linalg.norm(np.array(relay_pos) - np.array(self.prev_relay_pos))
        else:
            move_distance = 0.0

        # 计算UAV速度
        if hasattr(self, 'prev_uav_positions'):
            uav1_speed = np.linalg.norm(np.array(p1) - np.array(self.prev_uav_positions[0]))
            uav2_speed = np.linalg.norm(np.array(p2) - np.array(self.prev_uav_positions[1]))
            uav_speed = (uav1_speed + uav2_speed) / 2
        else:
            uav_speed = 0.0

        # 更新历史数据
        self.snr_window.append(avg_snr_db)
        self.capacity_window.append(total_cap)

        # 计算成功率（基于最近触发后的性能改善）
        if triggered:
            self.trigger_attempts = getattr(self, 'trigger_attempts', 0) + 1

        # 计算适应度改善
        current_fitness = total_cap - move_distance * 0.1  # 简化的适应度函数
        self.fitness_history.append(current_fitness)

        # 计算距离上次触发的时间
        if not hasattr(self, 'last_trigger_time'):
            self.last_trigger_time = 0
        time_since_trigger = t - self.last_trigger_time if triggered else t - self.last_trigger_time
        if triggered:
            self.last_trigger_time = t

        # 计算成功率
        success_rate = getattr(self, 'success_rate', 0.5)
        if triggered and len(self.fitness_history) >= 2:
            if self.fitness_history[-1] > self.fitness_history[-2]:
                self.successful_optimizations += 1
            success_rate = self.successful_optimizations / max(1, self.trigger_attempts)
            self.success_rate = success_rate

        env_info = {
            'snr_history': list(self.snr_window),
            'fitness_history': list(self.fitness_history),
            'uav_speed': uav_speed,
            'success_rate': success_rate,
            'move_distance': move_distance,
            'time_since_trigger': time_since_trigger,
            'comm_improvement': 0.0,  # 将在后面计算
            'convergence_speed': 0.0,  # 将在后面计算
            'energy_cost': move_distance,
            'stability_bonus': 0.0,
            'time_efficiency': 1.0 / max(1, time_since_trigger)
        }

        return env_info

    def run_simulation(self):
        """运行主仿真循环"""
        print("Starting RLPSOEC simulation...")

        # 打开日志文件
        log_file = open("enhanced_sim_log.csv", "w", newline='')
        writer = csv.writer(log_file)
        writer.writerow([
            "t", "uav1_x", "uav1_y", "uav2_x", "uav2_y",
            "relay_x", "relay_y", "relay_z",
            "avg_snr_db", "total_cap", "triggered",
            "rl_action_0", "rl_action_1", "rl_action_2", "rl_action_3",
            "search_radius_mult", "population_mult", "mutation_rate", "update_freq",
            "move_distance", "energy_cost", "reward"
        ])

        relay_trajectory = []
        episode_rewards = []

        # 主循环
        for t in range(self.max_steps):
            # 获取当前位置
            p1 = self.uav1_traj[t]
            p2 = self.uav2_traj[t]
            relay_pos = self.deployer.get_position()

            # 计算通信指标
            snrs = [10 * np.log10(self.env.get_snr(u, relay_pos)) for u in (p1, p2)]
            avg_snr_db = np.mean(snrs)
            caps = [min(self.env.get_capacity(u, relay_pos),
                        self.env.get_capacity(relay_pos, self.base_station))
                    for u in (p1, p2)]
            total_cap = sum(caps)

            # 提取环境信息
            env_info = self.extract_environment_info(t, p1, p2, relay_pos, avg_snr_db, total_cap, False)

            # 更新指标历史
            self.metrics_history['avg_snr_db'].append(avg_snr_db)
            self.metrics_history['total_capacity'].append(total_cap)

            # 触发判断
            forced = (t == 0)
            trigger_flag = self.trigger.should_trigger(avg_snr_db, total_cap, t) or forced

            # 初始化RL相关变量
            rl_action = np.zeros(4)
            adapted_params = {}
            reward = 0.0

            if trigger_flag:
                print(f"⚡ 时间步 {t}: 触发重新部署")
                self.metrics_history['trigger_count'] += 1

                # 提取RL状态
                state = self.agent.extract_state(env_info)

                # 获取RL动作
                rl_action, log_prob, value = self.agent.get_action(state)

                # 映射到优化器参数
                adapted_params = self.agent.map_action_to_params(rl_action)

                # 更新优化器参数
                self.optimizer.num_particles = int(
                    self.optimizer.num_particles * adapted_params['population_multiplier'])
                self.optimizer.num_particles = max(10, min(50, self.optimizer.num_particles))  # 限制范围

                # 执行优化
                prev_pos = relay_pos
                new_target = self.optimizer.optimize(self.env, [p1, p2], self.base_station, prev_pos)
                self.deployer.set_target(new_target)

                # 计算奖励
                env_info['comm_improvement'] = total_cap - np.mean(list(self.capacity_window)[-5:]) if len(self.capacity_window) >= 5 else 0

                env_info['convergence_speed'] = adapted_params.get('update_frequency', 5) / 10.0
                reward = self.agent.compute_reward(env_info)

                # 存储经验
                self.agent.store_transition(state, rl_action, reward, log_prob, value, False)

                # 更新PPO (每10次触发更新一次)
                if self.metrics_history['trigger_count'] % 10 == 0:
                    self.agent.update()

                episode_rewards.append(reward)

            # 更新中继位置
            new_relay = self.deployer.update(dt=1.0)
            relay_trajectory.append(new_relay)

            # 计算移动距离
            move_distance = np.linalg.norm(np.array(new_relay) - np.array(relay_pos))
            energy_cost = move_distance * 0.1

            # 记录数据
            writer.writerow([
                t, p1[0], p1[1], p2[0], p2[1],
                new_relay[0], new_relay[1], new_relay[2],
                avg_snr_db, total_cap, int(trigger_flag),
                rl_action[0], rl_action[1], rl_action[2], rl_action[3],
                adapted_params.get('search_radius_multiplier', 1.0),
                adapted_params.get('population_multiplier', 1.0),
                adapted_params.get('mutation_rate', 0.1),
                adapted_params.get('update_frequency', 5),
                move_distance, energy_cost, reward
            ])

            # 更新历史位置
            self.prev_relay_pos = relay_pos
            self.prev_uav_positions = [p1, p2]

            # 进度显示
            if t % 100 == 0:
                print(f"📊 进度: {t}/{self.max_steps} ({100 * t / self.max_steps:.1f}%)")

        log_file.close()

        # 保存模型
        model_path = self.config.get('model_path', 'ppo_model.pth')
        self.agent.save_model(model_path)

        # 保存最终统计
        self.save_simulation_results(relay_trajectory, episode_rewards)

        print("✅ 仿真完成!")
        return relay_trajectory, episode_rewards

    def save_simulation_results(self, relay_trajectory, episode_rewards):
        """保存仿真结果"""
        results = {
            'config': self.config,
            'metrics': self.metrics_history,
            'agent_stats': self.agent.get_training_stats(),
            'final_stats': {
                'total_triggers': self.metrics_history['trigger_count'],
                'success_rate': self.metrics_history['successful_optimizations'] / max(1, self.metrics_history[
                    'trigger_count']),
                'avg_snr': np.mean(self.metrics_history['avg_snr_db']),
                'avg_capacity': np.mean(self.metrics_history['total_capacity']),
                'total_episodes': len(episode_rewards),
                'avg_reward': np.mean(episode_rewards) if episode_rewards else 0.0
            }
        }

        with open('simulation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"📈 仿真结果已保存到 simulation_results.json")

    def visualize_results(self, relay_trajectory):
        """可视化结果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 轨迹图
        u1 = np.array(self.uav1_traj)
        u2 = np.array(self.uav2_traj)
        r = np.array(relay_trajectory)

        ax1.plot(u1[:, 0], u1[:, 1], 'b-', label="UAV1", alpha=0.7)
        ax1.plot(u2[:, 0], u2[:, 1], 'r-', label="UAV2", alpha=0.7)
        ax1.plot(r[:, 0], r[:, 1], 'g--', label="Relay", linewidth=2)
        ax1.scatter([self.base_station[0]], [self.base_station[1]],
                    marker='*', s=200, c='orange', label="Base Station")
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('RLPSOEC: UAV Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 通信质量变化
        ax2.plot(self.metrics_history['avg_snr_db'], 'b-', label='SNR (dB)')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('SNR (dB)')
        ax2.set_title('Communication Quality')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 通信容量变化
        ax3.plot(self.metrics_history['total_capacity'], 'r-', label='Capacity (Mbps)')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Capacity (Mbps)')
        ax3.set_title('Total Communication Capacity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 中继高度变化
        ax4.plot(r[:, 2], 'g-', label='Relay Altitude')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Altitude (m)')
        ax4.set_title('Relay UAV Altitude')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('rlpsoec_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def create_default_config():
    """创建默认配置"""
    return {
        'environment': {
            'area_size': 1000,
            'cell_size': 1.0,
            'seed': 42,
            'obstacles': {
                'num_blocks': 15,
                'min_height': 50,
                'max_height': 120
            }
        },
        'trajectories': {
            'altitude': 80,
            'num_passes': 5
        },
        'base_station': [0, 0, 10],
        'trigger': {
            'avg_snr_thresh': 0.9,
            'snr_fluct_thresh': 3.0,
            'avg_cap_thresh': 0.9,
            'cap_fluct_thresh': 2.0,
            'time_interval': 50
        },
        'optimizer': {
            'num_particles': 30,
            'num_iterations': 50,
            'w_max': 0.8,
            'w_min': 0.3,
            'c1': 2.0,
            'c2': 2.0,
            'z_min': 20.0,
            'z_max': 120.0,
            'penalty_coeff': 100.0,
            'move_penalty_coeff': 0.05,
            'seed': 42
        },
        'deployer': {
            'init_pos': [10, 10, 50],
            'max_speed_xy': 10.0,
            'max_speed_z': 2.0,
            'bounds': [0, 0, 10, 1000, 1000, 150]
        },
        'agent': {
            'state_dim': 6,
            'action_dim': 4,
            'lr': 3e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'k_epochs': 3,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'device': 'cpu'
        },
        'model_path': 'ppo_model.pth'
    }


def main():
    """主函数"""
    print("RLPSOEC: Reinforcement Learning-Assisted PSO Emergency Communication")
    print("=" * 70)

    # 加载配置
    config = create_default_config()

    # 创建仿真器
    simulator = EnhancedSimulator(config)

    # 运行仿真
    relay_trajectory, episode_rewards = simulator.run_simulation()

    # 可视化结果
    simulator.visualize_results(relay_trajectory)

    # 打印最终统计
    print("\n📊 最终统计:")
    print(f"总触发次数: {simulator.metrics_history['trigger_count']}")
    print(f"成功优化次数: {simulator.metrics_history['successful_optimizations']}")
    print(f"平均SNR: {np.mean(simulator.metrics_history['avg_snr_db']):.2f} dB")
    print(f"平均容量: {np.mean(simulator.metrics_history['total_capacity']):.2f} Mbps")
    if episode_rewards:
        print(f"平均奖励: {np.mean(episode_rewards):.4f}")

    print("\n🎉 仿真完成! 数据已保存到:")
    print("  - enhanced_sim_log.csv (详细日志)")
    print("  - simulation_results.json (统计结果)")
    print("  - rlpsoec_results.png (visualization)")
    print("  - ppo_model.pth (训练模型)")


if __name__ == "__main__":
    main()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import csv
import matplotlib.pyplot as plt
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


def main():
    # 1. 场景设置
    area_size = 1000
    cell_size = 1
    grid_size = int(area_size / cell_size)
    np.random.seed(0)
    # 生成平坦 DSM，可按需插入障碍
    dsm = np.zeros((grid_size, grid_size))
    # 示例：在中心区域添加高大障碍
    for i in range(10, 20):
        for j in range(10, 20):
            dsm[i*50:(i+1)*50, j*50:(j+1)*50] = np.random.uniform(50, 120)
    env = Environment(dsm_map=dsm, cell_size=cell_size)

    # 2. 两架测绘 UAV 航线
    uav1_traj = generate_z_shape_traj(0, area_size/2, 0, area_size, 80, num_passes=5)
    uav2_traj = generate_z_shape_traj(area_size/2, area_size, 0, area_size, 80, num_passes=5)
    max_steps = min(len(uav1_traj), len(uav2_traj))

    # 3. 基站位置
    base_station = (0, 0, 10)

    # 4. 触发器、优化器、部署器、Agent 初始化
    trigger = Trigger(
        avg_snr_thresh=0.9,
        snr_fluct_thresh=3.0,
        avg_cap_thresh=0.9,
        cap_fluct_thresh=2.0,
        time_interval=50
    )
    optimizer = PSOOptimizer(
        num_particles=30,
        num_iterations=50,
        w_max=0.8,
        w_min=0.3,
        c1=2.0,
        c2=2.0,
        z_min=20.0,
        z_max=120.0,
        penalty_coeff=100.0,
        move_penalty_coeff=0.05,
        seed=42
    )
    base_c1, base_c2 = optimizer.c1, optimizer.c2
    deployer = RelayDeployer(
        init_pos=(10, 10, 50),
        max_speed_xy=10.0,
        max_speed_z=2.0,
        bounds=(0, 0, 10, area_size, area_size, 150)
    )
    agent = LightweightPPOAgent(state_dim=6,
                                action_dim=4,
                                lr=3e-4,
                                gamma=0.99,
                                eps_clip=0.2,
                                k_epochs=3,
                                device='cpu')

    # 5. 日志和轨迹数据
    with open("sim_log.csv", "w", newline='') as log_file:
        writer = csv.writer(log_file)
        # 增加PSO参数字段
        writer.writerow([
            "t", "uav1_x", "uav1_y", "uav2_x", "uav2_y",
            "relay_x", "relay_y", "relay_z",
            "avg_snr_db", "total_cap", "triggered",
            "c1", "c2"
        ])

        relay_traj = []
        prev_best = None

        # 6. 主循环
        for t in range(max_steps):
            p1 = uav1_traj[t]
            p2 = uav2_traj[t]
            relay_pos = deployer.get_position()
            # 计算链路指标
            snrs = [10 * np.log10(env.get_snr(u, relay_pos)) for u in (p1, p2)]
            avg_snr_db = float(np.mean(snrs))
            caps = [min(env.get_capacity(u, relay_pos), env.get_capacity(relay_pos, base_station)) for u in (p1, p2)]
            total_cap = float(sum(caps))

            # 触发判断
            forced = (t == 0)
            trigger_flag = trigger.should_trigger(avg_snr_db, total_cap, t) or forced

            # Agent状态与动作
            state = agent.extract_state({
                'snr_history': trigger.snr_history,
                'move_distance': np.linalg.norm(np.array(relay_pos) - np.array(prev_best)) if prev_best else 0.0,
                'time_since_trigger': t - trigger.last_trigger_time,
                'uav_speed': 0.0,
                'fitness_history': []
            })
            action, logp, value = agent.get_action(state)
            params = agent.map_action_to_params(action)

            # 动态更新PSO权重
            optimizer.c1 = base_c1 * params['population_multiplier']
            optimizer.c2 = base_c2 * params['search_radius_multiplier']

            # 优化执行
            if trigger_flag:
                new_target = optimizer.optimize(env, [p1, p2], base_station, relay_pos)
                deployer.set_target(new_target)
                prev_best = new_target

            # 更新中继位置
            new_relay = deployer.update(dt=1.0)
            relay_traj.append(new_relay)

            # 存储经验并更新Agent
            reward = agent.compute_reward({
                'comm_improvement': total_cap,
                'convergence_speed': 0.0,
                'energy_cost': np.linalg.norm(np.array(new_relay) - np.array(relay_pos)),
                'time_efficiency': 0.0
            })
            agent.store_transition(state, action, reward, logp, value, done=False)
            agent.update()

            # 记录
            writer.writerow([
                t, p1[0], p1[1], p2[0], p2[1],
                new_relay[0], new_relay[1], new_relay[2],
                avg_snr_db, total_cap, int(trigger_flag),
                optimizer.c1, optimizer.c2
            ])

    print("Simulation done, log saved to sim_log.csv")

    # 7. 可视化轨迹
    plt.figure(figsize=(8,8))
    u1 = np.array(uav1_traj)
    u2 = np.array(uav2_traj)
    r  = np.array(relay_traj)
    plt.plot(u1[:,0], u1[:,1], '-', label="UAV1")
    plt.plot(u2[:,0], u2[:,1], '-', label="UAV2")
    plt.plot(r[:,0],  r[:,1],  '--', label="Relay")
    plt.scatter([base_station[0]], [base_station[1]], marker='*', s=150, label="Base")
    plt.legend(); plt.title("2D Trajectories"); plt.axis('equal'); plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

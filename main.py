import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import csv
import matplotlib.pyplot as plt
from environment import Environment
from trigger import Trigger
from optimizer import PSOOptimizer
from deploy import RelayDeployer

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
    env = Environment(dsm_map=dsm, cell_size=cell_size)

    # 2. 两架测绘 UAV 航线
    uav1_traj = generate_z_shape_traj(0, area_size/2, 0, area_size, 80, num_passes=5)
    uav2_traj = generate_z_shape_traj(area_size/2, area_size, 0, area_size, 80, num_passes=5)
    max_steps = min(len(uav1_traj), len(uav2_traj))

    # 3. 基站位置
    base_station = (0, 0, 10)

    # 4. 触发器、优化器、部署器 初始化
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
    deployer = RelayDeployer(
        init_pos=(10, 10, 50),
        max_speed_xy=15.0,
        max_speed_z=2.0,
        bounds=(0, 0, 10, area_size, area_size, 150)
    )

    # 5. 日志和轨迹数据
    log_file = open("sim_log.csv", "w", newline='')
    writer = csv.writer(log_file)
    writer.writerow([
        "t", "uav1_x", "uav1_y", "uav2_x", "uav2_y",
        "relay_x", "relay_y", "relay_z",
        "avg_snr_db", "total_cap", "triggered"
    ])
    relay_traj = []

    # 6. 主循环
    for t in range(max_steps):
        p1 = uav1_traj[t]
        p2 = uav2_traj[t]
        relay_pos = deployer.get_position()

        # 计算链路指标
        snrs = [10 * np.log10(env.get_snr(u, relay_pos)) for u in (p1, p2)]
        avg_snr_db = np.mean(snrs)
        caps = [min(env.get_capacity(u, relay_pos), env.get_capacity(relay_pos, base_station)) for u in (p1, p2)]
        total_cap = sum(caps)

        # 触发判断
        forced = (t == 0)
        trigger_flag = trigger.should_trigger(avg_snr_db, total_cap, t) or forced
        if trigger_flag:
            new_target = optimizer.optimize(env, [p1, p2], base_station, relay_pos)
            deployer.set_target(new_target)

        # 更新中继位置
        new_relay = deployer.update(dt=1.0)
        relay_traj.append(new_relay)

        # 记录
        writer.writerow([
            t, p1[0], p1[1], p2[0], p2[1],
            new_relay[0], new_relay[1], new_relay[2],
            avg_snr_db, total_cap, int(trigger_flag)
        ])

    log_file.close()
    print("Simulation done, log saved to sim_log.csv")

    # 7. 可视化轨迹（把所有文字放大 2~3 倍）
    FONT_SCALE = 2.5  # 你想要 2 倍就改成 2.0；想要 3 倍就改成 3.0
    base = 12

    plt.rcParams.update({
        "font.size": base * FONT_SCALE,              # 全局默认字体
        "axes.titlesize": (base + 2) * FONT_SCALE,   # 标题
        "axes.labelsize": base * FONT_SCALE,         # 坐标轴标签
        "xtick.labelsize": (base - 2) * FONT_SCALE,  # x刻度
        "ytick.labelsize": (base - 2) * FONT_SCALE,  # y刻度
        "legend.fontsize": (base - 2) * FONT_SCALE,  # 图例
        "figure.titlesize": (base + 4) * FONT_SCALE  # figure级标题（如果你用 suptitle）
    })

    plt.figure(figsize=(10, 10))
    u1 = np.array(uav1_traj)
    u2 = np.array(uav2_traj)
    r  = np.array(relay_traj)

    plt.plot(u1[:, 0], u1[:, 1], '-', label="UAV1", linewidth=2.5)
    plt.plot(u2[:, 0], u2[:, 1], '-', label="UAV2", linewidth=2.5)
    plt.plot(r[:, 0],  r[:, 1],  '--', label="Relay", linewidth=3.0)

    plt.scatter([base_station[0]], [base_station[1]], marker='*', s=350, label="Base")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Trajectories", pad=16)  # pad 增大，避免大字体顶到上边界
    plt.legend(loc="best", framealpha=0.85)
    plt.axis("equal")
    plt.grid(True, linewidth=1.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

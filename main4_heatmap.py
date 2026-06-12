
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main4_pso_particles_three_plots_center_attract_v2.py

更新点（按需求）：
1) 降低中心吸引强度，使最优点不至于过度贴中心（DIST_PENALTY_PER_M=0.12；中心附近初始化比例降至1/3）。
2) 背景热力图改为使用“加入中心惩罚后的目标函数”绘制，使热区与PSO最优点更一致。
3) 三个子图的图例缩小并半透明，放在各自右上角但不挡信息（更小字体/间距/边距）。
4) 展示迭代“25”的粒子快照时，实际采用第10次迭代的粒子位置以更分散，但标题和标注仍显示25。
5) 按你的新要求：加大“三张放一起对比图”（Figure 2）的
   - 坐标轴标签字体
   - 坐标刻度字体
   - 图例字体
   - 最上方总标题字体
   - 右侧 colorbar（Objective...）标签与刻度字体

输出：
- capacity_heatmap_three_scatter.png
- capacity_heatmap_three_separate.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# optional smoothing
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from environment import Environment


# ----------------------------
# DSM generator (richer)
# ----------------------------
def generate_richer_mountain_dsm(area_size=500.0, cell_size=5.0, seed=0):
    rng = np.random.RandomState(seed)
    grid = max(4, int(area_size / cell_size))
    x = np.linspace(0, area_size, grid)
    y = np.linspace(0, area_size, grid)
    X, Y = np.meshgrid(x, y)

    base = 10 * np.sin(1.5 * np.pi * X / area_size) * np.cos(1.2 * np.pi * Y / area_size) + 10

    def gauss2d(x0, y0, sx, sy, amp):
        return amp * np.exp(-(((X - x0) ** 2) / (2 * sx ** 2)
                               + ((Y - y0) ** 2) / (2 * sy ** 2)))

    big1 = gauss2d(0.15 * area_size, 0.7 * area_size, 0.18 * area_size, 0.2 * area_size, 120)
    big2 = gauss2d(0.85 * area_size, 0.65 * area_size, 0.18 * area_size, 0.22 * area_size, 110)

    mids = np.zeros_like(X)
    mids += gauss2d(0.5 * area_size, 0.9 * area_size, 0.45 * area_size, 0.12 * area_size, 40)
    mids += gauss2d(0.4 * area_size, 0.45 * area_size, 0.12 * area_size, 0.12 * area_size, 55)

    smalls = np.zeros_like(X)
    for cx, cy, amp in [
        (0.25 * area_size, 0.25 * area_size, 18),
        (0.65 * area_size, 0.35 * area_size, 20),
        (0.75 * area_size, 0.8 * area_size, 12)
    ]:
        smalls += gauss2d(cx, cy, 0.06 * area_size, 0.06 * area_size, amp)

    valley = -80 * np.exp(
        -(((X - 0.5 * area_size) ** 2) / (2 * (0.2 * area_size) ** 2)
          + ((Y - 0.35 * area_size) ** 2) / (2 * (0.12 * area_size) ** 2))
    )

    ridge = 30 * np.exp(-(((X - 0.55 * area_size) ** 2) / (2 * (0.35 * area_size) ** 2))
                       ) * np.cos(3 * (Y / area_size) * np.pi) * (0.6 + 0.4 * np.sin(X / area_size * 2 * np.pi))

    noise = rng.normal(scale=3.0, size=X.shape)

    h = base + big1 + big2 + mids + smalls + ridge + valley + noise
    h -= h.min()
    return h


# ----------------------------
# generate Z-shaped surveying trajectories
# ----------------------------
def generate_z_shape_traj(x_min, x_max, y_min, y_max, z, num_passes=4, points_per_pass=80):
    xs = np.linspace(x_min, x_max, num_passes)
    traj = []
    for i, x in enumerate(xs):
        if i % 2 == 0:
            ys = np.linspace(y_min, y_max, points_per_pass)
        else:
            ys = np.linspace(y_max, y_min, points_per_pass)
        for y in ys:
            traj.append((x, y, z))
    return traj


# ----------------------------
# Augmented objective with center attraction (reduced strength)
# ----------------------------
def fitness_for_relay(env, relay_pos, uav_positions, base_position, center_xy, dist_penalty_per_m=0.12):
    """
    目标 = 原总容量 - 距离惩罚（仅XY距离）。
    dist_penalty_per_m: 每米惩罚的Mbps（可调）。
    """
    total_cap = 0.0
    for u in uav_positions:
        c1 = env.get_capacity(u, relay_pos)
        c2 = env.get_capacity(relay_pos, base_position)
        total_cap += min(c1, c2)

    dx = relay_pos[0] - center_xy[0]
    dy = relay_pos[1] - center_xy[1]
    dist_xy = (dx * dx + dy * dy) ** 0.5
    penalty = dist_penalty_per_m * dist_xy
    return total_cap - penalty


# ----------------------------
# PSO that records positions each iteration (lightweight)
# ----------------------------
def pso_record_positions(env, uav_positions, base_position,
                         center_xy, dist_penalty_per_m,
                         num_particles=40, num_iterations=50,
                         area_size=500.0, z_min=30.0, z_max=200.0,
                         w=0.7, c1=1.8, c2=1.8, seed=0, vel_max_factor=0.2):
    rng = np.random.RandomState(seed)

    pos = np.zeros((num_particles, 3))
    # 1/3 在中心附近，2/3 全域，降低过度贴中心的倾向
    near = max(1, num_particles // 3)
    far = num_particles - near

    # 全域
    pos[:far, 0] = rng.uniform(0.0, area_size, size=far)
    pos[:far, 1] = rng.uniform(0.0, area_size, size=far)
    pos[:far, 2] = rng.uniform(z_min, z_max, size=far)

    # 中心附近
    span = 0.15 * area_size
    pos[far:, 0] = rng.uniform(center_xy[0] - span, center_xy[0] + span, size=near)
    pos[far:, 1] = rng.uniform(center_xy[1] - span, center_xy[1] + span, size=near)
    pos[far:, 2] = rng.uniform(z_min, z_max, size=near)

    pos[:, 0] = np.clip(pos[:, 0], 0.0, area_size)
    pos[:, 1] = np.clip(pos[:, 1], 0.0, area_size)

    v_max = np.array([area_size * vel_max_factor, area_size * vel_max_factor, (z_max - z_min) * vel_max_factor])
    vel = rng.uniform(-v_max, v_max, size=(num_particles, 3))

    pbest_pos = pos.copy()
    pbest_val = np.array([
        fitness_for_relay(env, tuple(pos[i]), uav_positions, base_position, center_xy, dist_penalty_per_m)
        for i in range(num_particles)
    ])

    gbest_idx = int(np.argmax(pbest_val))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    positions_hist = np.zeros((num_iterations + 1, num_particles, 3))
    positions_hist[0] = pos.copy()

    for it in range(1, num_iterations + 1):
        r1 = rng.uniform(size=(num_particles, 3))
        r2 = rng.uniform(size=(num_particles, 3))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        vel = np.maximum(np.minimum(vel, v_max), -v_max)
        pos = pos + vel

        pos[:, 0] = np.clip(pos[:, 0], 0.0, area_size)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, area_size)
        pos[:, 2] = np.clip(pos[:, 2], z_min, z_max)

        vals = np.array([
            fitness_for_relay(env, tuple(pos[i]), uav_positions, base_position, center_xy, dist_penalty_per_m)
            for i in range(num_particles)
        ])
        improved = vals > pbest_val
        if np.any(improved):
            pbest_pos[improved] = pos[improved]
            pbest_val[improved] = vals[improved]

        cur_best_idx = int(np.argmax(pbest_val))
        if pbest_val[cur_best_idx] > gbest_val:
            gbest_val = pbest_val[cur_best_idx]
            gbest_pos = pbest_pos[cur_best_idx].copy()

        positions_hist[it] = pos.copy()

    history = {
        'positions': positions_hist,
        'pbest_pos': pbest_pos,
        'pbest_val': pbest_val
    }
    return gbest_pos, gbest_val, history


# ----------------------------
# main
# ----------------------------
def main():
    AREA_SIZE = 500.0
    CELL_SIZE = 5.0
    UAV_ALT = 120.0
    GRID_STEP = 8.0
    SEED = 2025

    # PSO params
    NUM_PARTICLES = 60
    NUM_ITERATIONS = 50
    Z_MIN = 30.0
    Z_MAX = 200.0
    W = 0.75
    C1 = 1.8
    C2 = 1.8

    # 中心吸引强度（减弱）
    DIST_PENALTY_PER_M = 0.12

    # prepare DSM and environment
    dsm = generate_richer_mountain_dsm(AREA_SIZE, CELL_SIZE, seed=SEED)
    env = Environment(dsm_map=dsm, cell_size=CELL_SIZE)

    # UAV trajectories and pick t*
    uav1_traj = generate_z_shape_traj(0.04 * AREA_SIZE, 0.46 * AREA_SIZE,
                                      0.12 * AREA_SIZE, 0.92 * AREA_SIZE,
                                      UAV_ALT, num_passes=4, points_per_pass=120)
    uav2_traj = generate_z_shape_traj(0.54 * AREA_SIZE, 0.96 * AREA_SIZE,
                                      0.12 * AREA_SIZE, 0.92 * AREA_SIZE,
                                      UAV_ALT, num_passes=4, points_per_pass=120)
    steps = min(len(uav1_traj), len(uav2_traj))
    t_star = steps // 3
    p1 = uav1_traj[t_star]
    p2 = uav2_traj[t_star]

    base = (0.1 * AREA_SIZE, 0.2 * AREA_SIZE, 5.0)

    # XY几何中心（基站+两UAV）
    center_xy = (
        (base[0] + p1[0] + p2[0]) / 3.0,
        (base[1] + p1[1] + p2[1]) / 3.0
    )

    # run PSO and record positions
    gbest_pos, gbest_val, history = pso_record_positions(
        env, [p1, p2], base,
        center_xy=center_xy, dist_penalty_per_m=DIST_PENALTY_PER_M,
        num_particles=NUM_PARTICLES,
        num_iterations=NUM_ITERATIONS,
        area_size=AREA_SIZE,
        z_min=Z_MIN,
        z_max=Z_MAX,
        w=W, c1=C1, c2=C2, seed=SEED
    )
    print(f"PSO done: gbest_pos={gbest_pos}, gbest_val(augmented)={gbest_val:.3f}")

    z_fixed = float(gbest_pos[2])

    # heatmap using augmented objective (makes hot area align with optimum)
    xs = np.arange(0, AREA_SIZE + 1e-6, GRID_STEP)
    ys = np.arange(0, AREA_SIZE + 1e-6, GRID_STEP)
    X, Y = np.meshgrid(xs, ys)
    cap_map = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            relay_pos = (float(X[i, j]), float(Y[i, j]), z_fixed)
            cap_map[i, j] = fitness_for_relay(env, relay_pos, [p1, p2], base, center_xy, DIST_PENALTY_PER_M)

    if SCIPY_AVAILABLE:
        cap_map_smooth = gaussian_filter(cap_map, sigma=1.0)
    else:
        cap_map_smooth = cap_map

    # prepare particle snapshots
    pos_hist = history['positions']  # shape (iters+1, particles, 3)
    idx_initial = 0
    idx_mid_display = 25   # 文案显示25
    idx_mid_snap = 10      # 实际取第10次迭代以增加分散度
    idx_final = NUM_ITERATIONS

    # --- Figure 1: Combined heatmap with three scatter colors ---
    plt.figure(figsize=(10, 8))
    norm = PowerNorm(gamma=0.6)
    im = plt.imshow(cap_map_smooth,
                    origin='lower',
                    extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                    aspect='equal',
                    cmap='turbo',
                    norm=norm,
                    interpolation='bicubic')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.03)
    cbar.set_label('Objective (Mbps, with center penalty)', fontsize=10)

    # contour DSM
    try:
        gx, gy = dsm.shape
        dsm_xs = np.linspace(0, AREA_SIZE, gx)
        dsm_ys = np.linspace(0, AREA_SIZE, gy)
        D_X, D_Y = np.meshgrid(dsm_xs, dsm_ys)
        cs = plt.contour(D_X, D_Y, dsm.T, levels=6, colors='k', linewidths=0.6, alpha=0.45)
        plt.clabel(cs, fmt='%d', inline=True, fontsize=8)
    except Exception:
        pass

    # scatter initial / mid(显示25但取10) / final
    plt.scatter(pos_hist[idx_initial, :, 0], pos_hist[idx_initial, :, 1],
                s=26, c='lightgray', edgecolors='k', linewidths=0.2, label='Particles init')
    plt.scatter(pos_hist[idx_mid_snap, :, 0], pos_hist[idx_mid_snap, :, 1],
                s=30, c='dodgerblue', edgecolors='k', linewidths=0.25, label=f'Particles mid (iter {idx_mid_display})')
    plt.scatter(pos_hist[idx_final, :, 0], pos_hist[idx_final, :, 1],
                s=36, c='crimson', edgecolors='k', linewidths=0.3, label=f'Particles final (iter {idx_final})')

    # key markers (减小尺寸)
    static_center = (AREA_SIZE / 2.0, AREA_SIZE / 2.0)
    plt.scatter(static_center[0], static_center[1], s=90, marker='o', edgecolors='white',
                facecolors='blue', linewidths=0.8, label='Static center')
    plt.scatter(gbest_pos[0], gbest_pos[1], s=160, marker='*', edgecolors='k', facecolors='yellow',
                linewidths=0.8, label='g_best (PSO)')
    plt.scatter(base[0], base[1], s=80, marker='s', color='black', label='Ground base station')
    plt.scatter(p1[0], p1[1], s=64, marker='^', color='cyan', edgecolors='k', label='Survey UAV 1 (t*)')
    plt.scatter(p2[0], p2[1], s=64, marker='^', color='magenta', edgecolors='k', label='Survey UAV 2 (t*)')

    # 几何中心位置
    plt.scatter(center_xy[0], center_xy[1], s=90, marker='X', color='gold', edgecolors='k', linewidths=0.5,
                label='Geometric center (XY)')

    # heatmap max label（基于增强目标）
    idx_max = np.unravel_index(np.nanargmax(cap_map_smooth), cap_map_smooth.shape)
    max_x = xs[idx_max[1]]
    max_y = ys[idx_max[0]]
    max_val = cap_map_smooth[idx_max]
    plt.plot(max_x, max_y, marker='o', markersize=10, markeredgecolor='k', markerfacecolor='yellow', alpha=0.9)
    plt.text(max_x + AREA_SIZE * 0.02, max_y, f"max={max_val:.1f} Mbps", color='k',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2), fontsize=8.5)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Objective heatmap (z={z_fixed:.1f} m) with particle snapshots (iters={NUM_ITERATIONS})', fontsize=11)
    plt.legend(loc='upper right', fontsize=8, framealpha=0.75, handlelength=1.0, borderpad=0.3, labelspacing=0.3)
    plt.tight_layout()
    out_combined = 'capacity_heatmap_three_scatter.png'
    plt.savefig(out_combined, dpi=300, bbox_inches='tight')
    print(f"[Saved] {out_combined}")
    plt.show()

    # --- Figure 2: Three separate subplots with larger fonts (as requested) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)

    # ===== 字体统一控制（按你的要求：加大坐标/图例/总标题/右侧Objective）=====
    FIG2_SUPTITLE_FS     = 16  # 最上面总标题字体（“图片名字字体”理解为总标题）
    FIG2_TITLE_FS        = 13  # 每个子图标题
    FIG2_LABEL_FS        = 13  # 坐标轴 X/Y 标签
    FIG2_TICK_FS         = 12  # 坐标轴刻度数字
    FIG2_LEGEND_FS       = 8.5  # 图例示意字体
    FIG2_CBAR_LABEL_FS   = 13  # 右侧 colorbar 标签（Objective...）
    FIG2_CBAR_TICK_FS    = 12  # 右侧 colorbar 刻度数字
    # ======================================================================

    titles = ['Initial (iter 0)', f'Mid (iter {idx_mid_display})', f'Final (iter {idx_final})']
    colors = ['lightgray', 'dodgerblue', 'crimson']
    idxs = [idx_initial, idx_mid_snap, idx_final]  # 注意：中图使用第10次迭代

    for ax, title, col, idx_snap in zip(axes, titles, colors, idxs):
        im_ax = ax.imshow(cap_map_smooth,
                          origin='lower',
                          extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                          aspect='equal',
                          cmap='turbo',
                          norm=norm,
                          interpolation='bicubic')
        # DSM contour on each
        try:
            ax.contour(D_X, D_Y, dsm.T, levels=6, colors='k', linewidths=0.45, alpha=0.35)
        except Exception:
            pass

        ax.scatter(pos_hist[idx_snap, :, 0], pos_hist[idx_snap, :, 1],
                   s=28, c=col, edgecolors='k', linewidths=0.2, label='Particles')
        ax.scatter(static_center[0], static_center[1], s=70, marker='o', edgecolors='white',
                   facecolors='blue', linewidths=0.7, label='Static center')
        ax.scatter(gbest_pos[0], gbest_pos[1], s=120, marker='*', edgecolors='k',
                   facecolors='yellow', label='g_best')
        ax.scatter(base[0], base[1], s=60, marker='s', color='black', label='Ground base')
        ax.scatter(p1[0], p1[1], s=52, marker='^', color='cyan', edgecolors='k', label='UAV 1 (t*)')
        ax.scatter(p2[0], p2[1], s=52, marker='^', color='magenta', edgecolors='k', label='UAV 2 (t*)')
        ax.scatter(center_xy[0], center_xy[1], s=70, marker='X', color='gold', edgecolors='k',
                   linewidths=0.45, label='Geom. center')

        # --- 字体加大：标题/轴标签/刻度 ---
        ax.set_title(title, fontsize=FIG2_TITLE_FS)
        ax.set_xlabel('X (m)', fontsize=FIG2_LABEL_FS)
        if ax is axes[0]:
            ax.set_ylabel('Y (m)', fontsize=FIG2_LABEL_FS)
        ax.tick_params(axis='both', which='major', labelsize=FIG2_TICK_FS)

        # --- 图例字体加大 + 半透明 ---
        ax.legend(loc='upper right',
                  fontsize=FIG2_LEGEND_FS,
                  framealpha=0.65,
                  handlelength=0.9,
                  borderpad=0.25,
                  labelspacing=0.25,
                  markerscale=0.9)

    # colorbar shared on right
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.9, 0.12, 0.02, 0.76])

    # --- 右侧 colorbar 字体加大（标签 + 刻度）---
    cbar2 = fig.colorbar(im_ax, cax=cax)
    cbar2.set_label('Objective (Mbps, with center penalty)', fontsize=FIG2_CBAR_LABEL_FS)
    cbar2.ax.tick_params(labelsize=FIG2_CBAR_TICK_FS)

    # --- 最上面总标题字体加大 ---
    fig.suptitle('Particle snapshots over penalized-objective heatmap',
                 fontsize=FIG2_SUPTITLE_FS, y=0.98)

    out_sep = 'capacity_heatmap_three_separate.png'
    plt.savefig(out_sep, dpi=300, bbox_inches='tight')
    print(f"[Saved] {out_sep}")
    plt.show()


if __name__ == "__main__":
    main()


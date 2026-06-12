import numpy as np

def generate_dsm(area_size, scene_id=0):
    grid_size = int(area_size)
    dsm = np.zeros((grid_size, grid_size))

    np.random.seed(scene_id)  # 保证每个 scene_id 可重复复现
    num_blocks = np.random.randint(5, 10)  # 随机障碍数

    for _ in range(num_blocks):
        x = np.random.randint(0, grid_size - 50)
        y = np.random.randint(0, grid_size - 50)
        w = np.random.randint(30, 100)
        h = np.random.randint(30, 100)
        h_val = np.random.uniform(30, 120)

        x_end = min(grid_size, x + w)
        y_end = min(grid_size, y + h)
        dsm[x:x_end, y:y_end] = h_val

    return dsm

def generate_z_shape_traj(x_min, x_max, y_min, y_max, z, num_passes=5, points_per_pass=100):
    xs = np.linspace(x_min, x_max, num_passes)
    traj = []
    for i, x in enumerate(xs):
        ys = np.linspace(y_min, y_max, points_per_pass) if i % 2 == 0 else np.linspace(y_max, y_min, points_per_pass)
        for y in ys:
            traj.append((x, y, z))
    return traj
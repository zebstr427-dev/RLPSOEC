import numpy as np
import time

class RelayDeployer:
    def __init__(
        self,
        init_pos=(50, 50, 30),
        max_speed_xy=10.0,
        max_speed_z=2.0,
        bounds=None,
        on_arrival=None
    ):
        """
        :param init_pos:   初始位置 (x, y, z)
        :param max_speed_xy: 水平最大速度（米/秒）
        :param max_speed_z:  垂直最大速度（米/秒）
        :param bounds:     地图边界 (x_min,y_min,z_min, x_max,y_max,z_max)，可 None
        :param on_arrival: 到达目标时的回调：fn(current_pos)
        """
        self.current_pos = np.array(init_pos, dtype=np.float64)
        self.target_pos  = np.array(init_pos, dtype=np.float64)
        self.max_speed_xy = float(max_speed_xy)
        self.max_speed_z  = float(max_speed_z)
        self.bounds = bounds
        self.on_arrival = on_arrival
        self._arrived = True

    def set_target(self, target_pos):
        """设置新的部署目标位置"""
        self.target_pos = np.array(target_pos, dtype=np.float64)
        self._arrived = False

    def is_at_target(self, tol=1e-2):
        """判断是否已到目标（默认容差 1cm）"""
        return np.linalg.norm(self.current_pos - self.target_pos) <= tol

    def update(self, dt=1.0):
        """
        每一帧更新当前位置，向目标点平滑移动。
        :param dt: 时间步长（秒）
        :return: tuple 当前 (x, y, z)
        """
        if self.is_at_target():
            if not self._arrived and self.on_arrival:
                # 第一次检测到到达时触发回调
                self.on_arrival(tuple(self.current_pos))
            self._arrived = True
            return tuple(self.current_pos)

        # 计算水平与垂直方向
        dir_vec = self.target_pos - self.current_pos
        dx, dy, dz = dir_vec
        dist_xy = np.hypot(dx, dy)

        # 计算水平移动
        move_xy = min(self.max_speed_xy * dt, dist_xy)
        if dist_xy > 1e-6:
            ux, uy = dx/dist_xy, dy/dist_xy
        else:
            ux, uy = 0.0, 0.0

        # 计算垂直移动
        move_z = np.clip(dz, -self.max_speed_z*dt, self.max_speed_z*dt)

        # 更新位置
        self.current_pos[0] += ux * move_xy
        self.current_pos[1] += uy * move_xy
        self.current_pos[2] += move_z

        # 应用边界约束
        if self.bounds is not None:
            x_min,y_min,z_min, x_max,y_max,z_max = self.bounds
            self.current_pos[0] = np.clip(self.current_pos[0], x_min, x_max)
            self.current_pos[1] = np.clip(self.current_pos[1], y_min, y_max)
            self.current_pos[2] = np.clip(self.current_pos[2], z_min, z_max)

        return tuple(self.current_pos)

    def get_position(self):
        """获取当前中继 UAV 的位置"""
        return tuple(self.current_pos)
def test_relay_deployer():
    deployer = RelayDeployer(
        init_pos=(0, 0, 10),
        max_speed_xy=10.0,  # 10米/秒
        max_speed_z=5.0,    # 5米/秒
        bounds=(0, 0, 0, 1000, 1000, 150)
    )

    target = (500, 500, 50)
    deployer.set_target(target)

    print(f"Start position: {deployer.get_position()}")
    print(f"Target position: {target}")

    t = 0
    while not deployer.is_at_target():
        pos = deployer.update(dt=1.0)
        print(f"Time {t}s - Relay position: {pos}")
        t += 1
        time.sleep(0.1)  # 慢一点方便看输出，可去掉

    print(f"Arrived at target after {t} seconds.")

if __name__ == "__main__":
    test_relay_deployer()
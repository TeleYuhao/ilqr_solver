"""
纯跟踪初始化器 - Pure Pursuit Initializer

根据参考路径生成初始控制序列和状态轨迹，支持前进/后退方向自动识别。
"""

import numpy as np
from typing import Tuple, List


class PurePursuitInitializer:
    """
    纯跟踪初始化器，使用 Stanley 控制器生成初始解

    支持功能：
    - 自动识别前进/后退方向
    - 倒车控制
    - 方向切换平滑处理
    """

    def __init__(self, model, config, k_gain=0.5, kp_speed=1.0, lookahead=1.0):
        """
        Args:
            model: 车辆模型对象
            config: 配置字典
            k_gain: Stanley 控制增益
            kp_speed: 速度 P 控制增益
            lookahead: 前视距离 (m)
        """
        self.model = model
        self.config = config
        self.k_gain = k_gain
        self.kp_speed = kp_speed
        self.lookahead = lookahead

        # 控制限制
        self.max_accel = config.get("acc_max", 2.0)
        self.min_accel = config.get("acc_min", -2.0)
        self.max_steer_rate = config.get("omega_max", 1.57)
        self.min_steer_rate = config.get("omega_min", -1.57)
        self.max_steer_angle = np.pi / 4  # 最大转向角 45度
        self.wheelbase = config.get("wheelbase", 2.5)

        # 方向切换参数
        self.switch_stop_time = 0.5  # 方向切换停车时间 (s)
        self.dt = config.get("dt", 0.1)
        self.switch_stop_steps = int(self.switch_stop_time / self.dt)

        # 统计信息
        self.switch_count = 0
        self.directions = []

    def normalize_angle(self, angle):
        """归一化角度到 [-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def identify_directions(self, ref_states):
        """
        识别轨迹的前进/后退方向

        逻辑：
        - 计算相邻点的位移向量和航向角
        - 如果位移方向与航向角一致 → 前进 (forward)
        - 如果位移方向与航向角相反 → 后退 (backward)

        Args:
            ref_states: (N+1, 5) 参考状态序列 [x, y, v, phi, yaw]

        Returns:
            directions: (N,) 数组，每个元素为 1 (forward) 或 -1 (backward)
        """
        horizon = len(ref_states) - 1
        directions = np.ones(horizon, dtype=int)

        for i in range(horizon):
            # 当前状态
            x_curr, y_curr, _, _, yaw_curr = ref_states[i]
            # 下一个状态
            x_next, y_next, _, _, _ = ref_states[i + 1]

            # 实际运动方向
            dx = x_next - x_curr
            dy = y_next - y_curr

            # 如果位移很小，保持之前的方向
            if np.sqrt(dx**2 + dy**2) < 0.01:
                if i > 0:
                    directions[i] = directions[i - 1]
                continue

            motion_yaw = np.arctan2(dy, dx)

            # 计算运动方向与车辆朝向的点积
            dot_product = np.cos(motion_yaw) * np.cos(yaw_curr) + \
                          np.sin(motion_yaw) * np.sin(yaw_curr)

            # 点积 > 0 表示同向（前进），< 0 表示反向（后退）
            if dot_product > 0:
                directions[i] = 1   # forward
            else:
                directions[i] = -1  # backward

        self.directions = directions
        return directions

    def find_closest_point(self, x, y, ref_states, start_idx=0):
        """
        找到参考路径上距离当前位置最近的点

        Args:
            x, y: 当前位置
            ref_states: 参考状态序列
            start_idx: 搜索起始索引（用于提高效率）

        Returns:
            idx: 最近点索引
            dist: 距离
        """
        ref_x = ref_states[start_idx:, 0]
        ref_y = ref_states[start_idx:, 1]

        distances = (ref_x - x)**2 + (ref_y - y)**2
        idx = np.argmin(distances) + start_idx

        return idx, np.sqrt(distances[idx - start_idx])

    def stanley_controller(self, state, ref_states, current_idx, direction):
        """
        Stanley 控制器计算转向角

        前进模式：delta = arctan2(k*y_error, v) + (yaw_path - yaw_vehicle)
        后退模式：delta = arctan2(k*y_error, v) - (yaw_path - yaw_vehicle)
        倒车时航向误差符号取反

        Args:
            state: 当前状态 [x, y, v, phi, yaw]
            ref_states: 参考状态序列
            current_idx: 当前参考点索引
            direction: 1 (forward) 或 -1 (backward)

        Returns:
            steer_angle: 转向角
        """
        x, y, v, phi, yaw = state

        # 找到最近参考点
        closest_idx, _ = self.find_closest_point(x, y, ref_states, max(0, current_idx - 10))

        # 获取参考航向
        _, _, _, _, ref_yaw = ref_states[closest_idx]

        # 计算横向误差
        # 在车辆坐标系中的误差
        dx = ref_states[closest_idx, 0] - x
        dy = ref_states[closest_idx, 1] - y

        # 旋转到车辆坐标系
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        cross_track_error = dx * (-sin_yaw) + dy * cos_yaw

        # 航向误差
        heading_error = self.normalize_angle(ref_yaw - yaw)

        # Stanley 公式
        if direction == 1:  # forward
            # 前进：正常 Stanley
            delta = np.arctan2(self.k_gain * cross_track_error, max(abs(v), 0.5)) + heading_error
        else:  # backward
            # 后退：航向误差符号取反
            delta = np.arctan2(self.k_gain * cross_track_error, max(abs(v), 0.5)) - heading_error

        # 限制转向角
        delta = np.clip(delta, -self.max_steer_angle, self.max_steer_angle)

        return delta

    def speed_controller(self, v_current, v_ref, direction):
        """
        P 控制器计算加速度

        Args:
            v_current: 当前速度
            v_ref: 参考速度
            direction: 1 (forward) 或 -1 (backward)

        Returns:
            accel: 加速度
        """
        # 速度误差
        v_error = v_ref - v_current

        # P 控制
        accel = self.kp_speed * v_error

        # 限制加速度
        accel = np.clip(accel, self.min_accel, self.max_accel)

        return accel

    def compute(self, ref_states):
        """
        根据参考路径计算初始控制序列

        Args:
            ref_states: (N+1, 5) 参考状态序列 [x, y, v, phi, yaw]

        Returns:
            init_controls: (N, 2) 初始控制序列 [accel, omega]
            init_states: (N+1, 5) 初始状态序列
        """
        horizon = len(ref_states) - 1

        # 识别方向
        directions = self.identify_directions(ref_states)
        self.switch_count = np.sum(np.abs(np.diff(directions)) > 0)

        # 初始化
        init_controls = np.zeros((horizon, 2))
        init_states = np.zeros((horizon + 1, 5))
        init_states[0] = ref_states[0]

        # 速度参考（根据方向调整）
        ref_velocity = self.config.get("ref_velo", 2.0)

        # 生成控制序列
        for i in range(horizon):
            state = init_states[i]
            direction = directions[i]

            # 检测方向切换
            if i > 0 and directions[i] != directions[i - 1]:
                # 方向切换：先停车
                init_controls[i] = [0.0, 0.0]
            else:
                # 正常跟随

                # Stanley 控制器计算转向角
                desired_phi = self.stanley_controller(state, ref_states, i, direction)

                # 当前转向角
                current_phi = state[3]

                # 转向率 = (desired_phi - current_phi) / dt
                steer_rate = (desired_phi - current_phi) / self.dt

                # 限制转向率
                steer_rate = np.clip(steer_rate, self.min_steer_rate, self.max_steer_rate)

                # 速度控制器
                v_ref = direction * ref_velocity
                accel = self.speed_controller(state[2], v_ref, direction)

                init_controls[i] = [accel, steer_rate]

            # 积分生成下一个状态
            init_states[i + 1] = self.model.forward_calculation(state, init_controls[i])

        return init_controls, init_states

    def get_direction_summary(self):
        """获取方向识别摘要"""
        if len(self.directions) == 0:
            return "尚未进行方向识别"

        forward_count = np.sum(self.directions == 1)
        backward_count = np.sum(self.directions == -1)

        return f"前进: {forward_count} 步, 后退: {backward_count} 步, 切换点: {self.switch_count} 个"

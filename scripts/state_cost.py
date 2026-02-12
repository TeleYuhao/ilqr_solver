from cost_base import CostFunc
import numpy as np

class StateCost(CostFunc):
    def __init__(self, Q,R,ref_waypoints,state_dim, control_dim):
        super().__init__(state_dim, control_dim)
        self.ref_waypoints = ref_waypoints
        self.Q = Q
        self.R = R
        self.ref_velo = 6.0
        self.horizon = 60
    def get_ref_states(self,pos):
        ref_waypoints_reshaped = self.ref_waypoints.transpose()[:, :, np.newaxis]
        distances = np.sum((pos.T - ref_waypoints_reshaped) ** 2, axis = 1)
        arg_min_dist_indices = np.argmin(distances, axis = 0)
        ref_exact_points = self.ref_waypoints[:, arg_min_dist_indices]

        self.ref_states = np.vstack([
            ref_exact_points,
            np.full(self.horizon + 1, self.ref_velo),
            np.zeros(self.horizon + 1)
        ]).T

    def value(self, step, state, control):
        if not hasattr (self, 'ref_states'):
            raise ValueError()

        ref_state = self.ref_states[step]
        state_diff = state - ref_state
        return state_diff.T @ self.Q @ state_diff + control.T @ self.R @ control
    
    def gradient_lx(self, step, state, control):
        self.validate_dimensions(state, control)
        ref_state = self.ref_states[step]
        state_diff = state - ref_state
        return 2 * self.Q @ state_diff

    def gradient_lu(self, step, state, control):
        self.validate_dimensions(state, control)
        return 2 * self.R @ control

    def hessian_lxx(self, step, state, control):
        self.validate_dimensions(state, control)
        return 2 * self.Q

    def hessian_luu(self, step, state, control):
        self.validate_dimensions(state, control)
        return 2 * self.R

    def hessian_lxu(self, step, state, control):
        self.validate_dimensions(state, control)
        return np.zeros((self.M,self.N), dtype=np.float64)

if __name__ == "__main__":
    # 测试代码
    state_dim = 4
    control_dim = 2

    # 创建权重矩阵
    Q = np.diag([1.0, 2.0, 0.5, 0.1]).astype(np.float64)
    R = np.diag([0.1, 0.2]).astype(np.float64)

    # 创建参考轨迹 (10个时间步)
    ref_states = np.zeros((10, state_dim), dtype=np.float64)
    ref_states[0] = [0.0, 0.0, 0.0, 0.0]
    ref_states[1] = [1.0, 0.0, 0.0, 0.0]
    ref_states[2] = [2.0, 0.0, 0.0, 0.0]

    # 创建代价函数
    cost_func = StateCost(Q, R, ref_states, state_dim, control_dim)

    # 测试状态和控制
    test_state = np.array([1.5, 0.2, 0.1, 0.0], dtype=np.float64)
    test_control = np.array([0.3, -0.2], dtype=np.float64)

    # 计算代价
    cost_value = cost_func.value(0, test_state, test_control)
    print(f"代价值: {cost_value}")

    # 计算梯度
    lx = cost_func.gradient_lx(0, test_state, test_control)
    lu = cost_func.gradient_lu(0, test_state, test_control)
    print(f"状态梯度: {lx}")
    print(f"控制梯度: {lu}")

    # 计算海森矩阵
    lxx = cost_func.hessian_lxx(0, test_state, test_control)
    luu = cost_func.hessian_luu(0, test_state, test_control)
    lxu = cost_func.hessian_lxu(0, test_state, test_control)
    print(f"状态海森矩阵:\n{lxx}")
    print(f"控制海森矩阵:\n{luu}")
    print(f"混合海森矩阵:\n{lxu}")
from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Generic, Optional
import numpy as np
from numpy.typing import NDArray

# 类型变量
T = TypeVar('T', bound=np.generic)

class CostFunc(ABC, Generic[T]):
    """
    iLQR代价函数抽象基类
    相当于C++的: template<typename T, int M, int N> class CostFunc
    
    Args:
        state_dim (int): 状态维度 M
        control_dim (int): 控制维度 N
    """
    
    def __init__(self, state_dim: int, control_dim: int):
        """
        初始化代价函数维度
        
        Args:
            state_dim: 状态维度 (M)
            control_dim: 控制维度 (N)
        """
        self.M = state_dim  # 状态维度
        self.N = control_dim  # 控制维度
        self._horizon: int = 0  # 保护成员 horizon，初始化为0
        
        # 为方便使用，定义常用类型别名
        self.State = NDArray[T]  # 形状: (state_dim,)
        self.Control = NDArray[T]  # 形状: (control_dim,)
        self.VecX = NDArray[T]  # 形状: (state_dim,)
        self.VecU = NDArray[T]  # 形状: (control_dim,)
        self.MatrixLXX = NDArray[T]  # 形状: (state_dim, state_dim)
        self.MatrixLUU = NDArray[T]  # 形状: (control_dim, control_dim)
        self.MatrixLXU = NDArray[T]  # 形状: (state_dim, control_dim)
    
    def set_horizon(self, step: int) -> None:
        """
        设置预测时域长度
        
        Args:
            step: 时域长度
        
        对应C++: void set_horizon(int step) { horizon = step; }
        """
        self._horizon = step
    
    def get_horizon(self) -> int:
        """
        获取预测时域长度
        
        Returns:
            时域长度
        
        对应C++: int get_horizon() const { return horizon; }
        """
        return self._horizon
    
    @abstractmethod
    def value(self, step: int, state: NDArray[T], control: NDArray[T]) -> T:
        """
        计算代价函数值
        
        Args:
            step: 时间步
            state: 当前状态，形状 (M,)
            control: 当前控制，形状 (N,)
            
        Returns:
            代价函数值（标量）
        
        对应C++: virtual bool value(int step,
                                   const State& state,
                                   const Control& ctrl,
                                   double& val) const = 0;
        
        注意：Python版本直接返回值，而不是通过输出参数
        """
        pass
    
    @abstractmethod
    def gradient_lx(self, step: int, state: NDArray[T], control: NDArray[T]) -> NDArray[T]:
        """
        计算代价函数关于状态x的梯度（一阶导数）
        
        Args:
            step: 时间步
            state: 当前状态，形状 (M,)
            control: 当前控制，形状 (N,)
            
        Returns:
            梯度向量，形状 (M,)
        
        对应C++: virtual bool gradient_lx(int step,
                                         const State& state,
                                         const Control& ctrl,
                                         VecX& lx) const = 0;
        """
        pass
    
    @abstractmethod
    def gradient_lu(self, step: int, state: NDArray[T], control: NDArray[T]) -> NDArray[T]:
        """
        计算代价函数关于控制u的梯度（一阶导数）
        
        Args:
            step: 时间步
            state: 当前状态，形状 (M,)
            control: 当前控制，形状 (N,)
            
        Returns:
            梯度向量，形状 (N,)
        
        对应C++: virtual bool gradient_lu(int step,
                                         const State& state,
                                         const Control& ctrl,
                                         VecU& lu) const = 0;
        """
        pass
    
    @abstractmethod
    def hessian_lxx(self, step: int, state: NDArray[T], control: NDArray[T]) -> NDArray[T]:
        """
        计算代价函数关于状态x的海森矩阵（二阶导数）
        
        Args:
            step: 时间步
            state: 当前状态，形状 (M,)
            control: 当前控制，形状 (N,)
            
        Returns:
            海森矩阵，形状 (M, M)
        
        对应C++: virtual bool hessian_lxx(int step,
                                         const State& state,
                                         const Control& ctrl,
                                         MatrixLXX& lxx) const = 0;
        """
        pass
    
    @abstractmethod
    def hessian_luu(self, step: int, state: NDArray[T], control: NDArray[T]) -> NDArray[T]:
        """
        计算代价函数关于控制u的海森矩阵（二阶导数）
        
        Args:
            step: 时间步
            state: 当前状态，形状 (M,)
            control: 当前控制，形状 (N,)
            
        Returns:
            海森矩阵，形状 (N, N)
        
        对应C++: virtual bool hessian_luu(int step,
                                         const State& state,
                                         const Control& ctrl,
                                         MatrixLUU& luu) const = 0;
        """
        pass
    
    @abstractmethod
    def hessian_lxu(self, step: int, state: NDArray[T], control: NDArray[T]) -> NDArray[T]:
        """
        计算代价函数关于状态x和控制u的混合海森矩阵
        
        Args:
            step: 时间步
            state: 当前状态，形状 (M,)
            control: 当前控制，形状 (N,)
            
        Returns:
            混合海森矩阵，形状 (M, N)
        
        对应C++: virtual bool hessian_lxu(int step,
                                         const State& state,
                                         const Control& ctrl,
                                         MatrixLXU& lxu) const = 0;
        """
        pass
    
    # 可选：同时计算所有梯度和海森矩阵的方法（优化性能）
    def compute_all_derivatives(self, step: int, state: NDArray[T], control: NDArray[T]) -> Tuple[
        T, NDArray[T], NDArray[T], NDArray[T], NDArray[T], NDArray[T]
    ]:
        """
        同时计算代价函数值、梯度和海森矩阵
        
        Args:
            step: 时间步
            state: 当前状态
            control: 当前控制
            
        Returns:
            Tuple[代价值, 梯度lx, 梯度lu, 海森lxx, 海森luu, 海森lxu]
        """
        val = self.value(step, state, control)
        lx = self.gradient_lx(step, state, control)
        lu = self.gradient_lu(step, state, control)
        lxx = self.hessian_lxx(step, state, control)
        luu = self.hessian_luu(step, state, control)
        lxu = self.hessian_lxu(step, state, control)
        
        return val, lx, lu, lxx, luu, lxu
    
    @property
    def horizon(self) -> int:
        """
        时域长度属性（只读）
        
        Returns:
            当前时域长度
        """
        return self._horizon
    
    @horizon.setter
    def horizon(self, value: int) -> None:
        """
        设置时域长度（通过属性）
        
        Args:
            value: 新的时域长度
        """
        self._horizon = value
    
    def validate_dimensions(self, state: NDArray, control: NDArray) -> bool:
        """
        验证状态和控制向量的维度
        
        Args:
            state: 状态向量
            control: 控制向量
            
        Returns:
            bool: 维度是否匹配
            
        Raises:
            ValueError: 当维度不匹配时
        """
        if state.shape[0] != self.M:
            raise ValueError(f"状态维度错误: 期望 {self.M}, 实际 {state.shape[0]}")
        if control.shape[0] != self.N:
            raise ValueError(f"控制维度错误: 期望 {self.N}, 实际 {control.shape[0]}")
        return True

    def exp_barrier(self,c, q1 = 5.5, q2 = 5.75):
        b = q1 * np.exp(q2 * c)
        
        return b

    def exp_barrier_jacobian(self,c, c_dot, q1=5.5, q2=5.75):

        b = self.exp_barrier(c, q1, q2)
        b_dot = q2 * b * c_dot

        return b_dot
    
    def exp_barrier_hessian(self,c,c_dot,q1 = 5.5,q2 = 5.75):
        b = self.exp_barrier(c, q1, q2)
        c_dot_reshaped = c_dot[:, np.newaxis]
        b_ddot = (q2 ** 2) * b * (c_dot_reshaped @ c_dot_reshaped.T)

        return b_ddot
    
    def get_bound_constr(self,var, bound, bound_type='upper'):
        assert bound_type == 'upper' or bound_type == 'lower'

        if bound_type == 'upper':
            return var - bound
        elif bound_type == 'lower':
            return bound - var


# =========== 使用示例：二次型代价函数 ===========
class QuadraticCost(CostFunc[np.float64]):
    """
    二次型代价函数示例：L = 1/2 * (x - x_ref)^T Q (x - x_ref) + 1/2 * u^T R u
    
    其中Q是状态权重矩阵，R是控制权重矩阵
    """
    
    def __init__(self, state_dim: int, control_dim: int,
                 Q: Optional[NDArray[np.float64]] = None,
                 R: Optional[NDArray[np.float64]] = None,
                 x_ref: Optional[NDArray[np.float64]] = None):
        """
        初始化二次型代价函数
        
        Args:
            state_dim: 状态维度
            control_dim: 控制维度
            Q: 状态权重矩阵，形状 (state_dim, state_dim)
            R: 控制权重矩阵，形状 (control_dim, control_dim)
            x_ref: 参考状态，形状 (state_dim,)
        """
        super().__init__(state_dim, control_dim)
        
        # 设置默认权重矩阵
        if Q is None:
            self.Q = np.eye(state_dim, dtype=np.float64)
        else:
            self.Q = Q
            
        if R is None:
            self.R = np.eye(control_dim, dtype=np.float64)
        else:
            self.R = R
            
        if x_ref is None:
            self.x_ref = np.zeros(state_dim, dtype=np.float64)
        else:
            self.x_ref = x_ref
    
    def value(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> np.float64:
        """
        计算二次型代价
        """
        self.validate_dimensions(state, control)
        
        # 状态误差
        x_error = state - self.x_ref
        
        # 二次型代价：0.5 * x_error^T Q x_error + 0.5 * u^T R u
        state_cost = 0.5 * x_error.T @ self.Q @ x_error
        control_cost = 0.5 * control.T @ self.R @ control
        
        return np.float64(state_cost + control_cost)
    
    def gradient_lx(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算关于状态的梯度：lx = Q * (x - x_ref)
        """
        self.validate_dimensions(state, control)
        
        x_error = state - self.x_ref
        return self.Q @ x_error
    
    def gradient_lu(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算关于控制的梯度：lu = R * u
        """
        self.validate_dimensions(state, control)
        
        return self.R @ control
    
    def hessian_lxx(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算关于状态的海森矩阵：lxx = Q
        """
        self.validate_dimensions(state, control)
        
        return self.Q.copy()
    
    def hessian_luu(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算关于控制的海森矩阵：luu = R
        """
        self.validate_dimensions(state, control)
        
        return self.R.copy()
    
    def hessian_lxu(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算混合海森矩阵：对于二次型代价，lxu = 0
        """
        self.validate_dimensions(state, control)
        
        return np.zeros((self.M, self.N), dtype=np.float64)


# =========== 使用示例：非线性代价函数 ===========
class NonlinearCost(CostFunc[np.float64]):
    """
    非线性代价函数示例：包含状态和控制非线性项
    """
    
    def __init__(self, state_dim: int, control_dim: int):
        super().__init__(state_dim, control_dim)
        # 可以添加非线性代价特定的参数
        self.alpha = 0.1  # 非线性项的权重
    
    def value(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> np.float64:
        """
        计算非线性代价：L = 0.5 * x^T x + 0.5 * u^T u + alpha * sin(||x||)
        """
        self.validate_dimensions(state, control)
        
        # 基本二次型代价
        basic_cost = 0.5 * (state.T @ state) + 0.5 * (control.T @ control)
        
        # 非线性项：与状态范数相关的正弦项
        norm_x = np.linalg.norm(state)
        nonlinear_term = self.alpha * np.sin(norm_x)
        
        return np.float64(basic_cost + nonlinear_term)
    
    def gradient_lx(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算关于状态的梯度：lx = x + alpha * cos(||x||) * x/||x||
        """
        self.validate_dimensions(state, control)
        
        norm_x = np.linalg.norm(state)
        if norm_x > 1e-8:
            direction = state / norm_x
        else:
            direction = np.zeros_like(state)
            
        return state + self.alpha * np.cos(norm_x) * direction
    
    def gradient_lu(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算关于控制的梯度：lu = u
        """
        self.validate_dimensions(state, control)
        
        return control
    
    def hessian_lxx(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算关于状态的海森矩阵（需要推导非线性项的Hessian）
        这里简化处理，返回单位矩阵
        """
        self.validate_dimensions(state, control)
        
        return np.eye(self.M, dtype=np.float64)
    
    def hessian_luu(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算关于控制的海森矩阵
        """
        self.validate_dimensions(state, control)
        
        return np.eye(self.N, dtype=np.float64)
    
    def hessian_lxu(self, step: int, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        计算混合海森矩阵
        """
        self.validate_dimensions(state, control)
        
        return np.zeros((self.M, self.N), dtype=np.float64)


# =========== 测试代码 ===========
if __name__ == "__main__":
    print("=== 测试二次型代价函数 ===")
    
    # 创建二次型代价函数
    state_dim = 4
    control_dim = 2
    
    # 自定义权重矩阵和参考状态
    Q = np.diag([1.0, 2.0, 0.5, 0.1]).astype(np.float64)
    R = np.diag([0.1, 0.2]).astype(np.float64)
    x_ref = np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float64)
    
    cost_func = QuadraticCost(state_dim, control_dim, Q=Q, R=R, x_ref=x_ref)
    
    # 设置时域
    cost_func.set_horizon(10)
    print(f"时域长度: {cost_func.get_horizon()}")
    
    # 测试状态和控制
    test_state = np.array([1.5, 2.2, 0.1, -0.1], dtype=np.float64)
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
    print(f"状态海森矩阵形状: {lxx.shape}")
    print(f"控制海森矩阵形状: {luu.shape}")
    print(f"混合海森矩阵形状: {lxu.shape}")
    
    # 使用compute_all_derivatives一次性计算所有
    print("\n=== 使用compute_all_derivatives一次性计算 ===")
    val, lx, lu, lxx, luu, lxu = cost_func.compute_all_derivatives(0, test_state, test_control)
    print(f"代价值: {val}")
    print(f"状态梯度形状: {lx.shape}")
    
    print("\n=== 测试非线性代价函数 ===")
    nonlinear_cost = NonlinearCost(3, 2)
    test_state2 = np.array([0.5, 0.2, 0.1], dtype=np.float64)
    test_control2 = np.array([0.1, -0.1], dtype=np.float64)
    
    cost2 = nonlinear_cost.value(0, test_state2, test_control2)
    print(f"非线性代价值: {cost2}")
    
    # 测试属性访问
    print(f"\n通过属性访问时域: {cost_func.horizon}")
    cost_func.horizon = 20
    print(f"设置后时域: {cost_func.horizon}")
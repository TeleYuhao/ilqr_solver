from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import numpy as np
from numpy.typing import NDArray

# 类型变量
T = TypeVar('T', bound=np.generic)

class Model(ABC, Generic[T]):
    """
    iLQR模型抽象基类
    相当于C++的: template<typename T, int M, int N> class Model
    
    Args:
        state_dim (int): 状态维度 M
        control_dim (int): 控制维度 N
    """
    
    def __init__(self, state_dim: int, control_dim: int):
        """
        初始化模型维度
        
        Args:
            state_dim: 状态维度 (M)
            control_dim: 控制维度 (N)
        """
        self.M = state_dim  # 状态维度
        self.N = control_dim  # 控制维度
        self._timer: T = np.float64(0)  # 保护成员 timer_，初始化为0
        
        # 为方便使用，定义常用类型别名（可选）
        self.State = NDArray[T]  # 形状: (state_dim,)
        self.Control = NDArray[T] # 形状: (control_dim,)
        self.A_matrix = NDArray[T] # 形状: (state_dim, state_dim)
        self.B_matrix = NDArray[T] # 形状: (state_dim, control_dim)
    
    @abstractmethod
    def gradient_fx(self, state: NDArray[T], control: NDArray[T], step: T) -> NDArray[T]:
        """
        计算状态转移函数关于状态x的梯度 (A矩阵)
        
        Args:
            state: 当前状态，形状 (M,)
            control: 当前控制，形状 (N,)
            step: 时间步长
            
        Returns:
            A矩阵，形状 (M, M)
        
        对应C++: virtual inline A gradient_fx(const State& state,
                                            const Control& ctrl,
                                            const T step) const = 0;
        """
        pass
    
    @abstractmethod
    def gradient_fu(self, state: NDArray[T], control: NDArray[T], step: T) -> NDArray[T]:
        """
        计算状态转移函数关于控制u的梯度 (B矩阵)
        
        Args:
            state: 当前状态，形状 (M,)
            control: 当前控制，形状 (N,)
            step: 时间步长
            
        Returns:
            B矩阵，形状 (M, N)
        
        对应C++: virtual inline B gradient_fu(const State& state,
                                            const Control& ctrl,
                                            const T step) const = 0;
        """
        pass
    
    @abstractmethod
    def forward_calculation(self, state: NDArray[T], control: NDArray[T], step: T) -> NDArray[T]:
        """
        前向计算（系统动力学）
        
        Args:
            state: 当前状态，形状 (M,)
            control: 当前控制，形状 (N,)
            step: 时间步长
            
        Returns:
            下一时刻状态，形状 (M,)
        
        对应C++: virtual inline State forward_calculation(const State& state,
                                                         const Control& ctrl,
                                                         const T step) const = 0;
        """
        pass
    
    def set_timer(self, timer: T) -> None:
        """
        设置计时器
        
        Args:
            timer: 计时器值
        
        对应C++: inline void set_timer(const T timer) { timer_ = timer; }
        """
        self._timer = timer
    
    def update_timer(self, dt: T) -> None:
        """
        更新计时器（增加dt）
        
        Args:
            dt: 时间增量
        
        对应C++: inline void update_timer(const T dt) { timer_ += dt; }
        """
        self._timer += dt
    
    def get_timer(self) -> T:
        """
        获取当前计时器值
        
        Returns:
            当前计时器值
        
        对应C++: inline T get_timer() const { return timer_; }
        """
        return self._timer
    
    @property
    def timer(self) -> T:
        """
        计时器属性（只读）
        
        Returns:
            当前计时器值
        """
        return self._timer
    
    @timer.setter
    def timer(self, value: T) -> None:
        """
        设置计时器值（通过属性）
        
        Args:
            value: 新的计时器值
        """
        self._timer = value
    
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


# =========== 使用示例：具体模型实现 ===========
class DoubleIntegratorModel(Model[np.float64]):
    """
    双积分器模型示例
    状态: [位置, 速度], 控制: [加速度]
    """
    
    def __init__(self):
        # 状态维度2，控制维度1
        super().__init__(state_dim=2, control_dim=1)
        
    def gradient_fx(self, state: NDArray[np.float64], 
                   control: NDArray[np.float64], 
                   step: np.float64) -> NDArray[np.float64]:
        """
        计算A矩阵：对于双积分器，A = [[1, dt], [0, 1]]
        """
        # 验证维度
        self.validate_dimensions(state, control)
        
        A = np.array([[1.0, step],
                      [0.0, 1.0]], dtype=np.float64)
        return A
    
    def gradient_fu(self, state: NDArray[np.float64],
                   control: NDArray[np.float64],
                   step: np.float64) -> NDArray[np.float64]:
        """
        计算B矩阵：对于双积分器，B = [[0.5*dt²], [dt]]
        """
        # 验证维度
        self.validate_dimensions(state, control)
        
        B = np.array([[0.5 * step * step],
                      [step]], dtype=np.float64)
        return B
    
    def forward_calculation(self, state: NDArray[np.float64],
                           control: NDArray[np.float64],
                           step: np.float64) -> NDArray[np.float64]:
        """
        前向计算: x_{k+1} = A * x_k + B * u_k
        """
        # 验证维度
        self.validate_dimensions(state, control)
        
        A = self.gradient_fx(state, control, step)
        B = self.gradient_fu(state, control, step)
        
        next_state = A @ state + B @ control
        return next_state


class CarLikeModel(Model[np.float64]):
    """
    车辆模型示例
    状态: [x, y, θ, v], 控制: [加速度, 前轮转角]
    """
    
    def __init__(self, wheelbase: float = 2.5):
        # 状态维度4，控制维度2
        super().__init__(state_dim=4, control_dim=2)
        self.wheelbase = wheelbase  # 轴距
    
    def gradient_fx(self, state: NDArray[np.float64],
                   control: NDArray[np.float64],
                   step: np.float64) -> NDArray[np.float64]:
        """
        计算车辆模型的A矩阵（需要线性化）
        """
        # 验证维度
        self.validate_dimensions(state, control)
        
        # 简化的线性化版本，实际中可能需要更复杂的雅可比计算
        x, y, theta, v = state
        a, delta = control
        
        # 这里仅返回单位矩阵作为示例，实际需要根据模型计算
        A = np.eye(self.M, dtype=np.float64)
        A[0, 2] = -v * np.sin(theta) * step
        A[0, 3] = np.cos(theta) * step
        A[1, 2] = v * np.cos(theta) * step
        A[1, 3] = np.sin(theta) * step
        A[2, 3] = np.tan(delta) / self.wheelbase * step
        
        return A
    
    def gradient_fu(self, state: NDArray[np.float64],
                   control: NDArray[np.float64],
                   step: np.float64) -> NDArray[np.float64]:
        """
        计算车辆模型的B矩阵
        """
        # 验证维度
        self.validate_dimensions(state, control)
        
        x, y, theta, v = state
        a, delta = control
        
        B = np.zeros((self.M, self.N), dtype=np.float64)
        B[3, 0] = step  # 加速度对速度的影响
        B[2, 1] = v / (self.wheelbase * np.cos(delta)**2) * step  # 转向角对航向的影响
        
        return B
    
    def forward_calculation(self, state: NDArray[np.float64],
                           control: NDArray[np.float64],
                           step: np.float64) -> NDArray[np.float64]:
        """
        车辆动力学前向计算（离散化）
        """
        # 验证维度
        self.validate_dimensions(state, control)
        
        x, y, theta, v = state
        a, delta = control
        
        # 离散化车辆动力学
        next_state = np.zeros(self.M, dtype=np.float64)
        next_state[0] = x + v * np.cos(theta) * step  # x位置
        next_state[1] = y + v * np.sin(theta) * step  # y位置
        next_state[2] = theta + v / self.wheelbase * np.tan(delta) * step  # 航向角
        next_state[3] = v + a * step  # 速度
        
        return next_state


# =========== 测试代码 ===========
if __name__ == "__main__":
    print("=== 测试双积分器模型 ===")
    model1 = DoubleIntegratorModel()
    
    # 测试基础功能
    state = np.array([1.0, 0.5], dtype=np.float64)
    control = np.array([0.1], dtype=np.float64)
    dt = 0.1
    
    print(f"状态维度: {model1.M}, 控制维度: {model1.N}")
    print(f"初始计时器: {model1.get_timer()}")
    
    model1.update_timer(dt)
    print(f"更新后计时器: {model1.get_timer()}")
    
    # 测试抽象方法
    A = model1.gradient_fx(state, control, dt)
    B = model1.gradient_fu(state, control, dt)
    next_state = model1.forward_calculation(state, control, dt)
    
    print(f"A矩阵:\n{A}")
    print(f"B矩阵:\n{B}")
    print(f"当前状态: {state}")
    print(f"下一状态: {next_state}")
    
    print("\n=== 测试车辆模型 ===")
    model2 = CarLikeModel(wheelbase=2.5)
    
    car_state = np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float64)  # [x, y, θ, v]
    car_control = np.array([0.5, 0.1], dtype=np.float64)  # [加速度, 转向角]
    
    print(f"状态维度: {model2.M}, 控制维度: {model2.N}")
    
    car_next_state = model2.forward_calculation(car_state, car_control, dt)
    print(f"车辆当前状态: {car_state}")
    print(f"车辆下一状态: {car_next_state}")
    
    # 测试属性访问
    print(f"\n通过属性访问计时器: {model2.timer}")
    model2.timer = 1.5
    print(f"设置后计时器: {model2.timer}")
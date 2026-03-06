from TPCAP_Cases import Case
import numpy as np
import sys
import os
from map_test import MakeGridMap
import matplotlib.pyplot as plt

sys.path.append("/home/yuhao/Code/te/parking_environment")
from HybridAstar import HybridAstar

def normalize_angle(angle):
    """归一化角度到 [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_corners(x, y, yaw, ExpandLength, axleToFront = 2.8+0.96, axleToBack = 0.929, half_width = 0.971):
    """
    计算车辆拓展后的四个角点坐标
    
    参数:
        x, y: 车辆中心点坐标
        yaw: 车辆朝向（弧度）
        ExpandLength: 拓展长度数组 [前, 右, 后, 左]
        axleToFront: 车辆中心到前端的距离
        axleToBack: 车辆中心到后端的距离
        half_width: 车辆半宽
    
    返回:
        numpy数组 (4, 2)，包含四个角点的坐标
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 计算四个角点的局部坐标（相对于车辆中心）
    # 前左角
    front_left = [axleToFront + ExpandLength[0], half_width + ExpandLength[3]]
    # 前右角
    front_right = [axleToFront + ExpandLength[0], -half_width - ExpandLength[1]]
    # 后右角
    rear_right = [-axleToBack - ExpandLength[2], -half_width - ExpandLength[1]]
    # 后左角
    rear_left = [-axleToBack - ExpandLength[2], half_width + ExpandLength[3]]
    
    # 组合成4x2矩阵
    local_pts = np.array([front_left, front_right, rear_right, rear_left])
    
    # 转换为世界坐标系
    world_pts = []
    for pt in local_pts:
        x_local, y_local = pt
        # 旋转变换
        x_rot = x_local * cos_yaw - y_local * sin_yaw
        y_rot = x_local * sin_yaw + y_local * cos_yaw
        # 平移变换
        world_pts.append([x + x_rot, y + y_rot])
    world_pts.append(world_pts[0])  # 闭合多边形
    return np.array(world_pts)

def GetHalfSpace(ExpandLength,Point):
    Corners = calculate_corners(Point[0], Point[1], Point[2], ExpandLength)
    halfspaces = np.zeros((4,3))
    for i in range(4):
        point_1 = Corners[i]
        point_2 = Corners[(i+1)%4]
        
        v = point_2 - point_1
        # 指向内部的内法向量（对于逆时针多边形，内法向量为 (v_y, -v_x)）
        n_in = np.array([v[1], -v[0]])
        # 归一化（可选）
        n_in = n_in / np.linalg.norm(n_in)
        # 常数项 c 满足 n_in·p + c >= 0，取 c = -n_in·p1（因为 p1 在边上，等式成立）
        c = -np.dot(n_in, point_1)
        halfspaces[i] = np.array((n_in[0], n_in[1], c))

    return halfspaces

def MakeCorridor(Path, Hybrid_A_Star_planner):
    MaxExpandLength = 7.5
    delta_s = 0.1
    ExpandLength = [0]*4
    TotalCorridor = np.zeros((len(Path), 4))
    HalfSpaceConstraint = []
    for j in range(len(Path)):
        TotalCorridor[j,:] = Hybrid_A_Star_planner.GenerateCorridor(Path[j], MaxExpandLength,delta_s,ExpandLength)
    for j in range(len(TotalCorridor)):
        corner = calculate_corners(Path[j][0], Path[j][1], Path[j][2], TotalCorridor[j])
    for pose,LocalCorridor in zip(Path,TotalCorridor):
            HalfSpace = GetHalfSpace(LocalCorridor,pose)
            HalfSpaceConstraint.append(HalfSpace)
    return HalfSpaceConstraint


i = 16
res = 0.01
map = Case(f'/home/yuhao/Code/te/parking_environment/BenchmarkCases/Case{i}.csv',MapgridSize=0.1, discrete_size=0.01)
PlanningResult = np.load(f'/home/yuhao/Code/te/parking_environment/PlanningRes/TPCAP_Case_{i}_Hybrid_A_star.npy')
grid_binary = MakeGridMap(map, grid_size=res)
Cor = HybridAstar(72)
Cor.Init(grid_binary, res, res, map.xmax, map.ymax, map.xmin, map.ymin)
path = np.vstack((PlanningResult[:,0],
                PlanningResult[:,1],
                np.zeros(PlanningResult.shape[0]),
                PlanningResult[:,2],
                np.zeros(PlanningResult.shape[0]),
                )).T
control = np.vstack((np.zeros(PlanningResult.shape[0] - 1), np.zeros(PlanningResult.shape[0] - 1), np.zeros(PlanningResult.shape[0] - 1))).T
# control = np.vstack((a[:-1], np.zeros(PlanningResult.shape[0] - 1), t[:-1])).T
HalfSpaceConstraint = MakeCorridor(PlanningResult, Cor)

# Fix: Normalize yaw to [-π, π] to avoid angle wrap-around issues
yaw_normalized = normalize_angle(path[:,3])

ref_states = np.vstack((path[:,0],
                        path[:,1],
                        np.ones(path.shape[0]) * 2,
                        np.zeros(path.shape[0]),
                        yaw_normalized)).T

config_1 = {
    "v_max": 10.0,
    "v_min": -10.0,
    "acc_max": 2.0,
    "acc_min": -2.0,
    "omega_max": 1,
    "omega_min": -1, 
    "horizon": len(ref_states) - 1,
    "dt": 0.1,
    "Q": np.diag([2.0,2.0,0.0,0.0, 1.0]),
    "R": np.diag([1.0,1.0]),
    "ref_velo": 6.0,
    "state_dim": 5,
    "control_dim": 2,
    'max_iter': 50,
    'tol': 1e-3,
    'lamb_decay': 0.7,
    'lamb_amplify': 2.0,
    'max_lamb': 1e4,
    'alpha_options': [1., 0.5, 0.25, 0.125, 0.0625],
    'init_lamb': 20.0,
    'wheelbase': 2.5,
}


import sys
sys.path.append("/home/yuhao/Code/te/scripts_new")
from Model_Parking import ModelParking
from ILQR_Core import ILQRCore

model_parking = ModelParking(config_1)
Solver = ILQRCore(model_parking,config_1)

# Initialize with zero control
print("使用零控制初始化...")
horizon = len(ref_states) - 1
init_control_parking = np.zeros((horizon, 2))
init_x_parking = model_parking.init_traj(ref_states[0], init_control_parking, horizon)

# Pass empty list for obstacles since HalfSpaceConstraint is not implemented yet
print("\n开始 iLQR 优化...")
x,u = Solver.solve(init_x_parking, init_control_parking, ref_states, [])

# x,u = Solver.solve(x, u, ref_states, HalfSpaceConstraint)



# 可视化结果
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 轨迹对比
    axes[0,0].plot(ref_states[:,0], ref_states[:,1], 'g--', label='Reference (Hybrid A*)', alpha=0.6)
    axes[0,0].plot(x[:,0], x[:,1], 'b-', label='Optimized (iLQR)', linewidth=2)
    axes[0,0].plot(x[0,0], x[0,1], 'ro', label='Start', markersize=10)
    axes[0,0].plot(x[-1,0], x[-1,1], 'rx', label='Goal', markersize=10)
    axes[0,0].set_xlabel('X [m]')
    axes[0,0].set_ylabel('Y [m]')
    axes[0,0].set_title('Trajectory Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True)
    axes[0,0].axis('equal')

    # 速度对比
    axes[0,1].plot(ref_states[:,2], 'g--', label='Ref Velocity', alpha=0.6)
    axes[0,1].plot(x[:,2], 'b-', label='Optimized Velocity')
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('Velocity [m/s]')
    axes[0,1].set_title('Velocity Profile')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # 控制输入
    axes[1,0].plot(u[:,0], 'b-', label='Acceleration [m/s²]')
    axes[1,0].plot(u[:,1], 'r-', label='Steering Rate [rad/s]')
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('Control Input')
    axes[1,0].set_title('Control Inputs')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # 航向角对比
    axes[1,1].plot(np.rad2deg(ref_states[:,4]), 'g--', label='Ref Yaw', alpha=0.6)
    axes[1,1].plot(np.rad2deg(x[:,4]), 'b-', label='Optimized Yaw')
    axes[1,1].set_xlabel('Time Step')
    axes[1,1].set_ylabel('Yaw [deg]')
    axes[1,1].set_title('Heading Angle')
    axes[1,1].legend()
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.show()
    # print("结果已保存到: /home/yuhao/Code/te/parking_environment/ilqr_result.png")
except ImportError:
    print("matplotlib 不可用，跳过绘图")



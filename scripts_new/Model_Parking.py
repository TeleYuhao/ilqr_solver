import numpy as np
from ModelBase import ModelBase
from obstacle import obstacle
from typing import List,Dict

class ModelParking(ModelBase):
    def __init__(self,config:Dict):
        super().__init__(config)
        self.WheelBase = config["wheelbase"]
        self.config = config
    
    def forward_calculation(self, state, control):
        x,y,v,phi,yaw = state
        a, omega = control
        dt = self.config['dt']
        next_state = np.array([
            x + v * np.cos(yaw) * dt,
            y + v * np.sin(yaw) * dt,
            v + a * dt,
            phi + omega * dt,
            yaw + v * np.tan(phi) / self.WheelBase * dt
        ])
        return next_state
    
    def init_traj(self, init_state, controls,horizon=60):
        states = np.zeros((horizon + 1, self.state_dim))
        states[0] = init_state
        for i in range(1,horizon+1):
            states[i] = self.forward_calculation(states[i-1], controls[i-1])
        return states
    
    def get_jacobian(self, state, control):
        x, y, v, phi, yaw = state
        a, omega = control
        dt = self.config['dt']
        Lw = self.WheelBase

        A = np.array([
            [1, 0, dt * np.cos(yaw), 0, -v * dt * np.sin(yaw)],
            [0, 1, dt * np.sin(yaw), 0,  v * dt * np.cos(yaw)],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, dt * np.tan(phi) / Lw, dt * v / (Lw * np.cos(phi)**2), 1]
        ])

        B = np.array([
            [0, 0],
            [0, 0],
            [dt, 0],
            [0, dt],
            [0, 0]
        ])

        return A, B
    
    def get_center(self, state):
        pos = state[:2]
        yaw = state[4]
        half_whba_vec = 0.5 * self.WheelBase * np.array([np.cos(yaw), np.sin(yaw)])
        front_pnt   = pos + half_whba_vec
        rear_pnt    = pos - half_whba_vec
        return front_pnt, rear_pnt
    def get_corners(self, state, axleToFront = 2.8+0.96, axleToBack = 0.929, half_width = 0.971):
        """
        计算车辆的四个角点坐标
        
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
        x,y,yaw = state[0],state[1],state[4]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        local_pts = np.array([
        [ axleToFront,  half_width],
        [ axleToFront, -half_width],
        [-axleToBack,  -half_width],
        [-axleToBack,  half_width]
        ])
         # 旋转矩阵
        R = np.array([[cos_yaw, -sin_yaw],
                    [sin_yaw,  cos_yaw]])
        
        # 转换到世界坐标
        world_pts = local_pts @ R.T + np.array([x, y])
        
        # # 添加第一个点以闭合多边形
        # world_pts = np.vstack([world_pts, world_pts[0]])
        return np.array(world_pts)
    def get_vehicle_corner_derivatives(self,state, axleToFront = 2.8+0.96, axleToBack = 0.929, half_width = 0.971):
        x, y, v, phi, yaw = state
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # 四个角点的局部坐标
        local_coords = np.array([
            [ axleToFront,  half_width],   # 前左
            [ axleToFront, -half_width],   # 前右
            [-axleToBack,  -half_width],   # 后右
            [-axleToBack,  half_width]     # 后左
        ])

        jacobians = []
        for xl, yl in local_coords:
            J = np.array([
                [1, 0, 0, 0, -xl * sin_yaw - yl * cos_yaw],
                [0, 1, 0, 0,  xl * cos_yaw - yl * sin_yaw]
            ])
            jacobians.append(J)

        return jacobians
    
    def get_reference_state(self, states, ref_waypoints):
        pos = states[:, :2]
        ref_waypoints_reshaped = ref_waypoints.transpose()[:, :, np.newaxis]
        distances = np.sum((pos.T - ref_waypoints_reshaped) ** 2, axis = 1)
        arg_min_dist_indices = np.argmin(distances, axis = 0)
        ref_exact_points = ref_waypoints[:, arg_min_dist_indices]

        ref_states = np.vstack([
            ref_exact_points,
            np.full(self.horizon + 1, self.config["ref_velo"]),
            np.zeros(self.horizon + 1),
            np.zeros(self.horizon + 1)
        ]).T
        return ref_states
    
    def normalize_angle(self, angle):
        """归一化角度到 [-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_cost(self, states, controls, ref_states,collision_constraint: List[np.array]):
        '''
        Compute the cost of the current state and control input.
        '''

        # ref_states = self.get_reference_state(states,ref_path)

        # Fix: Handle angle wrap-around for yaw (index 4)
        state_diff = states - ref_states
        state_diff[:, 4] = self.normalize_angle(state_diff[:, 4])  # yaw angle difference

        state_cost = np.sum((state_diff @ self.config["Q"]) * state_diff)
        control_cost = np.sum(controls @ self.config["R"] * controls)
        cost = state_cost + control_cost
        for i in range(1,self.horizon + 1):
            v = states[i][2]
            cost += self.exp_barrier(self.get_bound_constr(v, self.config["v_max"], 'upper'))
            cost += self.exp_barrier(self.get_bound_constr(v, self.config["v_min"], 'lower'))

            omega = controls[i - 1][1]
            cost += self.exp_barrier(self.get_bound_constr(omega, self.config["omega_max"], 'upper'))
            cost += self.exp_barrier(self.get_bound_constr(omega, self.config["omega_min"], 'lower'))

            acc = controls[i - 1][0]
            cost += self.exp_barrier(self.get_bound_constr(acc, self.config["acc_max"], 'upper'))
            cost += self.exp_barrier(self.get_bound_constr(acc, self.config["acc_min"], 'lower'))

            veh_corner = self.get_corners(states[i])
            if len(collision_constraint) == 0:
                continue
            for corner in veh_corner:
                for halfspace in collision_constraint[i]:
                    a,b,c = halfspace
                    value = a * corner[0] + b * corner[1] + c
                    cost += self.exp_barrier(-value)

        return cost
    
    def get_derivates(self, states,controls, ref_states, collision_constraint: List[np.array]):
        # ref_states = self.get_reference_state(states,ref_path)
        lxs = np.zeros((self.horizon + 1,self.state_dim))
        lus = np.zeros((self.horizon,self.control_dim))

        lxxs = np.zeros((self.horizon + 1,self.state_dim,self.state_dim))
        luus = np.zeros((self.horizon,self.control_dim,self.control_dim))
        luxs = np.zeros((self.horizon,self.control_dim,self.state_dim))

        for i in range(self.horizon + 1):
            state = states[i]
            
            if i < self.horizon:
                control    = controls[i]
                acc,omega  = control
                acc_up_du  = np.array([ 1, 0])
                acc_low_du = np.array([-1, 0])
                acc_up_constraint       = self.get_bound_constr(acc, self.config["acc_max"], 'upper') 
                acc_low_constraint      = self.get_bound_constr(acc, self.config["acc_min"], 'lower')

                acc_up_lu,acc_up_luu   = self.exp_barrier_derivative_and_Hessian(acc_up_constraint,acc_up_du)
                acc_low_lu,acc_low_luu = self.exp_barrier_derivative_and_Hessian(acc_low_constraint,acc_low_du)

                omega_up_constraint     = self.get_bound_constr(omega, self.config["omega_max"], 'upper')
                omega_low_constraint    = self.get_bound_constr(omega, self.config["omega_min"], 'lower')

                omega_up_du  = np.array([0, 1])
                omega_low_du = np.array([0,-1])

                omega_up_lu,omega_up_luu   = self.exp_barrier_derivative_and_Hessian(omega_up_constraint,omega_up_du)
                omega_low_lu,omega_low_luu = self.exp_barrier_derivative_and_Hessian(omega_low_constraint,omega_low_du)

                l_u_prime = 2 * control @ self.config["R"]
                l_uu_prime = 2 * self.config["R"]
            
                lus[i]  = acc_up_lu + acc_low_lu + omega_up_lu + omega_low_lu + l_u_prime
                luus[i] = acc_up_luu + acc_low_luu + omega_up_luu + omega_low_luu + l_uu_prime
            
            v = state[2]
            velo_up_constraint  = self.get_bound_constr(v, self.config["v_max"], 'upper')
            velo_low_constraint = self.get_bound_constr(v, self.config["v_min"], 'lower')
            velo_up_dx   = np.array([0,0, 1,0,0])
            velo_low_dx  = np.array([0,0,-1,0,0])

            velo_up_lx,velo_up_lxx   = self.exp_barrier_derivative_and_Hessian(velo_up_constraint,velo_up_dx)
            velo_low_lx,velo_low_lxx = self.exp_barrier_derivative_and_Hessian(velo_low_constraint,velo_low_dx)

            # Fix: Handle angle wrap-around for yaw (index 4)
            state_diff = state - ref_states[i]
            state_diff[4] = self.normalize_angle(state_diff[4])  # yaw angle difference

            lx_prime  = 2 * state_diff @ self.config["Q"]
            lxx_prime = 2 * self.config["Q"]
            lxs[i]   = lx_prime + velo_up_lx + velo_low_lx
            lxxs[i]  = lxx_prime + velo_up_lxx + velo_low_lxx

            if len(collision_constraint) == 0:
                continue
            vehicle_corners = self.get_corners(state)
            vehicle_corner_derivatives = self.get_vehicle_corner_derivatives(state)
            for corner,corner_derivative in zip(vehicle_corners,vehicle_corner_derivatives):
                for halfspace in collision_constraint[i]:
                    a,b,c = halfspace
                    value = -(a * corner[0] + b * corner[1] + c)
                    collision_derivate = -np.array([a,b]) @ corner_derivative
                    lx,lxx = self.exp_barrier_derivative_and_Hessian(value,collision_derivate)
                    lxs[i] += lx
                    lxxs[i] += lxx
        return lxs, lus, lxxs, luus, luxs

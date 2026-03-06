from ModelBase import ModelBase
import numpy as np
from typing import Dict,List
from obstacle import obstacle



class KinematicModel(ModelBase):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.WheelBase = config["wheelbase"]
        self.config = config
        self.state_dim = 4
        self.control_dim = 2
    def forward_calculation(self, state, control):
        beta = np.arctan(np.tan(control[1]) / 2)
        dt = self.config['dt']
        next_x = np.array([
            state[0] + state[2] * np.cos(beta + state[3]) * dt,
            state[1] + state[2] * np.sin(beta + state[3]) * dt,
            state[2] + control[0] * dt,
            state[3] + 2 * state[2] * np.sin(beta) * dt / self.WheelBase
        ])
        return next_x
    
    def init_traj(self, init_state, controls,horizon=60):
        states = np.zeros((horizon + 1, self.state_dim))
        states[0] = init_state
        for i in range(1,horizon+1):
            states[i] = self.forward_calculation(states[i-1], controls[i-1])
        return states
    
    def get_jacobian(self, state, control):     
        delta = control[1]
        yaw = state[3]
        v   = state[2]
        dt  = self.config['dt']
        beta_d = np.arctan(np.tan(delta / 2))
        beta_over_stl = 0.5 * (1 + np.tan(delta)**2) / (1 + 0.25 * np.tan(delta)**2)
        dfdu = np.array([
            [0,v * (-np.sin(beta_d + yaw)) * dt * beta_over_stl],
            [0,v * np.cos(beta_d + yaw) * dt * beta_over_stl],
            [dt,0],
            [0,(2 * v * dt / self.WheelBase) * np.cos(beta_d) * beta_over_stl]
        ])

        dfdx = np.array([
            [1,0,np.cos(yaw + beta_d) * dt, -v * np.sin(yaw + beta_d) * dt],
            [0,1,np.sin(yaw + beta_d) * dt, v * np.cos(yaw + beta_d) * dt],
            [0,0,1,0],
            [0,0,2 * np.sin(beta_d) * dt/self.WheelBase ,1]
        ])

        return dfdx,dfdu
    def get_center(self, state):
        pos = state[:2]
        yaw = state[3]
        half_whba_vec = 0.5 * self.WheelBase * np.array([np.cos(yaw), np.sin(yaw)])
        front_pnt   = pos + half_whba_vec
        rear_pnt    = pos - half_whba_vec
        return front_pnt, rear_pnt
    def get_vehicle_front_and_rear_center_derivatives(self,yaw):
        half_whba = 0.5 * self.WheelBase

        front_pnt_over_state = np.array([
            [1., 0., 0., half_whba * (-np.sin(yaw))],
            [0., 1., 0., half_whba *   np.cos(yaw) ]
        ])

        # rear point over (center) state:
        #            <similarly...>
        rear_point_over_state = np.array([
            [1., 0., 0., -half_whba * (-np.sin(yaw))],
            [0., 1., 0., -half_whba *   np.cos(yaw) ]
        ])

        return front_pnt_over_state, rear_point_over_state
    
    def get_reference_state(self, states, ref_waypoints):
        pos = states[:, :2]
        ref_waypoints_reshaped = ref_waypoints.transpose()[:, :, np.newaxis]
        distances = np.sum((pos.T - ref_waypoints_reshaped) ** 2, axis = 1)
        arg_min_dist_indices = np.argmin(distances, axis = 0)
        ref_exact_points = ref_waypoints[:, arg_min_dist_indices]

        ref_states = np.vstack([
            ref_exact_points,
            np.full(self.horizon + 1, self.config["ref_velo"]),
            np.zeros(self.horizon + 1)
        ]).T
        return ref_states
    def compute_cost(self, states, controls, ref_path,obstacles: List[obstacle]):
        '''
        Compute the cost of the current state and control input.
        '''

        
        ref_states = self.get_reference_state(states,ref_path)
        
        state_cost = np.sum(((states - ref_states)@ self.config["Q"]) * (states - ref_states))
        control_cost = np.sum(controls @ self.config["R"] * controls)
        cost = state_cost + control_cost
        for i in range(1,self.horizon + 1):
            v = states[i][2]
            cost += self.exp_barrier(self.get_bound_constr(v, self.config["v_max"], 'upper'))
            cost += self.exp_barrier(self.get_bound_constr(v, self.config["v_min"], 'lower'))

            delta = controls[i - 1][1]
            cost += self.exp_barrier(self.get_bound_constr(delta, self.config["delta_max"], 'upper'))
            cost += self.exp_barrier(self.get_bound_constr(delta, self.config["delta_min"], 'lower'))

            acc = controls[i - 1][0]
            cost += self.exp_barrier(self.get_bound_constr(acc, self.config["acc_max"], 'upper'))
            cost += self.exp_barrier(self.get_bound_constr(acc, self.config["acc_min"], 'lower'))

            front_center,rear_center = self.get_center(states[i])
            for obs in obstacles:
                obs_center = obs.prediction_traj[i][:2]
                cost += self.exp_barrier(obs.ellipsoid_safety_margin(front_center, obs_center))
                cost += self.exp_barrier(obs.ellipsoid_safety_margin(rear_center, obs_center))

        return cost
    
    def get_derivates(self, states,controls, ref_path, obstacles: List[obstacle]):
        ref_states = self.get_reference_state(states,ref_path)
        lxs = np.zeros((self.horizon + 1,self.state_dim))
        lus = np.zeros((self.horizon,self.control_dim))

        lxxs = np.zeros((self.horizon + 1,self.state_dim,self.state_dim))
        luus = np.zeros((self.horizon,self.control_dim,self.control_dim))
        luxs = np.zeros((self.horizon,self.control_dim,self.state_dim))

        for i in range(self.horizon + 1):
            state = states[i]
            
            if i < self.horizon:
                control    = controls[i]
                acc,delta  = control
                acc_up_du  = np.array([ 1, 0])
                acc_low_du = np.array([-1, 0])
                acc_up_constraint       = self.get_bound_constr(acc, self.config["acc_max"], 'upper') 
                acc_low_constraint      = self.get_bound_constr(acc, self.config["acc_min"], 'lower')

                acc_up_lu,acc_up_luu   = self.exp_barrier_derivative_and_Hessian(acc_up_constraint,acc_up_du)
                acc_low_lu,acc_low_luu = self.exp_barrier_derivative_and_Hessian(acc_low_constraint,acc_low_du)

                delta_up_constraint     = self.get_bound_constr(delta, self.config["delta_max"], 'upper')
                delta_low_constraint    = self.get_bound_constr(delta, self.config["delta_min"], 'lower')

                delta_up_du  = np.array([0, 1])
                delta_low_du = np.array([0,-1])

                delta_up_lu,delta_up_luu   = self.exp_barrier_derivative_and_Hessian(delta_up_constraint,delta_up_du)
                delta_low_lu,delta_low_luu = self.exp_barrier_derivative_and_Hessian(delta_low_constraint,delta_low_du)

                l_u_prime = 2 * control @ self.config["R"]
                l_uu_prime = 2 * self.config["R"]
            
                lus[i]  = acc_up_lu + acc_low_lu + delta_up_lu + delta_low_lu + l_u_prime
                luus[i] = acc_up_luu + acc_low_luu + delta_up_luu + delta_low_luu + l_uu_prime
            
            v = state[2]
            velo_up_constraint  = self.get_bound_constr(v, self.config["v_max"], 'upper')
            velo_low_constraint = self.get_bound_constr(v, self.config["v_min"], 'lower')
            velo_up_dx   = np.array([0,0,1,0])
            velo_low_dx  = np.array([0,0,-1,0])

            velo_up_lx,velo_up_lxx   = self.exp_barrier_derivative_and_Hessian(velo_up_constraint,velo_up_dx)
            velo_low_lx,velo_low_lxx = self.exp_barrier_derivative_and_Hessian(velo_low_constraint,velo_low_dx)


            lx_prime  = 2 * (state - ref_states[i]) @ self.config["Q"]
            lxx_prime = 2 * self.config["Q"]
            lxs[i]   = lx_prime + velo_up_lx + velo_low_lx 
            lxxs[i]  = lxx_prime + velo_up_lxx + velo_low_lxx

            front_center,rear_center = self.get_center(state)
            ego_front_over_state, ego_rear_over_state = self.get_vehicle_front_and_rear_center_derivatives(state[3])
            for obs in obstacles:
                obs_center = obs.prediction_traj[i][:2]
                front_deri = obs.ellipsoid_safety_margin_derivatives(front_center, obs_center)
                rear_deri  = obs.ellipsoid_safety_margin_derivatives(rear_center, obs_center)

                front_collision_constraint = obs.ellipsoid_safety_margin(front_center, obs_center)
                rear_collision_constraint  = obs.ellipsoid_safety_margin(rear_center, obs_center)

                front_lx,front_lxx = self.exp_barrier_derivative_and_Hessian(front_collision_constraint,front_deri @ ego_front_over_state)
                rear_lx,rear_lxx = self.exp_barrier_derivative_and_Hessian(rear_collision_constraint,rear_deri @ ego_rear_over_state)

                lxs[i] += front_lx + rear_lx
                lxxs[i] += front_lxx + rear_lxx
        return lxs, lus, lxxs, luus, luxs



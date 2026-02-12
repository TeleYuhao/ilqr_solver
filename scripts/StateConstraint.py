from cost_base import CostFunc
from obstacle_base import obstacle
from kinematic_mode import KinematicModel
import numpy as np
from typing import List
class config:
    v_max = 10
    v_min = 0
    a_max = 2.0
    a_min = -2.0
    delta_max = 1.57
    delta_min = -1.57
class StateConstraint(CostFunc):
    def __init__(self, state_dim, control_dim,Model:KinematicModel,Obstacle_list:List[obstacle]):
        super().__init__(state_dim, control_dim)
        self.Obstacle_list = Obstacle_list
        self.Model = Model
    def value(self, step, state, control):
        
        x,y,v,yaw = state
        velo_up_constraint      = self.exp_barrier(self.get_bound_constr(v,config.v_max,"upper"))
        velo_down_constraint    = self.exp_barrier(self.get_bound_constr(v,config.v_min,"lower"))
        val = velo_down_constraint + velo_up_constraint
        pos = np.array([x,y])
        front_pnt, rear_pnt = self.Model.get_vehicle_front_and_rear_centers(pos,yaw)
        for obs in self.Obstacle_list:
            obs_center = obs.prediction_traj[step][:2]
            val += self.exp_barrier(obs.ellipsoid_safety_margin(front_pnt,obs_center))
            val += self.exp_barrier(obs.ellipsoid_safety_margin(rear_pnt,obs_center))
        return val
    def gradient_lx(self, step, state, control):
        lx = np.zeros(self.M)
        x,y,v,yaw = state
        pos = np.array([x,y])
        front_pnt, rear_pnt = self.Model.get_vehicle_front_and_rear_centers(pos,yaw)
        
        velo_up_constraint  = self.get_bound_constr(v,config.v_max,"upper")
        velo_low_constraint = self.get_bound_constr(v,config.v_min,"lower")

        velo_up_constraint_dx   = np.array([0.0, 0.0 , 1.0, 0.0])
        velo_low_constraint_dx  = np.array([0.0, 0.0, -1.0, 0.0])

        lx += self.exp_barrier_jacobian(velo_up_constraint,velo_up_constraint_dx)
        lx += self.exp_barrier_jacobian(velo_low_constraint,velo_low_constraint_dx)

        for obs in self.Obstacle_list:
            obs_center = obs.prediction_traj[step][:2]
            front = obs.ellipsoid_safety_margin(front_pnt,obs_center)
            rear =  obs.ellipsoid_safety_margin(rear_pnt,obs_center)

            front_safety_margin_over_ego_front  = obs.ellipsoid_safety_margin_derivatives(front_pnt,obs_center)
            rear_safety_margin_over_ego_front   = obs.ellipsoid_safety_margin_derivatives(rear_pnt,obs_center)

            ego_front_over_state, ego_rear_over_state = self.Model.get_vehicle_front_and_rear_center_derivatives(yaw)
            front_safety_margin_over_state = front_safety_margin_over_ego_front @ ego_front_over_state
            rear_safety_margin_over_state = rear_safety_margin_over_ego_front @ ego_rear_over_state

            front_dx    = self.exp_barrier_jacobian(front,front_safety_margin_over_state)
            rear_dx     = self.exp_barrier_jacobian(rear,rear_safety_margin_over_state)
            lx += front_dx + rear_dx
        return lx
    def gradient_lu(self, step, state, control):
        return np.zeros(self.N)
    def hessian_lxx(self, step, state, control):
        lxx = np.zeros((self.M,self.M))
        x,y,v,yaw = state
        pos = np.array([x,y])
        front_pnt, rear_pnt = self.Model.get_vehicle_front_and_rear_centers(pos,yaw)

        velo_up_constraint  = self.get_bound_constr(v,config.v_max,"upper")
        velo_low_constraint = self.get_bound_constr(v,config.v_min,"lower")

        velo_up_constraint_dx   = np.array([0.0, 0.0 , 1.0, 0.0])
        velo_low_constraint_dx  = np.array([0.0, 0.0, -1.0, 0.0])

        lxx += self.exp_barrier_hessian(velo_up_constraint,velo_up_constraint_dx)
        lxx += self.exp_barrier_hessian(velo_low_constraint,velo_low_constraint_dx)

        for obs in self.Obstacle_list:
            obs_center = obs.prediction_traj[step][:2]
            front   = obs.ellipsoid_safety_margin(front_pnt,obs_center)
            rear    = obs.ellipsoid_safety_margin(rear_pnt,obs_center)

            front_safety_margin_over_ego_front  = obs.ellipsoid_safety_margin_derivatives(front_pnt,obs_center)
            rear_safety_margin_over_ego_front   = obs.ellipsoid_safety_margin_derivatives(rear_pnt,obs_center)

            ego_front_over_state, ego_rear_over_state = self.Model.get_vehicle_front_and_rear_center_derivatives(yaw)
            front_safety_margin_over_state  = front_safety_margin_over_ego_front @ ego_front_over_state
            rear_safety_margin_over_state   = rear_safety_margin_over_ego_front @ ego_rear_over_state

            front_dxx = self.exp_barrier_hessian(front,front_safety_margin_over_state)
            rear_dxx = self.exp_barrier_hessian(rear,rear_safety_margin_over_state)
            lxx += front_dxx + rear_dxx
        return lxx
    def hessian_luu(self, step, state, control):
        return np.zeros((self.N,self.N))
    def hessian_lxu(self, step, state, control):
        return np.zeros((self.M,self.N))
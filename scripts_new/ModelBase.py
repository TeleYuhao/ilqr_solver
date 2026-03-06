import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List,Dict


class ModelBase:
    def __init__(self, config: Dict):
        self.state_dim      = config["state_dim"]
        self.control_dim    = config["control_dim"]
        self.horizon        = config["horizon"]
    @abstractmethod
    def forward_calculation(self, state, control):
        pass

    @abstractmethod
    def get_jacobian(self, state, control):
        pass

    @abstractmethod
    def get_derivates(self, states, controls, ref_path,collision_constraint):
        pass
    @abstractmethod
    def compute_cost(self, states, controls, ref_path, collision_constraint):
        pass

    @abstractmethod
    def get_center(self,state):
        pass

    @abstractmethod
    def get_reference_state(self,states,ref_path):
        pass

    @abstractmethod
    def init_traj(self, init_state, controls,horizon):
        pass

    def exp_barrier(self,c, q1 = 5.5, q2 = 5.75):
        b = q1 * np.exp(q2 * c)
        return b
    def exp_barrier_derivative_and_Hessian(self,c, c_dot, q1= 5.5, q2 = 5.75):
        b = self.exp_barrier(c, q1, q2)
        b_dot = q2 * b * c_dot
        c_dot_reshaped = c_dot[:, np.newaxis]
        b_ddot = (q2 ** 2) * b * (c_dot_reshaped @ c_dot_reshaped.T)

        return b_dot, b_ddot
    def get_bound_constr(self,var, bound, bound_type='upper'):
        assert bound_type == 'upper' or bound_type == 'lower'
        if bound_type == 'upper':
            return var - bound
        elif bound_type == 'lower':
            return bound - var
    
    

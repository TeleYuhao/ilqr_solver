from cost_base import CostFunc
import numpy as np
class config:
    v_max = 10
    v_min = 0
    a_max = 2.0
    a_min = -2.0
    delta_max = 1.57
    delta_min = -1.57


class ControlConstraint(CostFunc):
    def __init__(self, state_dim, control_dim):
        super().__init__(state_dim, control_dim)

    def value(self, step, state, control):
        x, y, v, yaw = state
        a,delta = control

        acc_up_contraint = self.exp_barrier(self.get_bound_constr(a,config.a_max,"upper"))
        acc_low_contrint = self.exp_barrier(self.get_bound_constr(a,config.a_min,"lower"))

        delta_up_contraint  = self.exp_barrier(self.get_bound_constr(delta,config.delta_max,"upper"))
        delta_low_contraint = self.exp_barrier(self.get_bound_constr(delta,config.delta_min,"lower"))

        
        return acc_up_contraint + acc_low_contrint + delta_up_contraint + delta_low_contraint
    
    def gradient_lx(self, step, state, control):
        return np.zeros(self.M)

    def gradient_lu(self, step, state, control):
        lu = np.zeros(self.N)
        acc, delta = control
        x, y, v, yaw = state
        acc_up_constraint_du = np.array([1,0])
        acc_low_constraint_du = np.array([-1,0])

        delta_up_constraint_du = np.array([0,1])
        delta_low_constraint_du = np.array([0,-1])

        acc_up_constraint = self.get_bound_constr(acc,config.a_max,"upper")
        acc_low_constraint = self.get_bound_constr(acc,config.a_min,"lower")

        delta_up_constraint = self.get_bound_constr(delta,config.delta_max,"upper")
        delta_low_constraint = self.get_bound_constr(delta,config.delta_min,"lower")

        lu += self.exp_barrier_jacobian(acc_up_constraint,acc_up_constraint_du)
        lu += self.exp_barrier_jacobian(acc_low_constraint,acc_low_constraint_du)

        lu += self.exp_barrier_jacobian(delta_up_constraint,delta_up_constraint_du)
        lu += self.exp_barrier_jacobian(delta_low_constraint,delta_low_constraint_du)

        return lu

    def hessian_lxx(self, step, state, control):
        return np.zeros((self.M, self.M))

    def hessian_luu(self, step, state, control):
        luu = np.zeros((self.N,self.N))
        acc, delta = control
        x, y, v, yaw = state
        acc_up_constraint_du = np.array([1,0])
        acc_low_constraint_du = np.array([-1,0])

        delta_up_constraint_du = np.array([0,1])
        delta_low_constraint_du = np.array([0,-1])

        acc_up_constraint = self.get_bound_constr(acc,config.a_max,"upper")
        acc_low_constraint = self.get_bound_constr(acc,config.a_min,"lower")

        delta_up_constraint = self.get_bound_constr(delta,config.delta_max,"upper")
        delta_low_constraint = self.get_bound_constr(delta,config.delta_min,"lower")

        luu += self.exp_barrier_hessian(acc_up_constraint,acc_up_constraint_du)
        luu += self.exp_barrier_hessian(acc_low_constraint,acc_low_constraint_du)

        luu += self.exp_barrier_hessian(delta_up_constraint,delta_up_constraint_du)
        luu += self.exp_barrier_hessian(delta_low_constraint,delta_low_constraint_du)
        return luu
    
    def hessian_lxu(self, step, state, control):
        return np.zeros((self.M,self.N))
    
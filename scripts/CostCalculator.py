from StateConstraint import StateConstraint
from ControlConstraint import ControlConstraint
from state_cost import StateCost
import numpy as np

class CostCalculator:
    def __init__(self,
                 StateCost:StateCost,
                 StateConstraints:StateConstraint,
                 ControlConstraints:ControlConstraint,
                 horizon :int,
                 state_dim:int,
                 control_dim:int):
        self.StateCost = StateCost
        self.StateConstrints = StateConstraints
        self.ControlConstraints = ControlConstraints
        self.horizon = horizon
        self.state_dim = state_dim
        self.control_dim = control_dim
    def CalculateTotalCost(self,states,controls):
        self.StateCost.get_ref_states(states[:,:2])
        state_cost =  self.StateCost.value(0,states[0],np.zeros(2))
        cons_cost = 0
        for i in range(1,self.horizon + 1):
            state_cost += self.StateCost.value(i,states[i],controls[i-1])
            cons_cost += self.StateConstrints.value(i,states[i],states[i-1]) + \
                         self.ControlConstraints.value(i,states[i],controls[i-1]) 

        return cons_cost + state_cost
    def CalculateDerivates(self,states,controls):
        lx = np.zeros((self.horizon + 1,self.state_dim))
        lxx = np.zeros((self.horizon + 1, self.state_dim,self.state_dim))
        lu = np.zeros((self.horizon,self.control_dim))
        luu = np.zeros((self.horizon,self.control_dim,self.control_dim))
        lxu = np.zeros((self.horizon,self.state_dim,self.control_dim))

        lx[0] += self.StateConstrints.gradient_lx(0,states[0],np.zeros(2))
        lxx[0] += self.StateConstrints.hessian_lxx(0,states[0],np.zeros(2))

        lx[0] += self.StateCost.gradient_lx(0,states[0],np.zeros(2))
        lxx[0] += self.StateCost.hessian_lxx(0,states[0],np.zeros(2))

        for i in range(1,self.horizon + 1):
            state = states[i]
            control = controls[i-1]
            lx[i] += self.StateConstrints.gradient_lx(i,state,control)
            lxx[i] += self.StateConstrints.hessian_lxx(i,state,control)

            lx[i] += self.StateCost.gradient_lx(i,state,control)
            lxx[i] += self.StateCost.hessian_lxx(i,state,control)

            lu[i-1] += self.ControlConstraints.gradient_lu(i,state,control)
            luu[i-1] += self.ControlConstraints.hessian_luu(i,state,control)
            lu[i-1] += self.StateCost.gradient_lu(i,state,control)
            luu[i-1] += self.StateCost.hessian_luu(i,state,control)

            # lxu += self.StateCost.hessian_lxu(i,state,control)
            # lxu += self.ControlConstraints.hessian_lxu(i,state,control)
        return lx,lxx,lu,luu,lxu



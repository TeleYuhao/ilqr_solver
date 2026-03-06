from ModelBase import ModelBase
import numpy as np
from typing import Dict
import sys

config = {
    'max_iter': 100,
    'tol': 1e-3,
    'lamb_decay': 0.5,
    'lamb_amplify': 2.0,
    'max_lamb': 1e4,
    'alpha_options': [1., 0.5, 0.25, 0.125, 0.0625],
}

class ILQRCore:
    def __init__(self, model: ModelBase, config: Dict):
        self.model = model
        self.max_iter = config['max_iter']
        self.tol = config['tol']
        self.lamb_decay = config['lamb_decay']
        self.lamb_amplify = config['lamb_amplify']
        self.max_lamb = config['max_lamb']
        self.alpha_options = config['alpha_options']
        self.init_lamb = config['init_lamb']
        self.horizon = model.horizon
        self.state_dim = model.state_dim
        self.control_dim = model.control_dim

    def forwardpass(self, states, controls, d, K, alpha):
        new_u = np.zeros((self.horizon,self.control_dim) )
        new_x = np.zeros((self.horizon + 1, self.state_dim ))
        new_x[0] = states[0]

        for i in range(self.horizon):
            new_u_i = controls[i] + alpha * d[i] + K[i] @ (new_x[i] - states[i])
            new_u[i] = new_u_i
            new_x[i + 1] = self.model.forward_calculation(
                new_x[i], new_u[ i]
            )
        return new_u, new_x
    def backwardpass(self, states, controls, ref_path, obstacles,lamb=1.0):
        l_x,l_u,l_xx,l_uu,l_ux = self.model.get_derivates(states,controls,ref_path,obstacles)
        delt_V = 0
        V_x = l_x[-1]
        V_xx = l_xx[-1]
        d = np.zeros((self.horizon,self.control_dim) )
        K = np.zeros((self.horizon,self.control_dim, self.state_dim))

        regu_I = lamb * np.eye(V_xx.shape[0])

        for i in reversed(range(self.horizon)):
            dfdx, dfdu = self.model.get_jacobian(states[i],controls[i])

            # Q_terms
            Q_x  = l_x[i]  + dfdx.T @ V_x
            Q_u  = l_u[i]  + dfdu.T @ V_x
            Q_xx = l_xx[i] + dfdx.T @ V_xx @ dfdx
            Q_uu = l_uu[i] + dfdu.T @ V_xx @ dfdu
            Q_ux = l_ux[i] + dfdu.T @ V_xx @ dfdx

            # gains
            df_du_regu = dfdu.T @ regu_I
            Q_ux_regu  = Q_ux + df_du_regu @ dfdx
            Q_uu_regu  = Q_uu + df_du_regu @ dfdu
            Q_uu_inv   = np.linalg.inv(Q_uu_regu)

            d[i] = -Q_uu_inv @ Q_u
            K[i] = -Q_uu_inv @ Q_ux_regu

            # Update value function for next time step
            V_x  = Q_x  + K[i].T @ Q_uu @ d[i] + K[i].T @ Q_u + Q_ux.T @ d[i]
            V_xx = Q_xx + K[i].T @ Q_uu @ K[i] + K[i].T @ Q_ux + Q_ux.T @ K[i]

            # expected cost reduction
            delt_V += 0.5 * d[i].T @ Q_uu @ d[i] + d[i].T @ Q_u

            # print(f'Time step {i}, Expected Cost Reduction: {delt_V}')
            # print(f'Vx:\n {V_x}')
            # print(f'Vxx:\n {V_xx}')
            # print(f'------------------')
        return d, K, delt_V
    def iterate(self, states, controls,J, ref_path, obstacles, lamb=1.0):
        d, K, expc_redu = self.backwardpass(states, controls, ref_path, obstacles,lamb)
        iter_effective_flag = False
        new_u, new_x, new_J = \
            np.zeros((self.horizon,self.control_dim) ), np.zeros((self.horizon + 1, self.state_dim)), sys.float_info.max
        for alpha in self.alpha_options:
            new_u, new_x = self.forwardpass(states, controls, d, K, alpha)
            new_J = self.model.compute_cost(new_x, new_u, ref_path, obstacles)

            if new_J < J:
                iter_effective_flag = True
                break

        return new_u, new_x, new_J, iter_effective_flag
    def solve(self, states, controls, ref_path, obstacles):
        # Fix: Use passed controls instead of resetting to zeros
        if controls.shape[0] != self.horizon:
            raise ValueError(f"controls shape {controls.shape} doesn't match horizon {self.horizon}")
        init_u = controls
        init_x = states
        J = self.model.compute_cost(init_x,init_u,ref_path,obstacles)
        u, x = init_u, init_x

        lamb = self.init_lamb

        for itr in range(self.max_iter):
            new_u, new_x, new_J, iter_effective_flag = self.iterate(
                 x, u, J, ref_path, obstacles, lamb)
            print(f'Iteration {itr}, Cost: {new_J}, Lambda: {lamb}, iter_effective_flag: { iter_effective_flag}')
            if iter_effective_flag:
                x = new_x
                u = new_u
                J_old = J
                J = new_J

                if abs(J - J_old) < self.tol:
                    print(f'Tolerance condition satisfied. {itr}')
                    break

                lamb *= self.lamb_decay
            else:
                lamb *= self.lamb_amplify

                if lamb > self.max_lamb:
                    print('Regularization parameter reached the maximum.')
                    break

        return x,u

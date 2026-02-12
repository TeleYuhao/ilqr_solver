import sys
sys.path.append("/home/yuhao/Code/Panda_ilqr/scripts")
from CostCalculator import CostCalculator
import numpy as np
from kinematic_mode import KinematicModel
import sys
HORIZON_LENGTH = 60
DT = 0.1
np.set_printoptions(precision=8, suppress=True, floatmode='fixed')
class ilqr:
    def __init__(self,Model:KinematicModel,CostFunc:CostCalculator):
        self.N = HORIZON_LENGTH
        self.Model = Model
        self.nx = Model.M
        self.nu = Model.N
        self.dt = DT
        self.CostFunc = CostFunc
        self.max_iter = 50
        self.init_lamb = 20
        self.lamb_decay = 0.7
        self.lamb_amplify = 2.0
        self.max_lamb = 10000.0
        self.alpha_options = [1., 0.5, 0.25, 0.125, 0.0625]
        self.tol = 0.001
    
    def backward_pass(self,u, x, lamb):
        self.CostFunc.StateCost.get_ref_states(x[:,:2])
        l_x,l_xx,l_u,l_uu,l_xu = self.CostFunc.CalculateDerivates(x,u)
        delt_V = 0
        V_x = l_x[-1]
        V_xx = l_xx[-1]
        d = np.zeros((self.N,self.nu) )
        K = np.zeros((self.N,self.nu, self.nx))

        regu_I = lamb * np.eye(V_xx.shape[0])

        for i in reversed(range(self.N)):

            dfdx = self.Model.gradient_fx(x[i],u[i])
            dfdu = self.Model.gradient_fu(x[i],u[i])
            # Q_terms
            Q_x = l_x[i] + dfdx.T @ V_x
            Q_u = l_u[i] + dfdu.T @ V_x
            Q_xx = l_xx[i] + dfdx.T @ V_xx @ dfdx
            Q_uu = l_uu[i] + dfdu.T @ V_xx @ dfdu
            Q_ux = l_xu[i].T + dfdu.T @ V_xx @ dfdx

            # gains
            df_du_regu = dfdu.T @ regu_I
            Q_ux_regu = Q_ux + df_du_regu @ dfdx
            Q_uu_regu = Q_uu + df_du_regu @ dfdu
            Q_uu_inv = np.linalg.inv(Q_uu_regu)

            d[i] = -Q_uu_inv @ Q_u
            K[i] = -Q_uu_inv @ Q_ux_regu

            # Update value function for next time step
            V_x = Q_x + K[i].T @ Q_uu @ d[i] + K[i].T @ Q_u + Q_ux.T @ d[i]
            V_xx = Q_xx + K[i].T @ Q_uu @ K[i] + K[i].T @ Q_ux + Q_ux.T @ K[i]

            # expected cost reduction
            delt_V += 0.5 * d[i].T @ Q_uu @ d[i] + d[i].T @ Q_u

            # print(f'Time step {i}, Expected Cost Reduction: {delt_V}')

        return d, K, delt_V
    def forward_pass(self, u, x, d, K, alpha):
        new_u = np.zeros((self.N,self.nu))
        new_x = np.zeros((self.N + 1, self.nx ))
        new_x[0] = x[0]

        for i in range(self.N):
            new_u_i = u[i] + alpha * d[i] + K[i] @ (new_x[i] - x[i])
            new_u[i] = new_u_i
            new_x[i + 1] = self.Model.forward_calculation(
                new_x[i], new_u[ i]
            )

        return new_u, new_x
    
    def iter(self,u, x, J,lamb):
        d, K, expc_redu = self.backward_pass(u, x,lamb)
        iter_effective_flag = False
        new_u, new_x, new_J = \
            np.zeros((self.nu, self.N)), np.zeros((self.nx, self.N + 1)), sys.float_info.max
        for alpha in self.alpha_options:
            new_u, new_x = self.forward_pass(u, x, d, K, alpha)
            new_J = self.CostFunc.CalculateTotalCost(new_x, new_u )

            if new_J < J:
                iter_effective_flag = True
                break

        return new_u, new_x, new_J, iter_effective_flag
    def solve(self, x0):
        init_u = np.zeros((self.N,self.nu))
        init_x = self.Model.init_traj(x0,init_u)
        J = self.CostFunc.CalculateTotalCost(init_x, init_u)
        u, x = init_u, init_x

        lamb = self.init_lamb

        for itr in range(self.max_iter):
            new_u, new_x, new_J, iter_effective_flag = self.iter(
                u, x, J, lamb)
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

        return u, x
